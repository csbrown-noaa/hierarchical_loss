import torch
from .hierarchy_tensor_utils import accumulate_hierarchy, expand_target_hierarchy
from .utils import log1mexp

def hierarchical_bce(
    pred: torch.Tensor, targets: torch.Tensor, hierarchy_index: torch.Tensor
) -> torch.Tensor:
    """Computes a hierarchical cross-entropy loss.

    This function interprets the raw `pred` logits as representing the
    log-odds of the *conditional* probability `P(category | parent)`.
    It then calculates the *marginal* probability `P(category)` by
    multiplying all conditional probabilities along the ancestor path.

    This multiplication is done in log-space for numerical stability.

    The loss is a variant of binary cross-entropy, calculated as:
    `-[t * log(s) + (1-t) * log(1-s)]`
    where:
    - `t` is the expanded target (1 for the node and all ancestors, 0 otherwise).
    - `s` is the *marginal* probability, `P(category)`.
    - `log(s)` is `sum(log(P(ancestor | parent_of_ancestor)))`, which is
      computed by `accumulate_hierarchy`.

    Parameters
    ----------
    pred : torch.Tensor
        The raw logit predictions from the model, with shape `(B, D, N)`,
        where `N` is the number of classes.
    targets : torch.Tensor
        The one-hot target tensor, with shape `(B, D, N)`.
    hierarchy_index : torch.Tensor
        An int tensor of shape `(N, M)` mapping each node `i` to its
        ancestral path `[i, parent, grandparent, ...]`.

    Returns
    -------
    torch.Tensor
        A tensor of the same shape as `pred` containing the loss
        value for each prediction.
    """
    # logsigmoids = log(P(category | parent))
    logsigmoids = torch.nn.functional.logsigmoid(pred)
    # This computes log(P(marginal)) = sum(log(P(c | p))) (Bayes' rule in log space)
    hierarchical_summed_logsigmoids = accumulate_hierarchy(logsigmoids, hierarchy_index, torch.sum, 0.)
    # Expand target to be 1 for the node and all its ancestors
    #targets = expand_target_hierarchy(targets, hierarchy_index)
    # log(1 - s) = log(1 - P(marginal))
    #            = log(1 - exp(log(P(marginal))))
    hierarchical_summed_log1sigmoids = log1mexp(hierarchical_summed_logsigmoids)
    # Standard BCE loss: -[t*log(s) + (1-t)*log(1-s)]
    return -(
      (targets * hierarchical_summed_logsigmoids) 
      + (1 - targets) * hierarchical_summed_log1sigmoids
    )

def hierarchical_conditional_bce(
    pred: torch.Tensor,
    target_indices: torch.Tensor,
    ancestor_mask: torch.Tensor,
    ancestor_sibling_mask: torch.Tensor
) -> torch.Tensor:
    """Computes the hierarchy-normalized conditional BCE loss per sample.

    This function calculates a specialized Binary Cross Entropy (BCE) loss that
    enforces structural consistency in a hierarchy. For each target, it identifies
    the specific set of nodes that *must* be True (ancestors) and the specific
    set of nodes that *must* be False (ancestor siblings/uncles). All other
    nodes are ignored.

    Crucially, this function performs **Structural Normalization**. It divides
    the summed loss for a sample by the number of active nodes (depth + width)
    relevant to that specific sample. This ensures that deep, complex classes
    do not generate larger gradients than shallow, simple classes solely due to
    having more ancestors or uncles.

    Parameters
    ----------
    pred : torch.Tensor
        The raw logits from the model with shape `(..., N)`, where `N` is the
        number of classes.
    target_indices : torch.Tensor
        LongTensor of class indices with shape `(...)`. These indices correspond
        to the leaf (or most specific) class of the target.
    ancestor_mask : torch.Tensor
        Boolean tensor of shape `(N, N)`. `mask[i, j]` is True if `j` is an
        ancestor of `i`.
    ancestor_sibling_mask : torch.Tensor
        Boolean tensor of shape `(N, N)`. `mask[i, j]` is True if `j` is an
        ancestor sibling of `i` (a negative target).

    Returns
    -------
    torch.Tensor
        A tensor of shape `(...)` (matching `target_indices`) containing the
        average BCE loss per decision node.

    Examples
    --------
    >>> # Hierarchy: 0(root) -> 1, 2. Target is Node 1.
    >>> # Ancestors of 1: {1, 0} (Positives)
    >>> # Ancestor Siblings of 1: {2} (Negatives) -> Node 2 is sibling of 1
    >>>
    >>> # 1. Setup Masks (N=3)
    >>> # Ancestor Mask: 0->{0}, 1->{1,0}, 2->{2,0}
    >>> anc_mask = torch.tensor([
    ...     [1, 0, 0], [1, 1, 0], [1, 0, 1]
    ... ], dtype=torch.bool)
    >>> # Sibling Mask: 1's sibling is 2, 2's sibling is 1
    >>> sib_mask = torch.tensor([
    ...     [0, 0, 0], [0, 0, 1], [0, 1, 0]
    ... ], dtype=torch.bool)
    >>>
    >>> # 2. Mock Predictions (Batch=1)
    >>> # Case A: Perfect Prediction.
    >>> # Node 0(High), Node 1(High), Node 2(Low)
    >>> pred_good = torch.tensor([[10.0, 10.0, -10.0]])
    >>> target_idx = torch.tensor([1])
    >>>
    >>> loss_good = hierarchical_conditional_bce(pred_good, target_idx, anc_mask, sib_mask)
    >>> round(loss_good.item(), 5)
    5e-05
    >>>
    >>> # Case B: Wrong Prediction (High score on sibling Node 2)
    >>> # We expect penalty from Node 2 (should be 0) and Node 1 (should be 1)
    >>> pred_bad = torch.tensor([[10.0, -10.0, 10.0]])
    >>> loss_bad = hierarchical_conditional_bce(pred_bad, target_idx, anc_mask, sib_mask)
    >>> # Loss comes from 3 active nodes (0, 1, 2).
    >>> # Node 0 is correct (~0 loss). Node 1 and 2 are wrong (~10 loss each).
    >>> # Average loss ~ 20 / 3 = 6.66
    >>> round(loss_bad.item(), 2)
    6.67
    """
    # 1. Lookup Masks based on target indices
    # Shape expands from (...) to (..., N)
    pos_mask = ancestor_mask[target_indices]
    neg_mask = ancestor_sibling_mask[target_indices]

    # 2. Compute component-wise BCE
    # We use -logsigmoid(x) for log(P(x))
    # We use -logsigmoid(-x) for log(1 - P(x))
    loss_pos = -torch.nn.functional.logsigmoid(pred)
    loss_neg = -torch.nn.functional.logsigmoid(-pred)

    # 3. Apply masks and sum structural loss
    # We sum across the class dimension (dim=-1) to get total loss per sample
    total_structure_loss = (
        (loss_pos * pos_mask).sum(dim=-1) +
        (loss_neg * neg_mask).sum(dim=-1)
    )

    # 4. Normalize by structure size (Depth + Width)
    # Count how many nodes contributed to the loss for each sample
    active_nodes = pos_mask.sum(dim=-1)# + neg_mask.sum(dim=-1)

    # Clamp to 1.0 to ensure numerical stability (e.g., if index is -1/ignored)
    structure_scale = active_nodes.float().clamp(min=1.0)
    #structure_scale = 1

    return total_structure_loss / structure_scale


def hierarchical_conditional_bce_soft_root(
    pred: torch.Tensor,
    target_scores: torch.Tensor,
    target_indices: torch.Tensor,
    ancestor_mask: torch.Tensor,
    ancestor_sibling_mask: torch.Tensor,
    root_mask: torch.Tensor,
    clamp_val: float = 80.0
) -> torch.Tensor:
    """
    Computes Hierarchical Loss with Soft Targets for Roots and Hard Targets for the Branch.
    
    1. Root Nodes (e.g., Mammalia): Trained with Soft Target = Assigner Score (e.g., 0.1).
       - Enforces calibration and background suppression.
    2. Descendants (e.g., Felidae, Felis): Trained with Hard Target = 1.0.
       - Enforces conditional classification.
    3. Uncles (e.g., Canidae): Trained with Hard Target = 0.0.
       - Enforces rejection of incorrect branches.

    Parameters
    ----------
    pred : torch.Tensor
        Logits (B, Anchors, N).
    target_scores : torch.Tensor
        Assigner quality scores (B, Anchors). Used as target for Root nodes.
    target_indices : torch.Tensor
        Leaf class indices (B, Anchors).
    ancestor_mask : torch.Tensor
        (N, N) boolean mask of ancestors.
    ancestor_sibling_mask : torch.Tensor
        (N, N) boolean mask of uncles.
    root_mask : torch.Tensor
        (N,) boolean mask where True = Root Node.

    Returns
    -------
    torch.Tensor
        Summed loss per anchor (B, Anchors).
    """
    # 1. Lookup and Expand Masks
    # Shape: (B, Anchors, N)
    pos_mask = ancestor_mask[target_indices]
    neg_mask = ancestor_sibling_mask[target_indices]
    
    # Broadcast root_mask to (1, 1, N) for logical operations
    root_mask_expanded = root_mask.view(1, 1, -1)

    # 2. Separate Positives into Roots vs Descendants
    # Intersection: Ancestors that are ALSO roots
    is_active_root = pos_mask & root_mask_expanded
    # Difference: Ancestors that are NOT roots
    is_active_descendant = pos_mask & (~root_mask_expanded)

    # 3. Compute Losses
    # Safety Clamp
    #pred_clamped = pred.clamp(-clamp_val, clamp_val)
    pred_clamped = pred

    # A. Descendants: Hard Target 1.0 (Optimization)
    #    Loss = -log(sigmoid(x))
    loss_descendants = -F.logsigmoid(pred_clamped) * is_active_descendant

    # B. Uncles: Hard Target 0.0 (Rejection)
    #    Loss = -log(1 - sigmoid(x))
    loss_uncles = -F.logsigmoid(-pred_clamped) * neg_mask

    # C. Roots: Soft Target (Calibration)
    #    Target is the assigner score (e.g., 0.1)
    #    We expand scores to (B, Anchors, 1) to broadcast against (B, Anchors, N)
    scores_expanded = target_scores.unsqueeze(-1)
    
    #    BCEWithLogits(x, target)
    loss_roots_raw = F.binary_cross_entropy_with_logits(
        pred_clamped, scores_expanded, reduction='none'
    )
    #    Only apply where is_active_root is True
    loss_roots = loss_roots_raw * is_active_root

    # 4. Sum All Components
    # We sum across the class dimension (dim=-1)
    total_loss = (loss_descendants + loss_uncles + loss_roots).sum(dim=-1)

    return total_loss
