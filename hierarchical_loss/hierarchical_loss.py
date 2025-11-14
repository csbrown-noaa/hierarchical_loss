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
    hierarchical_expanded_targets = expand_target_hierarchy(targets, hierarchy_index)
    # log(1 - s) = log(1 - P(marginal))
    #            = log(1 - exp(log(P(marginal))))
    hierarchical_summed_log1sigmoids = log1mexp(hierarchical_summed_logsigmoids)
    # Standard BCE loss: -[t*log(s) + (1-t)*log(1-s)]
    return -(
      (hierarchical_expanded_targets * hierarchical_summed_logsigmoids) 
      + (1 - hierarchical_expanded_targets) * hierarchical_summed_log1sigmoids
    )