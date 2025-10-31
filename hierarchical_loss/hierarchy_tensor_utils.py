import torch
from .tree_utils import *
from typing import Callable

def set_indices(index: int, parent_index: int | None, tensor: torch.Tensor) -> int:
    """A helper function for `preorder_apply` to build an ancestor path tensor.

    This function populates a single row of the `tensor` (the row specified
    by `index`). It sets the first element of the row to `index` itself.
    If a `parent_index` is provided, it copies the parent's ancestor path
    (its row, excluding the last element) into the current node's row
    (starting from the second element).

    This creates the desired row format: [node_id, parent_id, grand_parent_id, ...].

    It is designed to be used with `tree_utils.preorder_apply`, where:
    - `index` is the `node`
    - `parent_index` is the `parent_result` (the return value from the
      parent's call, which is the parent's index)
    - `tensor` is the `*args`

    Parameters
    ----------
    index : int
        The node ID, which corresponds to the row index in the tensor.
    parent_index : int or None
        The ID of the parent node, or None if the node is a root.
    tensor : torch.Tensor
        The 2D tensor being populated with ancestor paths.

    Returns
    -------
    int
        The `index` of the current node, to be passed as the
        `parent_index` to its children during the pre-order traversal.
    """
    tensor[index, 0] = index
    if parent_index is not None:
        tensor[index, 1:] = tensor[parent_index, :-1]
    return index


def build_parent_tensor(
    tree: dict[int, int], device: torch.device | str | None = None
) -> torch.Tensor:
    """Converts a {child: parent} dictionary tree into a 1D parent tensor.

    This function creates a 1D tensor where the value at each index `i`
    is the ID of that node's parent. Nodes that are not children (i.e.,
    roots) will have a value of -1.

    The size of the tensor is determined by the maximum node ID present
    in the tree (in either keys or values).

    Parameters
    ----------
    tree : dict[int, int]
        A tree in {child: parent} format. Node IDs are assumed to be
        non-negative integers.
    device : torch.device | str | None, optional
        The desired device for the output tensor. If `None`, uses the
        default PyTorch device. By default `None`.

    Returns
    -------
    torch.Tensor
        A 1D tensor of shape `(C,)`, where `C` is `max(all_node_ids) + 1`.
        `parent_tensor[i]` contains the ID of the parent of node `i`,
        or -1 if `i` is a root or not in the hierarchy.

    Examples
    --------
    >>> tree = {0: 1, 1: 2, 3: 2, 5: 6}
    >>> # Max node ID is 6, so tensor size is 7
    >>> build_parent_tensor(tree)
    tensor([ 1,  2, -1,  2, -1,  6, -1])
    """
    nodes = set(tree.keys()) | set(tree.values())
    C = max(nodes) + 1

    parent_tensor = torch.full((C,), -1, dtype=torch.long, device=device)

    for child, parent in tree.items():
        parent_tensor[child] = parent

    return parent_tensor


def build_hierarchy_sibling_mask(
    parent_tensor: torch.Tensor, device: torch.device | str | None = None
) -> torch.Tensor:
    """Creates a boolean mask identifying sibling groups from a parent tensor.

    This function is used to prepare a mask for `utils.logsumexp_over_siblings`.
    It takes the 1D parent tensor (where `parent_tensor[i] = parent_id`)
    and creates a 2D mask.

    Each column `g` in the mask represents a unique sibling group (i.e., a
    unique parent, including -1 for the root group). A node `i` will have
    `True` in column `g` if its parent is the parent corresponding to
    sibling group `g`.

    Parameters
    ----------
    parent_tensor : torch.Tensor
        A 1D tensor of shape `(C,)`, where `C` is the number of classes.
        `parent_tensor[i]` contains the integer ID of the parent of node `i`,
        or -1 for root nodes.  See the `build_parent_tensor` function.
    device : torch.device | str | None, optional
        The desired device for the output tensor. If `None`, uses the
        default PyTorch device. By default `None`.

    Returns
    -------
    torch.Tensor
        A 2D boolean tensor of shape `(C, G)`, where `G` is the number of
        unique parent groups (including roots). `mask[i, g]` is `True`
        if node `i` belongs to sibling group `g`.

    Examples
    --------
    >>> # Node parents: 0->1, 1->2, 2->-1, 3->2, 4->-1, 5->6, 6->-1
    >>> parent_tensor = torch.tensor([ 1,  2, -1,  2, -1,  6, -1])
    >>> # Unique parents (groups): -1, 1, 2, 6
    >>> build_hierarchy_sibling_mask(parent_tensor)
    tensor([[False,  True, False, False],
            [False, False,  True, False],
            [ True, False, False, False],
            [False, False,  True, False],
            [ True, False, False, False],
            [False, False, False,  True],
            [ True, False, False, False]])
    """
    C = parent_tensor.shape[0]

    # Identify all unique parents (which uniquely define child groups), including -1 for roots
    unique_parents, inverse_indices = torch.unique(parent_tensor, return_inverse=True)
    G = len(unique_parents)

    sibling_mask = torch.zeros(C, G, dtype=torch.bool, device=device)

    # Assign each node to the column of its parent group
    sibling_mask[torch.arange(C), inverse_indices] = True

    return sibling_mask


def build_hierarchy_index_tensor(
    hierarchy: dict[int, int], device: torch.device | str | None = None
) -> torch.Tensor:
    """Creates a 2D tensor mapping each node to its full ancestor path.

    This function translates a {child: parent} dictionary hierarchy into a
    2D tensor. The hierarchy MUST BE DENSE, in the sense that the keys and values
    must run from 0 to C-1 where C is the number of nodes.
    Each row `i` of the tensor corresponds to node `i`. The
    row contains the full ancestor path starting with the node itself:
    `[node_id, parent_id, grandparent_id, ..., root_id]`.

    The paths are right-padded with -1 to the length of the longest
    ancestor path in the hierarchy.

    This tensor is used as an index for hierarchical accumulation operations.

    Parameters
    ----------
    hierarchy : dict[int, int]
        A tree in {child: parent} format. Node IDs must be non-negative
        integers that can be used as tensor indices.
    device : torch.device | str | None, optional
        The desired device for the output tensor. If `None`, uses the
        default PyTorch device. By default `None`.

    Returns
    -------
    torch.Tensor
        A 2D tensor of shape `(C, M)`, where `M` is the maximum hierarchy
        depth and `C` is the number of categories (nodes in the tree).
        `tensor[i]` contains the ancestor path for node `i`, padded with -1.

    Examples
    --------
    >>> hierarchy = {0: 1, 1: 2, 3: 4}
    >>> # Nodes found: {0, 1, 2, 3, 4} -> len=5
    >>> # Max depth: 3 (for node 0)
    >>> build_hierarchy_index_tensor(hierarchy)
    tensor([[ 0,  1,  2],
            [ 1,  2, -1],
            [ 2, -1, -1],
            [ 3,  4, -1],
            [ 4, -1, -1]], dtype=torch.int32)
    """
    lens = get_ancestor_chain_lens(hierarchy)
    index_tensor = torch.full((len(lens), max(lens.values())), -1, dtype=torch.int32, device=device)
    preorder_apply(hierarchy, set_indices, index_tensor)
    return index_tensor


def accumulate_hierarchy(
    predictions: torch.Tensor,
    hierarchy_index: torch.Tensor,
    reduce_op: Callable[[torch.Tensor, int], torch.Tensor],
    identity_value: float | int,
) -> torch.Tensor:
    """Performs a reduce operation along a hierarchical structure.

    This function applies a reduction operation (e.g., `torch.sum`,
    `torch.max`) along each ancestral path in a hierarchy. The implementation
    is fully vectorized. It first gathers the initial values for all
    nodes along each path, replaces padded values with the `identity_value`,
    and then applies the `reduce_op` along the path dimension.

    Parameters
    ----------
    predictions : torch.Tensor
        A tensor of shape `[B, D, N]`, where `B` is the batch size, `D` is the
        number of detections, and `N` is the number of classes.
    hierarchy_index : torch.Tensor
        An int tensor of shape `[N, M]` encoding the hierarchy, where `N` is the
        number of classes and `M` is the maximum hierarchy depth. Each row `i`
        contains the path from node `i` to its root. Parent node IDs are to
        the right of child node IDs. A value of -1 is used for padding.
    reduce_op : Callable[[torch.Tensor, int], torch.Tensor]
        A function that performs a reduction operation along a dimension,
        such as `torch.sum` or `torch.max`. It must accept a tensor
        and a `dim` argument, and return a tensor.
    identity_value : float | int
        The identity value for the reduction operation. For example,
        `0.0` for `torch.sum` or `-torch.inf` for `torch.max`.

    Returns
    -------
    torch.Tensor
        A new tensor with the same shape as `predictions` (but with the
        last dimension, M, reduced) containing the aggregated values
        along each path.

    Examples
    --------
    >>> hierarchy_index = torch.tensor([
    ...     [ 0,  1,  2],
    ...     [ 1,  2, -1],
    ...     [ 2, -1, -1],
    ...     [ 3,  4, -1],
    ...     [ 4, -1, -1]
    ... ], dtype=torch.int64)
    >>> # Predictions for 5 classes: [0., 10., 20., 30., 40.]
    >>> predictions = torch.arange(0, 50, 10, dtype=torch.float32).view(1, 1, 5)
    >>>
    >>> # Example 1: Hierarchical Sum
    >>> # Path 0: [0, 1, 2] -> 0. + 10. + 20. = 30.
    >>> # Path 1: [1, 2]   -> 10. + 20. = 30.
    >>> # Path 2: [2]      -> 20. = 20.
    >>> # Path 3: [3, 4]   -> 30. + 40. = 70.
    >>> # Path 4: [4]      -> 40. = 40.
    >>> sum_preds = accumulate_hierarchy(predictions, hierarchy_index, torch.sum, 0.0)
    >>> print(sum_preds.squeeze())
    tensor([30., 30., 20., 70., 40.])
    >>>
    >>> # Example 2: Hierarchical Max
    >>> # Path 0: [0, 1, 2] -> max(0., 10., 20.) = 20.
    >>> # Path 1: [1, 2]   -> max(10., 20.) = 20.
    >>> # Path 2: [2]      -> max(20.) = 20.
    >>> # Path 3: [3, 4]   -> max(30., 40.) = 40.
    >>> # Path 4: [4]      -> max(40.) = 40.
    >>> max_op = lambda x, dim: torch.max(x, dim=dim).values
    >>> max_preds = accumulate_hierarchy(predictions, hierarchy_index, max_op, -torch.inf)
    >>> print(max_preds.squeeze())
    tensor([20., 20., 20., 40., 40.])
    """
    B, D, N = predictions.shape
    M = hierarchy_index.shape[1]

    # 1. GATHER: Collect prediction values for each node in each path.
    # Create a mask for valid indices (non -1)
    valid_mask = hierarchy_index != -1

    # Create a "safe" index tensor to prevent out-of-bounds errors from -1.
    # We replace -1 with a valid index (e.g., 0) and will zero out its
    # contribution later using the mask.
    safe_index = hierarchy_index.masked_fill(~valid_mask, 0)

    # Use advanced indexing to gather values. `predictions[:, :, safe_index]`
    # creates a tensor of shape [B, D, N, M].
    path_values = predictions[:, :, safe_index]

    # Replace the invalid, padded values with the appropriate identity value.
    # The valid_mask broadcasts from [N, M] to [B, D, N, M].
    path_values = torch.where(
        valid_mask,
        path_values,
        identity_value
    )

    # 2. ACCUMULATE: Apply the reduction operation along the path dimension.
    final_values = reduce_op(path_values, -1)

    return final_values

def hierarchically_index_flat_scores(
    flat_scores: torch.Tensor,
    target_indices: torch.Tensor,
    hierarchy_index_tensor: torch.Tensor,
    hierarchy_mask: torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Gathers scores from a flat score tensor along specified hierarchical paths.

    This function takes a batch of flat scores (`B, P, C`) and a batch of
    target category indices (`B, P`). For each target index, it looks up
    the full ancestral path in `hierarchy_index_tensor` (`C, H`) and
    gathers the corresponding scores from `flat_scores`.

    It then applies the `hierarchy_mask` to the gathered scores, zeroing
    out entries where the mask is `True`.

    Parameters
    ----------
    flat_scores : torch.Tensor
        A tensor of flat scores with shape `(B, P, C)`, where `B` is
        batch size, `P` is number of proposals, and `C` is number
        of categories.
    target_indices : torch.Tensor
        A long tensor of shape `(B, P)` containing the leaf category
        index for each proposal.
    hierarchy_index_tensor : torch.Tensor
        A long tensor of shape `(C, H)` mapping each category `c` to its
        ancestral path. `H` is the max hierarchy depth.
    hierarchy_mask : torch.Tensor
        A boolean **invalidity** mask of shape `(C, H)`. `True` indicates
        an invalid entry (e.g., padding) that should be zeroed out.
        This mask is indexed by `target_indices` and applied to the
        gathered scores.
    device : torch.device | str | None, optional
        The desired device for `torch.arange`. If `None`, uses the
        default PyTorch device. By default `None`.

    Returns
    -------
    torch.Tensor
        A tensor of shape `(B, P, H)` containing the gathered scores
        along each target's ancestral path, after masking.

    Examples
    --------
    >>> import torch
    >>> # B=1, P=1, C=2
    >>> flat_scores = torch.tensor([[[10., 20.]]])
    >>> # Target index is 0
    >>> target_indices = torch.tensor([[0]])
    >>> # C=2, H=3. Path 0 is [0, 1, -1]
    >>> hierarchy_index_tensor = torch.tensor([[0, 1, -1], [1, -1, -1]], dtype=torch.long)
    >>> # Create an invalidity mask (True where path is -1)
    >>> invalidity_mask = (hierarchy_index_tensor == -1)
    >>> print(invalidity_mask)
    tensor([[False, False,  True],
            [False,  True,  True]])
    >>>
    >>> # The function will gather scores for path [0, 1, -1] -> [10., 20., 10.]
    >>> # (Note: -1 safely indexes 0)
    >>> # It will apply the mask for index 0: [False, False, True]
    >>> # Result: [10., 20., 0.]
    >>> hierarchically_index_flat_scores(
    ...     flat_scores, target_indices, hierarchy_index_tensor, invalidity_mask
    ... )
    tensor([[[10., 20.,  0.]]])
    """
    batch_size, n_proposals, n_categories = flat_scores.shape
    hierarchy_size = hierarchy_index_tensor.shape[1]

    hierarchy_indices = hierarchy_index_tensor[target_indices]
    flat_mask = hierarchy_mask[target_indices]

    # Construct batch indices
    batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1, 1).expand(batch_size, n_proposals, hierarchy_size) # (B, N, H)
    proposal_indices = torch.arange(n_proposals, device=device).view(1, n_proposals, 1).expand(batch_size, n_proposals, hierarchy_size) # (B, N, H)

    gathered_scores = flat_scores[batch_indices, proposal_indices, hierarchy_indices]  # (B, N, H)

    # Mask out invalid entries
    masked_scores = gathered_scores.masked_fill(flat_mask, 0.)

    return masked_scores


def expand_target_hierarchy(
    target: torch.Tensor, hierarchy_index: torch.Tensor
) -> torch.Tensor:
    """Expands a one-hot target tensor up the hierarchy.

    This function takes a target tensor that is "one-hot" along the class
    dimension (i.e., contains a single non-zero value) and propagates that
    value to all ancestors of the target class. The implementation is fully
    vectorized.

    Parameters
    ----------
    target : torch.Tensor
        A tensor of shape `[B, D, N]`, where `B` is the batch size, `D` is the
        number of detections, and `N` is the number of classes. It is assumed
        to be one-hot along the last dimension.
    hierarchy_index : torch.Tensor
        An int tensor of shape `[N, M]` encoding the hierarchy, where `N` is the
        number of classes and `M` is the maximum hierarchy depth. Each row `i`
        contains the path from node `i` to its root.

    Returns
    -------
    torch.Tensor
        A new tensor with the same shape as `target` where the target value has
        been propagated up the hierarchy.

    Examples
    --------
    >>> import torch
    >>> hierarchy_index = torch.tensor([
    ...     [ 0,  1,  2],
    ...     [ 1,  2, -1],
    ...     [ 2, -1, -1],
    ...     [ 3,  4, -1],
    ...     [ 4, -1, -1]
    ... ], dtype=torch.int64)
    >>> # Target is one-hot at index 0
    >>> target = torch.tensor([0.4, 0., 0., 0., 0.]).view(1, 1, 5)
    >>> expanded_target = expand_target_hierarchy(target, hierarchy_index)
    >>> print(expanded_target.squeeze())
    tensor([0.4000, 0.4000, 0.4000, 0.0000, 0.0000])
    >>> target = torch.tensor([0., 0., 0., 0.3, 0.]).view(1, 1, 5)
    >>> expanded_target = expand_target_hierarchy(target, hierarchy_index)
    >>> print(expanded_target.squeeze())
    tensor([0.0000, 0.0000, 0.0000, 0.3000, 0.3000])
    """
    M = hierarchy_index.shape[1]

    # Find the single non-zero value and its index in the target tensor.
    hot_value, hot_index = torch.max(target, dim=-1)

    # Gather the ancestral paths corresponding to the hot indices.
    # The shape will be [B, D, M].
    paths = hierarchy_index[hot_index]

    # Create a mask for valid indices (non -1) to handle padded paths.
    valid_mask = paths != -1

    # Create a "safe" index tensor to prevent out-of-bounds errors from -1.
    # We replace -1 with a valid index (e.g., 0) and will zero out its
    # contribution later using a masked source.
    safe_paths = paths.masked_fill(~valid_mask, 0)
    safe_paths_ints = safe_paths.to(torch.int64)

    # Prepare the source tensor for the scatter operation.
    # It should have the same value (`hot_value`) for all valid path members.
    src_values = hot_value.unsqueeze(-1).expand(-1, -1, M)
    masked_src = src_values * valid_mask.to(src_values.dtype)

    # Create an output tensor and scatter the hot value into all ancestral positions.
    expanded_target = torch.zeros_like(target)
    expanded_target.scatter_(dim=-1, index=safe_paths_ints, src=masked_src)

    return expanded_target
