from collections.abc import Hashable
import torch
import numpy as np

def log_matrix(m: np.ndarray | torch.Tensor) -> str:
    """Formats a 2D array or tensor into a human-readable string.

    Useful for logging or debugging. Each row is prefixed with its index,
    and values are formatted to 4 decimal places.

    Parameters
    ----------
    m : np.ndarray | torch.Tensor
        The 2D array or tensor to format. Must support `.shape[0]`,
        iteration, and `.tolist()`.

    Returns
    -------
    str
        A multi-line string representation of the matrix.

    Examples
    --------
    >>> import torch
    >>> t = torch.tensor([[1.0, 0.5], [0.25, 0.125]])
    >>> print(log_matrix(t))
    0000: 1.0000, 0.5000
    0001: 0.2500, 0.1250
    >>> import numpy as np
    >>> n = np.array([[0.1, 0.2], [0.3, 0.4]])
    >>> print(log_matrix(n))
    0000: 0.1000, 0.2000
    0001: 0.3000, 0.4000
    """
    formatted_lines = []
    for i in range(m.shape[0]):
        vec = m[i]
        line = f"{i:04d}: " + ", ".join(f"{x:.4f}" for x in vec.tolist())
        formatted_lines.append(line)
    return "\n".join(formatted_lines)


def argmax_from_subset(scores: np.ndarray | torch.Tensor, indices: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Finds the argmax from a subset of indices.

    This function is type-agnostic and works for both NumPy arrays
    and PyTorch tensors.

    The core operation is performed on the last dimension of the tensor.

    Parameters
    ----------
    scores : np.ndarray | torch.Tensor
        Tensor/array of scores with shape (*D, N).
    indices : np.ndarray | torch.Tensor
        A 1D Tensor/array of viable indices with shape (K,).

    Returns
    ----------
    np.ndarray | torch.Tensor
        A tensor/array of shape (*D) containing the argmax index.
        The return type will match the type of `indices`.

    Examples
    ----------
    >>> scores_np = np.array([
    ...     [10, 20, 30, 5, 40],
    ...     [99, 88, 77, 66, 55]
    ... ])
    >>> indices_np = np.array([0, 2, 4])
    >>> argmax_from_subset(scores_np, indices_np)
    array([4, 0])
    >>>
    >>> scores_pt = torch.tensor(scores_np)
    >>> indices_pt = torch.tensor(indices_np)
    >>> argmax_from_subset(scores_pt, indices_pt)
    tensor([4, 0])
    """
    subset_scores = scores[..., indices]

    local_argmax_indices = subset_scores.argmax(axis=-1)

    return indices[local_argmax_indices]


def logsumexp_over_siblings(flat_scores: torch.Tensor, sibling_mask: torch.Tensor) -> torch.Tensor:
    '''Computes logsumexp over sibling groups for each category.

    This function calculates the logsumexp of scores for all siblings
    within each group, and then populates the result for each category
    belonging to that group.

    Parameters
    ----------
    flat_scores: tensor (BxC)
        raw scores for each category, batch-wise
    sibling_mask: tensor (CxG)
        a mask where sibling_mask[i,j] == sibling_mask[k,j] == 1 iff i and k are siblings.  Must be boolean.

    Returns
    -------
    logsumexp: tensor (BxC)
        the logsumexp over all of the siblings of each category.  logsumexp[i,j] == logsumexp[i,k] if j,k are siblings.

    Examples
    --------
    >>> # Example 0: Normal operation
    >>> flat_scores = torch.tensor([
    ...     [0.1, 0.5, 2.0, 3.0],  # Batch 1
    ...     [0.8, 0.2, 1.5, 1.0]   # Batch 2
    ... ])
    >>> sibling_mask = torch.tensor([
    ...     [True, False],
    ...     [True, False],
    ...     [False, True],
    ...     [False, True]
    ... ], dtype=torch.bool)
    >>> result = logsumexp_over_siblings(flat_scores, sibling_mask)
    >>> torch.allclose(result, torch.tensor([
    ...     [1.0130, 1.0130, 3.3133, 3.3133],
    ...     [1.2375, 1.2375, 1.9741, 1.9741]]), atol=1e-4)
    True
    >>> # Example 1: Infinity Handling (Correctness Check)
    >>> flat_scores_inf = torch.tensor([
    ...     [-torch.inf, -torch.inf, 2.0, torch.inf]
    ... ], dtype=torch.float32)
    >>> result_inf = logsumexp_over_siblings(flat_scores_inf, sibling_mask)
    >>> torch.allclose(result_inf, torch.tensor(
    ...     [[-torch.inf, -torch.inf, torch.inf, torch.inf]]
    ... ), atol=1e-4)
    True
    '''

    # B, C = flat_scores.shape
    # G = sibling_mask.shape[1]
    scores_expanded = flat_scores.unsqueeze(2)  # (B, C, 1)
    
    mask_bool = sibling_mask.unsqueeze(0)  # (1, C, G)
    masked_scores = torch.where(
        mask_bool, scores_expanded, -torch.inf
    )  # (B, C, G)

    logsumexp_by_group = torch.logsumexp(masked_scores, dim=1)  # (B, G)

    lse_expanded = logsumexp_by_group.unsqueeze(1)  # (B, 1, G)

    #logsumexp_by_group is by group.  We want to replicate that over categories.
    logsumexp_by_group_category_map = torch.where(
        mask_bool,           # (1, C, G)
        lse_expanded,        # (B, 1, G) -> broadcasts to (B, C, G)
        0.0                  # Scalar
    ) # (B, C, G)

    #logsumexp_by_group_category_map should have exactly one non-zero group entry per category, so we just sum over groups.
    logsumexp = logsumexp_by_group_category_map.sum(dim=-1)  # (B, C)

    return logsumexp


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Compute log(1 - exp(x)) in a numerically stable way.

    This function is designed to prevent the loss of precision that occurs
    when `x` is very close to zero (i.e., a small negative number).
    Directly computing `log(1 - exp(x))` can lead to catastrophic
    cancellation and result in `-inf`.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor containing negative values (log-probabilities).
        The function is not designed for `x >= 0`, as `1 - exp(x)` would be
        zero or negative, making the logarithm undefined.

    Returns
    -------
    torch.Tensor
        The computed `log(1 - exp(x))` values, with the same shape as `x`.

    Notes
    -----
    The function uses two different mathematical identities based on the
    value of `x` to ensure stability:
    
    1. For `x > -ln(2)` (i.e., `x` is close to 0), it computes
       `log(-expm1(x))`. The `torch.expm1(x)` function computes `exp(x) - 1`
       with high precision, avoiding cancellation.
    2. For `x <= -ln(2)`, `exp(x)` is small, so the expression `1 - exp(x)`
       is not problematic. For better precision, `log1p(-exp(x))` is used,
       where `torch.log1p(y)` computes `log(1 + y)`.

    Examples
    --------
    >>> import torch
    >>> log_p = torch.tensor([-1e-9, -2.0, -20.0])
    >>> log1mexp(log_p)
    tensor([-2.0723e+01, -1.4541e-01, -2.0612e-09])


    """
    # The threshold is -ln(2) approx -0.7
    threshold = -0.7
    # For x > threshold, exp(x) is close to 1
    result_close_to_zero = torch.log(-torch.expm1(x))
    # For x <= threshold, exp(x) is small
    result_far_from_zero = torch.log1p(-torch.exp(x))

    return torch.where(x > threshold, result_close_to_zero, result_far_from_zero)


def dict_keyvalue_replace(old_dict: dict[Hashable, Hashable], replacemap: dict[Hashable, Hashable]) -> dict[Hashable, Hashable]:
    """Remaps both keys and values of a dictionary using a replacement map.

    Iterates through `old_dict`, using `replacemap` to find the new
    key and the new value. Assumes both keys and values from
    `old_dict` are valid keys in `replacemap`.

    Parameters
    ----------
    old_dict : dict[Hashable, Hashable]
        The original dictionary. Both its keys and values must
        be hashable and exist as keys in `replacemap`.
    replacemap : dict[Hashable, Hashable]
        A dictionary mapping old keys/values to new hashable keys/values.
        Both its keys and values must be hashable.

    Returns
    -------
    dict[Hashable, Hashable]
        A new dictionary with remapped keys and values.

    Examples
    --------
    >>> old_d = {'a': 'b', 'c': 'd'}
    >>> r_map = {'a': 100, 'b': 200, 'c': 300, 'd': 400}
    >>> dict_keyvalue_replace(old_d, r_map)
    {100: 200, 300: 400}
    """
    new_dict = {}
    for key in old_dict:
        new_dict[replacemap[key]] = replacemap[old_dict[key]]
    return new_dict

