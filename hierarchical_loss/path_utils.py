import torch
import itertools
from .tree_utils import *

def truncate_path_conditionals(path: list[int], score: torch.Tensor, threshold: float = 0.25) -> tuple[list[int], torch.Tensor]:
    """Truncates a path based on a conditional probability threshold.

    This function iterates through a path and its corresponding conditional
    probabilities, stopping at the first element where the probability
    is below the given threshold.

    Parameters
    ----------
    path : list[int]
        A list of category indices representing the path.
    score : torch.Tensor
        A 1D tensor where each element is the conditional probability
        of the corresponding category in the path.
    threshold : float, optional
        The probability threshold below which to truncate, by default 0.25.

    Returns
    -------
    tuple[list[int], torch.Tensor]
        A tuple containing the truncated path and its corresponding scores.

    Examples
    --------
    >>> import torch
    >>> path = [4, 7]
    >>> score = torch.tensor([0.5412, 0.4371])
    >>> truncate_path_conditionals(path, score, threshold=0.589)
    ([], tensor([]))
    >>> path = [4, 2]
    >>> score = torch.tensor([0.9896, 0.5891])
    >>> truncate_path_conditionals(path, score, threshold=0.589)
    ([4, 2], tensor([0.9896, 0.5891]))
    """
    truncated_path, truncated_score = [], []
    for category, p in zip(path, score):
        if p < threshold:
            break
        truncated_path.append(category)
    return truncated_path, score[:len(truncated_path)]


def truncate_path_marginals(path: list[int], score: torch.Tensor, threshold: float = 0.25) -> tuple[list[int], torch.Tensor]:
    """Truncates a path based on a marginal probability threshold.

    This function iterates through a path, calculating the cumulative
    product (marginal probability) of the scores. It stops at the first
    element where this cumulative product falls below the given threshold.

    Parameters
    ----------
    path : list[int]
        A list of category indices representing the path.
    score : torch.Tensor
        A 1D tensor where each element is the conditional probability
        of the corresponding category in the path.
    threshold : float, optional
        The probability threshold below which to truncate, by default 0.25.

    Returns
    -------
    tuple[list[int], torch.Tensor]
        A tuple containing the truncated path and its corresponding scores.

    Examples
    --------
    >>> import torch
    >>> path = [4, 2]
    >>> score = torch.tensor([0.9896, 0.5891])
    >>> truncate_path_marginals(path, score, threshold=0.589)
    ([4], tensor([0.9896]))
    >>> path = [4, 6]
    >>> score = torch.tensor([0.9246, 0.7684])
    >>> truncate_path_marginals(path, score, threshold=0.589)
    ([4, 6], tensor([0.9246, 0.7684]))
    """
    truncated_path, truncated_score = [], []
    marginal_p = 1
    for category, p in zip(path, score):
        marginal_p *= p
        if marginal_p < threshold:
            break
        truncated_path.append(category)
    return truncated_path, score[:len(truncated_path)]


def truncate_paths_marginals(predicted_paths: list[list[int]], predicted_path_scores: list[torch.Tensor], threshold: float = 0.25) -> tuple[list[list[int]], list[torch.Tensor]]:
    """Applies marginal probability truncation to a list of paths.

    This function iterates through lists of paths and scores, applying
    the `truncate_path_marginals` function to each path-score pair.

    Parameters
    ----------
    predicted_paths : list[list[int]]
        A list of paths, where each path is a list of category indices.
    predicted_path_scores : list[torch.Tensor]
        A list of 1D tensors, each corresponding to a path in `predicted_paths`.
    threshold : float, optional
        The probability threshold to pass to the truncation function,
        by default 0.25.

    Returns
    -------
    tuple[list[list[int]], list[torch.Tensor]]
        A tuple containing the list of truncated paths and the list of
        their corresponding truncated scores.

    Examples
    --------
    >>> import torch
    >>> paths = [[4, 2], [4, 6], [4, 5], [4, 7], [4, 2]]
    >>> scores = [torch.tensor([0.9896, 0.5891]), torch.tensor([0.9246, 0.7684]), torch.tensor([0.8949, 0.8765]), torch.tensor([0.5412, 0.4371]), torch.tensor([0.5001, 0.0830])]
    >>> tpaths, tscores = truncate_paths_marginals(paths, scores, threshold=0.589)
    >>> tpaths
    [[4], [4, 6], [4, 5], [], []]
    >>> tscores
    [tensor([0.9896]), tensor([0.9246, 0.7684]), tensor([0.8949, 0.8765]), tensor([]), tensor([])]
    """
    tpaths, tscores = [], []
    for paths, scores in zip(predicted_paths, predicted_path_scores):
        tpath, tscore = truncate_path_marginals(paths, scores, threshold=threshold)
        tpaths.append(tpath), tscores.append(tscore)
    return tpaths, tscores


def truncate_paths_conditionals(predicted_paths: list[list[int]], predicted_path_scores: list[torch.Tensor], threshold: float = 0.25) -> tuple[list[list[int]], list[torch.Tensor]]:
    """Applies conditional probability truncation to a list of paths.

    This function iterates through lists of paths and scores, applying
    the `truncate_path_conditionals` function to each path-score pair.

    Parameters
    ----------
    predicted_paths : list[list[int]]
        A list of paths, where each path is a list of category indices.
    predicted_path_scores : list[torch.Tensor]
        A list of 1D tensors, each corresponding to a path in `predicted_paths`.
    threshold : float, optional
        The probability threshold to pass to the truncation function,
        by default 0.25.

    Returns
    -------
    tuple[list[list[int]], list[torch.Tensor]]
        A tuple containing the list of truncated paths and the list of
        their corresponding truncated scores.

    Examples
    --------
    >>> import torch
    >>> paths = [[4, 2], [4, 6], [4, 5], [4, 7], [4, 2]]
    >>> scores = [torch.tensor([0.9896, 0.5891]), torch.tensor([0.9246, 0.7684]), torch.tensor([0.8949, 0.8765]), torch.tensor([0.5412, 0.4371]), torch.tensor([0.5001, 0.0830])]
    >>> tpaths, tscores = truncate_paths_conditionals(paths, scores, threshold=0.589)
    >>> tpaths
    [[4, 2], [4, 6], [4, 5], [], []]
    >>> tscores
    [tensor([0.9896, 0.5891]), tensor([0.9246, 0.7684]), tensor([0.8949, 0.8765]), tensor([]), tensor([])]
    """
    tpaths, tscores = [], []
    for paths, scores in zip(predicted_paths, predicted_path_scores):
        tpath, tscore = truncate_path_conditionals(paths, scores, threshold=threshold)
        tpaths.append(tpath), tscores.append(tscore)
    return tpaths, tscores


def batch_truncate_paths_marginals(predicted_paths: list[list[list[int]]], predicted_path_scores: list[list[torch.Tensor]], threshold: float = 0.25) -> list[tuple[list[list[int]], list[torch.Tensor]]]:
    """Applies marginal probability truncation to a batch of path lists.

    This function maps the `truncate_paths_marginals` function over a
    batch of predicted paths and scores.

    Parameters
    ----------
    predicted_paths : list[list[list[int]]]
        A batch of path lists. Each item in the outer list corresponds to
        an item in the batch.
    predicted_path_scores : list[list[torch.Tensor]]
        A batch of score lists, corresponding to `predicted_paths`.
    threshold : float, optional
        The probability threshold to use for truncation, by default 0.25.

    Returns
    -------
    list[tuple[list[list[int]], list[torch.Tensor]]]
        A list of tuples, where each tuple contains the truncated paths and
        scores for an item in the batch.

    Examples
    --------
    >>> import torch
    >>> paths_batch = [[[4, 2], [4, 6], [4, 5], [4, 7], [4, 2]], [[4, 2], [4, 6], [4, 5], [4, 7], [4, 2]]]
    >>> scores_batch = [[torch.tensor([0.9896, 0.5891]), torch.tensor([0.9246, 0.7684]), torch.tensor([0.8949, 0.8765]), torch.tensor([0.5412, 0.4371]), torch.tensor([0.5001, 0.0830])], [torch.tensor([0.9896, 0.5891]), torch.tensor([0.9246, 0.7684]), torch.tensor([0.8949, 0.8765]), torch.tensor([0.5412, 0.4371]), torch.tensor([0.5001, 0.0830])]]
    >>> batch_truncate_paths_marginals(paths_batch, scores_batch, 0.589)
    [([[4], [4, 6], [4, 5], [], []], [tensor([0.9896]), tensor([0.9246, 0.7684]), tensor([0.8949, 0.8765]), tensor([]), tensor([])]), ([[4], [4, 6], [4, 5], [], []], [tensor([0.9896]), tensor([0.9246, 0.7684]), tensor([0.8949, 0.8765]), tensor([]), tensor([])])]
    """
    B = len(predicted_paths)
    return list(itertools.starmap(truncate_paths_marginals, zip(predicted_paths, predicted_path_scores, itertools.repeat(threshold, B))))


def batch_truncate_paths_conditionals(predicted_paths: list[list[list[int]]], predicted_path_scores: list[list[torch.Tensor]], threshold: float = 0.25) -> list[tuple[list[list[int]], list[torch.Tensor]]]:
    """Applies conditional probability truncation to a batch of path lists.

    This function maps the `truncate_paths_conditionals` function over a
    batch of predicted paths and scores.

    Parameters
    ----------
    predicted_paths : list[list[list[int]]]
        A batch of path lists. Each item in the outer list corresponds to
        an item in the batch.
    predicted_path_scores : list[list[torch.Tensor]]
        A batch of score lists, corresponding to `predicted_paths`.
    threshold : float, optional
        The probability threshold to use for truncation, by default 0.25.

    Returns
    -------
    list[tuple[list[list[int]], list[torch.Tensor]]]
        A list of tuples, where each tuple contains the truncated paths and
        scores for an item in the batch.

    Examples
    --------
    >>> import torch
    >>> paths_batch = [[[4, 2], [4, 6], [4, 5], [4, 7], [4, 2]], [[4, 2], [4, 6], [4, 5], [4, 7], [4, 2]]]
    >>> scores_batch = [[torch.tensor([0.9896, 0.5891]), torch.tensor([0.9246, 0.7684]), torch.tensor([0.8949, 0.8765]), torch.tensor([0.5412, 0.4371]), torch.tensor([0.5001, 0.0830])], [torch.tensor([0.9896, 0.5891]), torch.tensor([0.9246, 0.7684]), torch.tensor([0.8949, 0.8765]), torch.tensor([0.5412, 0.4371]), torch.tensor([0.5001, 0.0830])]]
    >>> batch_truncate_paths_conditionals(paths_batch, scores_batch, 0.589)
    [([[4, 2], [4, 6], [4, 5], [], []], [tensor([0.9896, 0.5891]), tensor([0.9246, 0.7684]), tensor([0.8949, 0.8765]), tensor([]), tensor([])]), ([[4, 2], [4, 6], [4, 5], [], []], [tensor([0.9896, 0.5891]), tensor([0.9246, 0.7684]), tensor([0.8949, 0.8765]), tensor([]), tensor([])])]
    """
    B = len(predicted_paths)
    return list(itertools.starmap(truncate_paths_conditionals, zip(predicted_paths, predicted_path_scores, itertools.repeat(threshold, B))))


def filter_empty_paths(predicted_boxes: torch.Tensor, predicted_paths: list[list[int]], predicted_path_scores: list[torch.Tensor]) -> tuple[torch.Tensor, list[list[int]], list[torch.Tensor]]:
    """Filters out predictions with empty paths.

    After truncation, some paths may become empty. This function removes
    those empty paths along with their corresponding scores and bounding
    boxes.

    Parameters
    ----------
    predicted_boxes : torch.Tensor
        A 2D tensor of bounding box predictions, where columns correspond
        to individual predictions (e.g., shape [4, N]).
    predicted_paths : list[list[int]]
        A list of predicted paths.
    predicted_path_scores : list[torch.Tensor]
        A list of predicted path scores.

    Returns
    -------
    tuple[torch.Tensor, list[list[int]], list[torch.Tensor]]
        A tuple containing the filtered boxes, paths, and scores,
        with empty path predictions removed.

    Examples
    --------
    >>> import torch
    >>> boxes = torch.tensor([[482.27, 395.77, 241.98, 359.60, 258.38], [8.11, 156.87, 152.91, 335.40, 24.81], [610.42, 429.38, 307.70, 382.68, 413.79], [103.86, 200.93, 197.57, 352.40, 197.61]])
    >>> paths = [[4], [4, 6], [4, 5], [], []]
    >>> scores = [torch.tensor([0.9896]), torch.tensor([0.9246, 0.7684]), torch.tensor([0.8949, 0.8765]), torch.tensor([]), torch.tensor([])]
    >>> f_boxes, f_paths, f_scores = filter_empty_paths(boxes, paths, scores)
    >>> f_boxes
    tensor([[482.2700, 395.7700, 241.9800],
            [  8.1100, 156.8700, 152.9100],
            [610.4200, 429.3800, 307.7000],
            [103.8600, 200.9300, 197.5700]])
    >>> f_paths
    [[4], [4, 6], [4, 5]]
    >>> f_scores
    [tensor([0.9896]), tensor([0.9246, 0.7684]), tensor([0.8949, 0.8765])]
    """
    keep_idx = [i for i, path in enumerate(predicted_paths) if len(path) > 0]
    return (
        predicted_boxes[:,keep_idx],
        [predicted_paths[k] for k in keep_idx],
        [predicted_path_scores[k] for k in keep_idx]
    )


def batch_filter_empty_paths(predicted_boxes: list[torch.Tensor], predicted_paths: list[list[list[int]]], predicted_path_scores: list[list[torch.Tensor]]) -> list[tuple[torch.Tensor, list[list[int]], list[torch.Tensor]]]:
    """Applies empty path filtering to a batch of predictions.

    This function maps the `filter_empty_paths` function over a batch of
    predicted boxes, paths, and scores.

    Parameters
    ----------
    predicted_boxes : list[torch.Tensor]
        A batch of bounding box tensors.
    predicted_paths : list[list[list[int]]]
        A batch of predicted path lists.
    predicted_path_scores : list[list[torch.Tensor]]
        A batch of predicted path score lists.

    Returns
    -------
    list[tuple[torch.Tensor, list[list[int]], list[torch.Tensor]]]
        A list of tuples, where each tuple contains the filtered boxes,
        paths, and scores for an item in the batch.

    Examples
    --------
    >>> import torch
    >>> boxes_batch = [torch.tensor([[482.27, 395.77, 241.98, 359.60, 258.38], [8.11, 156.87, 152.91, 335.40, 24.81], [610.42, 429.38, 307.70, 382.68, 413.79], [103.86, 200.93, 197.57, 352.40, 197.61]]), torch.tensor([[482.27, 395.77, 241.98, 359.60, 258.38], [8.11, 156.87, 152.91, 335.40, 24.81], [610.42, 429.38, 307.70, 382.68, 413.79], [103.86, 200.93, 197.57, 352.40, 197.61]])]
    >>> paths_batch = [[[4], [4, 6], [4, 5], [], []], [[4], [4, 6], [4, 5], [], []]]
    >>> scores_batch = [[torch.tensor([0.9896]), torch.tensor([0.9246, 0.7684]), torch.tensor([0.8949, 0.8765]), torch.tensor([]), torch.tensor([])], [torch.tensor([0.9896]), torch.tensor([0.9246, 0.7684]), torch.tensor([0.8949, 0.8765]), torch.tensor([]), torch.tensor([])]]
    >>> result = batch_filter_empty_paths(boxes_batch, paths_batch, scores_batch)
    >>> len(result)
    2
    >>> result[0][0] # boxes for first batch item
    tensor([[482.2700, 395.7700, 241.9800],
            [  8.1100, 156.8700, 152.9100],
            [610.4200, 429.3800, 307.7000],
            [103.8600, 200.9300, 197.5700]])
    >>> result[0][1] # paths for first batch item
    [[4], [4, 6], [4, 5]]
    """
    B = len(predicted_paths)
    return list(itertools.starmap(filter_empty_paths, zip(predicted_boxes, predicted_paths, predicted_path_scores)))


def optimal_hierarchical_paths(class_scores: list[torch.Tensor], hierarchy: dict[int, int]) -> tuple[list[list[list[int]]], list[list[torch.Tensor]]]:
    """
    .. deprecated:: 0.X.X
       This function is deprecated as it re-computes the hierarchy
       on every call, causing a performance bottleneck.
       Use a `Hierarchy` object to pre-compute the `inverted_tree`
       and `roots`, and then call `optimal_hierarchical_path` directly.
    """
    inverted_tree = construct_parent_childtensor_tree(hierarchy, device=class_scores[0].device)
    roots = torch.tensor(get_roots(hierarchy), device=class_scores[0].device)
    return optimal_hierarchical_path(class_scores, inverted_tree, roots)


def optimal_hierarchical_path(class_scores: list[torch.Tensor], inverted_tree: dict[int, torch.Tensor], roots: torch.Tensor) -> tuple[list[list[list[int]]], list[list[torch.Tensor]]]:
    """Finds optimal paths and extracts their corresponding scores.

    This function wraps `get_optimal_ancestral_chain` to find the
    single best greedy path for each detection, and then gathers
    the raw scores associated with each node in those paths.

    Parameters
    ----------
    class_scores : list[torch.Tensor]
        A list of confidence tensors, one per batch item. Each tensor
        should have shape (C, N), where C is the number of classes
        and N is the number of detections.
    inverted_tree : dict[int, torch.Tensor]
        The class hierarchy in `{parent_id: tensor([child1, child2, ...])}`
        format.
    roots : torch.Tensor
        A 1D tensor containing the integer IDs of the root nodes.

    Returns
    -------
    tuple[list[list[list[int]]], list[list[torch.Tensor]]]
        A tuple containing two items:
        1. `optimal_paths`: The nested list of paths, as returned
           by `get_optimal_ancestral_chain`.
        2. `optimal_path_scores`: A nested list of the same structure,
           but containing 1D tensors of the scores for each path.

    Examples
    --------
    >>> hierarchy = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    >>> # C=7 classes, N=2 detections, B=1 batch item
    >>> # Scores are shaped (C, N)
    >>> scores = torch.tensor([
    ...     [10., 10.],  # 0 (Root)
    ...     [ 5.,  1.],  # 1 (Child of 0)
    ...     [ 1.,  5.],  # 2 (Child of 0)
    ...     [ 2.,  0.],  # 3 (Child of 1)
    ...     [ 8.,  0.],  # 4 (Child of 1)
    ...     [ 0.,  8.],  # 5 (Child of 2)
    ...     [ 0.,  2.]   # 6 (Child of 2)
    ... ], dtype=torch.float32)
    >>> class_scores = [scores]
    >>> inverted_tree = construct_parent_childtensor_tree(hierarchy, device=class_scores[0].device)
    >>> roots = torch.tensor(get_roots(hierarchy), device=class_scores[0].device)
    >>> paths, path_scores = optimal_hierarchical_path(class_scores, inverted_tree, roots)
    >>> paths
    [[[0, 1, 4], [0, 2, 5]]]
    >>> path_scores
    [[tensor([10.,  5.,  8.]), tensor([10.,  5.,  8.])]]
    """
    bpaths = []
    bscores = []
    for b, confidence in enumerate(class_scores):
        paths = []
        scores = []
        for i in range(confidence.shape[1]):
            confidence_row = confidence[..., i]
            path = []
            path_score = []
            siblings = roots
            while siblings is not None:
                best = confidence_row.index_select(0, siblings).argmax()
                best_node_id = int(siblings[best])
                path.append(best_node_id)
                path_score.append(confidence_row[best_node_id])
                siblings = inverted_tree[best_node_id] if best_node_id in inverted_tree else None
            paths.append(path)
            scores.append(torch.stack(path_score))
        bpaths.append(paths)
        bscores.append(scores)
    return bpaths, bscores


def construct_parent_childset_tree(tree: dict[Hashable, Hashable]) -> dict[Hashable, set]:
    """Converts a {child: parent} tree into a {parent: set[children]} tree.

    This function inverts the standard {child: parent} structure, creating
    a dictionary for navigating the tree top-down.

    Parameters
    ----------
    tree : dict[Hashable, Hashable]
        A tree in {child: parent} format.

    Returns
    -------
    dict
        A nested dictionary representing the tree in a top-down format,
        e.g., `{parent: set[children]}`.

    Examples
    --------
    >>> childparent_tree = {0:1, 1:2, 3:2, 4:5}
    >>> construct_parent_childset_tree(childparent_tree)
    {1: {0}, 2: {1, 3}, 5: {4}}
    """
    parent_childset_tree = {}
    for child, parent in tree.items():
        if parent not in parent_childset_tree:
            parent_childset_tree[parent] = set()
        parent_childset_tree[parent].add(child)
    return parent_childset_tree

def construct_parent_childtensor_tree(tree: dict[Hashable, Hashable], device=None) -> dict[Hashable, torch.Tensor]:
    """Converts a {child: parent} tree into a {parent: tensor[children]} tree.

    This function inverts the standard {child: parent} structure, creating
    a dictionary for navigating the tree top-down.

    Parameters
    ----------
    tree : dict[Hashable, Hashable]
        A tree in {child: parent} format.

    Returns
    -------
    dict
        A nested dictionary representing the tree in a top-down format,
        e.g., `{parent: tensor[children]}`.

    Examples
    --------
    >>> childparent_tree = {0:1, 1:2, 3:2, 4:5}
    >>> construct_parent_childtensor_tree(childparent_tree)
    {1: tensor([0]), 2: tensor([1, 3]), 5: tensor([4])}
    """
    childset_tree = construct_parent_childset_tree(tree)
    return {k: torch.tensor(list(v), device=device) for k,v  in childset_tree.items()} 
    