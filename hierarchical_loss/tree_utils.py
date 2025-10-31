from collections.abc import Iterator, Callable, Hashable
from typing import Any

def tree_walk(tree: dict[Hashable, Hashable], node: Hashable) -> Iterator[Hashable]:
    """Walks up the ancestor chain from a starting node.

    This generator yields the starting node first, then its parent,
    its grandparent, and so on, until a root (a node not
    present as a key in the tree) is reached.

    Parameters
    ----------
    tree : dict[Hashable, Hashable]
        The hierarchy tree, in {child: parent} format.
    node : Hashable
        The node to start the walk from.

    Yields
    ------
    Iterator[Hashable]
        An iterator of node IDs in the ancestor chain, starting
        with the given node.

    Examples
    --------
    >>> tree = {0: 1, 1: 2, 3: 4, 4: 2}
    >>> list(tree_walk(tree, 0))
    [0, 1, 2]
    >>> list(tree_walk(tree, 3))
    [3, 4, 2]
    >>> list(tree_walk(tree, 2))
    [2]
    """
    yield node
    while node in tree:
        node = tree[node]
        yield node

def preorder_apply(tree: dict[Hashable, Hashable], f: Callable, *args: Any) -> dict[Hashable, Any]:
    """Applies a function to all nodes in a tree in a pre-order (top-down) fashion.

    This function works by first finding an ancestor path (from leaf to root).
    It then applies the function `f` to the root (or highest unvisited node)
    and iterates *down* the path, applying `f` to each child and passing in
    the result from its parent. This top-down application is a pre-order
    traversal.

    It uses memoization (the `visited` dict) to ensure that `f` is
    applied to each node only once, even in multi-branch trees.

    Parameters
    ----------
    tree : dict[Hashable, Hashable]
        The hierarchy tree, in {child: parent} format.
    f : Callable
        The function to apply to each node. Its signature must be
        `f(node: Hashable, parent_result: Any, *args: Any) -> Any`.
    *args: Any
        Additional positional arguments to be passed to every call of `f`.

    Returns
    -------
    dict[Hashable, Any]
        A dictionary mapping each node ID to the result of `f(node, ...)`.

    Examples
    --------
    >>> # Example: Calculate node depth (pre-order calculation)
    >>> tree = {0: 1, 1: 2, 3: 2} # 0->1->2, 3->2
    >>> def f(node, parent_depth):
    ...     # parent_depth is the result from the parent node
    ...     return 1 if parent_depth is None else parent_depth + 1
    ...
    >>> preorder_apply(tree, f)
    {2: 1, 1: 2, 0: 3, 3: 2}
    """
    visited = {}
    for node in tree:
        path = [node]
        while (node in tree) and (node not in visited):
            node = tree[node]
            path.append(node)
        if node not in visited:
            visited[node] = f(node, None, *args)
        for i in range(-2, -len(path) - 1, -1):
            visited[path[i]] = f(path[i], visited[path[i+1]], *args)
    return visited

def _increment_chain_len(_: Hashable, parent_chain_len: int | None) -> int:
    """Helper function for `preorder_apply` to calculate node depth.

    If a node has no parent result (i.e., it's a root), its depth is 1.
    Otherwise, its depth is its parent's depth + 1.

    Parameters
    ----------
    _ : Any
        The node ID (unused, required by `preorder_apply`).
    parent_chain_len : int | None
        The result from the parent node (its depth).

    Returns
    -------
    int
        The depth of the current node.
    """
    if not parent_chain_len: return 1
    return parent_chain_len + 1

def get_ancestor_chain_lens(tree: dict[Hashable, Hashable]) -> dict[Hashable, int]:
    '''
    Get lengths of ancestor chains in a { child: parent } dictionary tree

    Examples
    --------
    >>> get_ancestor_chain_lens({ 0:1, 1:2, 2:3, 4:5, 5:6, 7:8 })
    {3: 1, 2: 2, 1: 3, 0: 4, 6: 1, 5: 2, 4: 3, 8: 1, 7: 2}

    Parameters
    ----------
    tree: dict[Hashable, Hashable]
        A tree in { child: parent } format.

    Returns
    -------
    lengths: dict[Hashable, int]
        The lengths of the path to the root from each node { node: length }

    '''
    return preorder_apply(tree, _increment_chain_len)

def get_roots(tree: dict[Hashable, Hashable]) -> list[Hashable]:
    """Finds all root nodes in a {child: parent} tree.

    A root node is defined as any node that is not a child of another
    node in the tree (i.e., its ancestor chain length is 1).

    Parameters
    ----------
    tree : dict[Hashable, Hashable]
        A tree in {child: parent} format.

    Returns
    -------
    list[Hashable]
        A list of all root nodes.

    Examples
    --------
    >>> tree = {0: 1, 1: 2, 3: 2, 5: 6}
    >>> get_roots(tree) # Roots are 2 and 6
    [2, 6]
    """
    ancestor_chain_lens = get_ancestor_chain_lens(tree)
    return [node for node in ancestor_chain_lens if ancestor_chain_lens[node] == 1]

def _append_to_parentchild_tree(
    node: Hashable,
    ancestral_chain: list[Hashable] | None,
    parentchild_tree: dict,
) -> list[Hashable]:
    """Helper function for `preorder_apply` to build a nested {parent: {child: ...}} dict.

    This function traverses the nested `parentchild_tree` dict using the
    `ancestral_chain` provided by `preorder_apply` (which is the path from the
    root down to the parent). It then inserts the current `node` as a child
    of its parent.

    It returns the new ancestral chain (parent's chain + current node) to be
    passed to its own children.

    Parameters
    ----------
    node : Hashable
        The current node ID.
    ancestral_chain : list[Hashable] | None
        The path from the root to this node's parent (the result from
        the parent call in `preorder_apply`).
    parentchild_tree : dict
        The main nested dictionary being built (passed as `*args`).

    Returns
    -------
    list[Hashable]
        The ancestral chain for the current node, to be passed to its children.
    """
    ancestral_chain = ancestral_chain or []
    for parent in ancestral_chain:
        parentchild_tree = parentchild_tree[parent]
    if node not in parentchild_tree:
        parentchild_tree[node] = {}
    return ancestral_chain + [node]

def invert_childparent_tree(tree: dict[Hashable, Hashable]) -> dict:
    """Converts a {child: parent} tree into a nested {parent: {child: ...}} tree.

    This function inverts the standard {child: parent} structure, creating
    a nested dictionary that starts from the root(s). It uses
    `preorder_apply` to traverse the tree top-down and build the
    nested structure.

    Parameters
    ----------
    tree : dict[Hashable, Hashable]
        A tree in {child: parent} format.

    Returns
    -------
    dict
        A nested dictionary representing the tree in a top-down format,
        e.g., `{root: {child: {grandchild: {}}}}`.

    Examples
    --------
    >>> tree = {0: 1, 1: 2, 3: 2, 5: 6} # 0->1->2, 3->2, 5->6
    >>> invert_childparent_tree(tree)
    {2: {1: {0: {}}, 3: {}}, 6: {5: {}}}
    """
    parentchild_tree = {}
    preorder_apply(tree, _append_to_parentchild_tree, parentchild_tree)
    return parentchild_tree


def find_closest_permitted_parent(
    node: Hashable,
    tree: dict[Hashable, Hashable],
    permitted_nodes: set[Hashable],
) -> Hashable | None:
    """Finds the first ancestor of a node that is in a permitted set.

    This function walks up the ancestral chain of a node (using the
    {child: parent} tree) and returns the first ancestor it finds
    that is present in the `permitted_nodes` set.

    If no ancestor (including the node itself) is in the set,
    or if the node is not in the tree to begin with, it returns None.

    Parameters
    ----------
    node : Hashable
        The ID of the node to start searching from.
    tree : dict[Hashable, Hashable]
        A tree in {child: parent} format.
    permitted_nodes : set[Hashable]
        A set of node IDs that are considered "permitted".

    Returns
    -------
    Hashable | None
        The ID of the closest permitted ancestor, or None if none is found.

    Examples
    --------
    >>> tree = {1: 2, 2: 3, 3: 4, 4: 5}
    >>> permitted = {0, 2, 5}
    >>> find_closest_permitted_parent(1, tree, permitted) # 1 -> 2 (permitted)
    2
    >>> find_closest_permitted_parent(0, tree, permitted) # 0 is not in tree keys, returns None
    >>> tree[0] = 1 # Add 0 to the tree
    >>> find_closest_permitted_parent(0, tree, permitted) # 0 -> 1 -> 2 (permitted)
    2
    >>> tree = {10: 20, 20: 30, 30: 40}
    >>> find_closest_permitted_parent(10, tree, {50, 60}) # No permitted ancestors, returns None
    """
    if node not in tree:
        return None
    parent = tree[node]
    while parent not in permitted_nodes:
        if parent in tree:
            parent = tree[parent]
        else:
            return None
    return parent

def trim_childparent_tree(
    tree: dict[Hashable, Hashable], permitted_nodes: set[Hashable]
) -> dict[Hashable, Hashable | None]:
    """Trims a {child: parent} tree to only include permitted nodes.

    This function first remaps every node in the tree to its closest
    permitted ancestor. It then filters this map, keeping only the
    entries where the node (the key) is *also* in the `permitted_nodes`
    set.

    The result is a new {child: parent} tree containing *only*
    permitted nodes, mapped to their closest permitted ancestor
    (which will be another permitted node or None).

    Parameters
    ----------
    tree : dict[Hashable, Hashable]
        A tree in {child: parent} format.
    permitted_nodes : set[Hashable]
        A set of node IDs to keep.

    Returns
    -------
    dict[Hashable, Hashable | None]
        A new {child: parent} tree containing only permitted nodes,
        each re-mapped to its closest permitted ancestor.

    Examples
    --------
    >>> tree = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5} # 0->1->2->3->4->5
    >>> permitted = {0, 2, 5} # 0, 2, and 5 are permitted
    >>> trim_childparent_tree(tree, permitted)
    {0: 2, 2: 5}
    """
    new_tree = {}
    for node in tree:
        closest_permitted_parent = find_closest_permitted_parent(node, tree, permitted_nodes)
        new_tree[node] = closest_permitted_parent
    for node in list(new_tree.keys()):
        if new_tree[node] is None or (node not in permitted_nodes):
            del new_tree[node]
    return new_tree