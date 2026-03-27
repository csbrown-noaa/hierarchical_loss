import copy

def get_lineage(taxon: str, hierarchy_tree: dict[str, str]) -> list[str]:
    """
    Traces a taxon up to its root ancestor to build its phylogenetic lineage.

    Parameters
    ----------
    taxon : str
        The scientific name of the taxon to trace.
    hierarchy_tree : dict[str, str]
        A dictionary mapping {child_name: parent_name}.

    Returns
    -------
    list[str]
        The lineage ordered from root (index 0) to the target taxon (last index).

    Examples
    --------
    >>> tree = {
    ...     "Sphyraena barracuda": "Sphyraena", 
    ...     "Sphyraena": "Sphyraenidae", 
    ...     "Sphyraenidae": "Carangiformes"
    ... }
    >>> get_lineage("Sphyraena barracuda", tree)
    ['Carangiformes', 'Sphyraenidae', 'Sphyraena', 'Sphyraena barracuda']
    >>> get_lineage("Carangiformes", tree)
    ['Carangiformes']
    """
    lineage = [taxon]
    current = taxon
    
    while current in hierarchy_tree:
        parent = hierarchy_tree[current]
        if parent == current:
            break
        lineage.append(parent)
        current = parent
        
    return lineage[::-1]


def build_all_lineages(hierarchy_tree: dict[str, str]) -> dict[str, list[str]]:
    """
    Pre-computes the lineage from root to leaf for every taxon in the tree.

    Parameters
    ----------
    hierarchy_tree : dict[str, str]
        A dictionary mapping {child_name: parent_name}.

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping {taxon_name: [root, ..., taxon_name]}.

    Examples
    --------
    >>> tree = {"Sphyraena barracuda": "Sphyraena", "Sphyraena": "Sphyraenidae"}
    >>> lineages = build_all_lineages(tree)
    >>> lineages["Sphyraena barracuda"]
    ['Sphyraenidae', 'Sphyraena', 'Sphyraena barracuda']
    >>> lineages["Sphyraenidae"]
    ['Sphyraenidae']
    """
    all_taxa = set(hierarchy_tree.keys()).union(set(hierarchy_tree.values()))
    
    lineages = {}
    for taxon in all_taxa:
        lineages[taxon] = get_lineage(taxon, hierarchy_tree)
        
    return lineages


def build_depth_map(
    lineages: dict[str, list[str]], 
    target_depth: int, 
    name_to_id: dict[str, int]
) -> dict[int, int]:
    """
    Creates a mapping of original category IDs to their ancestor IDs at a target depth.
    
    If a taxon's lineage bottoms out earlier than the target depth, it gracefully 
    maps to its deepest available node.

    Parameters
    ----------
    lineages : dict[str, list[str]]
        The pre-computed lineages for all taxa, typically from `build_all_lineages`.
    target_depth : int
        The targeted level of phylogenetic depth (0 = root).
    name_to_id : dict[str, int]
        Mapping of taxon names to their master COCO category IDs.

    Returns
    -------
    dict[int, int]
        A dictionary mapping {original_category_id: target_depth_category_id}.

    Examples
    --------
    >>> lineages = {'Sphyraena barracuda': ['Carangiformes', 'Sphyraenidae', 'Sphyraena', 'Sphyraena barracuda']}
    >>> name_to_id = {'Carangiformes': 1, 'Sphyraenidae': 2, 'Sphyraena': 3, 'Sphyraena barracuda': 4}
    >>> # Depth 1 maps to 'Sphyraenidae' (ID 2)
    >>> build_depth_map(lineages, 1, name_to_id)
    {4: 2}
    >>> # Depth 5 bottoms out at 'Sphyraena barracuda' (ID 4)
    >>> build_depth_map(lineages, 5, name_to_id)
    {4: 4}
    """
    depth_map = {}
    for taxon, lineage in lineages.items():
        original_id = name_to_id[taxon]
        target_idx = min(target_depth, len(lineage) - 1)
        depth_map[original_id] = name_to_id[lineage[target_idx]]
        
    return depth_map


def cast_coco_to_depth(coco_dict: dict, depth_map: dict[int, int]) -> dict:
    """
    Mutates a COCO dictionary to cast all annotations to a target phylogenetic depth.
    
    This performs a deep copy to ensure memory isolation from the master dataset. 
    It iterates through the annotations and strictly updates the `category_id` 
    to its remapped ancestor ID.

    Parameters
    ----------
    coco_dict : dict
        The source COCO dictionary containing 'annotations' and 'categories'.
    depth_map : dict[int, int]
        The mapping of original category IDs to ancestor category IDs.

    Returns
    -------
    dict
        A new COCO dictionary with updated annotations. The categories block
        remains identical to the source.
    """
    new_coco = copy.deepcopy(coco_dict)
    for ann in new_coco.get('annotations', []):
        ann['category_id'] = depth_map[ann['category_id']]
    return new_coco


def get_active_category_ids(*coco_dicts: dict) -> set[int]:
    """
    Scans multiple COCO dictionaries to find all unique category IDs currently 
    in use by the annotations.

    Parameters
    ----------
    *coco_dicts : dict
        An arbitrary number of COCO-formatted dictionaries to scan.

    Returns
    -------
    set[int]
        A set containing every unique `category_id` found across all annotations.

    Examples
    --------
    >>> coco1 = {'annotations': [{'category_id': 1}, {'category_id': 2}]}
    >>> coco2 = {'annotations': [{'category_id': 2}, {'category_id': 5}]}
    >>> sorted(list(get_active_category_ids(coco1, coco2)))
    [1, 2, 5]
    """
    active_ids = set()
    for c_dict in coco_dicts:
        for ann in c_dict.get('annotations', []):
            active_ids.add(ann['category_id'])
    return active_ids


def build_dense_category_map(active_ids: set[int]) -> tuple[dict[int, int], dict[int, int]]:
    """
    Maps a sparse set of category IDs to a dense, contiguous range (1 to N).
    
    This is necessary for standard classification networks (like flat YOLO) that 
    expect output tensors to correspond to contiguous class indices without gaps.

    Parameters
    ----------
    active_ids : set[int]
        A set of the sparse category IDs (e.g., {4, 18, 200}).

    Returns
    -------
    tuple[dict[int, int], dict[int, int]]
        A tuple of (old_to_new_mapping, new_to_old_mapping).

    Examples
    --------
    >>> active_ids = {4, 18, 200}
    >>> old_to_new, new_to_old = build_dense_category_map(active_ids)
    >>> old_to_new
    {4: 1, 18: 2, 200: 3}
    >>> new_to_old
    {1: 4, 2: 18, 3: 200}
    """
    old_to_new = {}
    new_to_old = {}
    for new_id, old_id in enumerate(sorted(list(active_ids)), start=1):
        old_to_new[old_id] = new_id
        new_to_old[new_id] = old_id
    return old_to_new, new_to_old


def restrict_and_reindex_coco(
    coco_dict: dict, 
    old_to_new: dict[int, int], 
    master_categories: list[dict]
) -> dict:
    """
    Filters the categories block to only active IDs and re-indexes the entire 
    dataset to a dense 1-to-N space.
    
    This rebuilds both the 'categories' list and the 'category_id' inside every 
    annotation, ensuring the resulting dataset is completely contiguous and 
    contains no empty classification buckets.

    Parameters
    ----------
    coco_dict : dict
        The source COCO dictionary.
    old_to_new : dict[int, int]
        The mapping dictionary to translate sparse original IDs to dense new IDs.
    master_categories : list[dict]
        The original complete master list of COCO category dictionaries (used to 
        preserve metadata like names and supercategories during the rebuild).

    Returns
    -------
    dict
        A new COCO dictionary restricted to active categories and contiguously indexed.
    """
    new_coco = copy.deepcopy(coco_dict)
    master_cat_map = {c['id']: c for c in master_categories}
    
    # Rebuild the categories block sparsely and densely indexed
    new_categories = []
    for old_id, new_id in old_to_new.items():
        cat = copy.deepcopy(master_cat_map[old_id])
        cat['id'] = new_id
        new_categories.append(cat)
        
    new_coco['categories'] = new_categories
    
    # Re-index the annotations
    for ann in new_coco.get('annotations', []):
        ann['category_id'] = old_to_new[ann['category_id']]
        
    return new_coco
