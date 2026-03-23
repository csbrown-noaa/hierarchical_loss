import copy
from . import worms_utils
import pycocowriter.cocomerge

class WormsCocoExpander:
    """
    Standardizes and expands COCO datasets to a unified WoRMS taxonomy hierarchy.

    This class parses multiple COCO dictionaries to identify all unique taxa, fetches
    their full taxonomic trees from the WoRMS API, and aligns the datasets so that 
    they share a perfectly uniform, contiguous, and alphabetically sorted category ID 
    space encompassing the entire hierarchy.

    Future Workflow Strategy
    ------------------------
    If you acquire multiple unaligned datasets (e.g., train, val, test) from different 
    sources, your workflow to standardise them looks like this:

    >>> expander = WormsCocoExpander(use_cache=True)
    >>> 
    >>> # 1. Extract taxa and build the master hierarchy from all inputs
    >>> expander.build_master_hierarchy(train_coco, val_coco, test_coco)
    >>> 
    >>> # 2. Align each dataset to the master taxonomy independently
    >>> aligned_train = expander.align_dataset(train_coco)
    >>> aligned_val = expander.align_dataset(val_coco)
    >>> aligned_test = expander.align_dataset(test_coco)
    >>> 
    >>> # 3. Save the global hierarchy artifact for downstream loss functions
    >>> with open('hierarchy.json', 'w') as f:
    ...     json.dump(expander.hierarchy_tree, f)

    Parameters
    ----------
    use_cache : bool, optional
        If True, uses `requests_cache` to transparently intercept WoRMS API calls 
        and save them to a local SQLite database, avoiding repeated/flaky network 
        requests. Default is True.

    Attributes
    ----------
    hierarchy_tree : dict[str, str]
        A mapping of {child_scientific_name: parent_scientific_name} representing 
        the full unified taxonomic tree.
    name_to_id : dict[str, int]
        A mapping of taxonomic names to their official WoRMS AphiaID.
    master_coco_dummy : dict[str, list]
        An internal empty COCO structure containing the superset of all taxonomic 
        categories discovered, used for dataset alignment.
    """

    def __init__(self, use_cache: bool = True):
        if use_cache:
            import requests_cache
            # Cache expires after 30 days; helps significantly with API rate limiting/flakiness
            requests_cache.install_cache('worms_api_cache', backend='sqlite', expire_after=2592000)
            
        self.hierarchy_tree: dict[str, str] = {}
        self.name_to_id: dict[str, int] = {}
        self.master_coco_dummy: dict[str, list] = {
            "images": [], 
            "annotations": [], 
            "licenses": [], 
            "categories": []
        }

    def build_master_hierarchy(self, *coco_dicts: dict) -> None:
        """
        Extracts unique category names across all provided datasets, fetches their
        WoRMS trees using a deferred-resolution graph discovery strategy, and builds 
        the unified master category list.

        Parameters
        ----------
        *coco_dicts : dict
            An arbitrary number of COCO-formatted dictionaries to scan for taxa.
        """
        # 1. Extract all unique names from all incoming datasets
        all_names = set()
        for c_dict in coco_dicts:
            for cat in c_dict.get('categories', []):
                all_names.add(cat['name'])

        to_lookup = set(all_names)
        deferred = set()
        
        self.hierarchy_tree.clear()
        self.name_to_id.clear()

        # 2. Deferred Resolution Discovery Loop
        while to_lookup:
            name = to_lookup.pop()
            
            # Optimistic ID fetch
            try:
                aphia_id = worms_utils.get_WORMS_id(name)
            except ValueError as e:
                # Catch 204 No Content or HTTP errors and defer them
                deferred.add(name)
                continue

            if aphia_id < 0:
                # Catch -999 (Ambiguous multiple matches) and defer them
                deferred.add(name)
                continue
                
            # Tree Harvest (If ID succeeded)
            try:
                tree = worms_utils.get_WORMS_tree(aphia_id)
            except ValueError as e:
                # Hard crash on tree failure if ID lookup previously succeeded
                raise RuntimeError(
                    f"CRITICAL: Failed to fetch tree for successfully resolved ID {aphia_id} (Name: {name}). "
                    f"Underlying error: {e}"
                )

            # Walk the tree and extract all nodes
            new_hierarchy, new_name_id_map = worms_utils.WORMS_tree_to_name_hierarchy([tree])
            
            # Update global state
            self.hierarchy_tree.update(new_hierarchy)
            self.name_to_id.update(new_name_id_map)
            
            # Retroactively clear all discovered contextual nodes from our sets
            for discovered_name in new_name_id_map.keys():
                to_lookup.discard(discovered_name)
                deferred.discard(discovered_name)

        # 3. Orphan Review
        if deferred:
            print("\n" + "="*70)
            print("WARNING: TRUE TAXONOMIC ORPHANS DETECTED")
            print("="*70)
            print("The following names returned ambiguous results or errors from WoRMS")
            print("and had NO related phylogenetic taxa in the datasets to implicitly")
            print("resolve them via tree context. These require manual curation:")
            for orphan in sorted(deferred):
                print(f" - {orphan}")
            print("="*70 + "\n")

        # 4. Build the master dummy COCO object containing the superset of all taxonomy nodes
        categories = [
            {'id': i + 1, 'name': name} 
            for i, name in enumerate(sorted(self.name_to_id.keys()))
        ]
        
        self.master_coco_dummy['categories'] = categories


    def align_dataset(self, target_coco: dict) -> dict:
        """
        Aligns a single COCO dataset to the unified master hierarchy.

        This method injects the full taxonomic tree into the dataset's categories,
        merges duplicate taxa names into unified IDs, and assigns perfectly contiguous
        (1 to N) IDs alphabetically to satisfy downstream computer vision frameworks.

        Parameters
        ----------
        target_coco : dict
            The target COCO dictionary to be aligned.

        Returns
        -------
        dict
            A new, fully aligned COCO dictionary.
        """
        # Ensure base keys exist so pycocowriter doesn't throw KeyErrors during merge
        safe_target = {
            'images': target_coco.get('images', []),
            'annotations': target_coco.get('annotations', []),
            'categories': target_coco.get('categories', []),
            'licenses': target_coco.get('licenses', [])
        }
        if 'info' in target_coco:
            safe_target['info'] = target_coco['info']

        # Leverage pycocowriter to merge, collapse duplicates, and reindex seamlessly
        aligned = pycocowriter.cocomerge.coco_merge(safe_target, self.master_coco_dummy)
        aligned = pycocowriter.cocomerge.coco_collapse_categories(aligned)
        aligned = pycocowriter.cocomerge.coco_reindex_categories(aligned)
        
        return aligned
