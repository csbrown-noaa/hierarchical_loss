#!/usr/bin/env python
# coding: utf-8

import copy
import os
import urllib.request
import json
import shutil
import argparse
import pycocowriter.coco2yolo
import pycocowriter.cocomerge

from . import worms_utils



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
        self._to_lookup: set[str] = set()
        self._deferred: set[str] = set()
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
        self._extract_unique_names(*coco_dicts)
        self._process_graph_discovery()
        self._rescue_orphans()
        self._log_true_orphans()
        self._compile_dummy_dataset()

    def _extract_unique_names(self, *coco_dicts: dict) -> None:
        """Collects all unique taxonomic names from the datasets and initializes state."""
        self._to_lookup.clear()
        self._deferred.clear()
        self.hierarchy_tree.clear()
        self.name_to_id.clear()

        for c_dict in coco_dicts:
            for cat in c_dict.get('categories', []):
                self._to_lookup.add(cat['name'])

    def _process_graph_discovery(self) -> None:
        """Pops names, performs optimistic ID fetches, and retroactively resolves using tree context."""
        while self._to_lookup:
            name = self._to_lookup.pop()
            
            # Optimistic ID fetch
            try:
                aphia_id = worms_utils.get_WORMS_id(name)
            except ValueError:
                # Catch 204 No Content or HTTP errors and defer them
                self._deferred.add(name)
                continue

            if aphia_id < 0:
                # Catch -999 (Ambiguous multiple matches) and defer them
                self._deferred.add(name)
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
                self._to_lookup.discard(discovered_name)
                self._deferred.discard(discovered_name)

    def _rescue_orphans(self) -> None:
        """Attempts to rescue true orphans using strict exact-match accepted-status disambiguation."""
        for orphan_name in list(self._deferred):
            try:
                rescued_id = worms_utils.disambiguate_taxon(orphan_name)
                
                # If rescue succeeds, we must fetch its tree to capture its parent hierarchy
                tree = worms_utils.get_WORMS_tree(rescued_id)
                new_hierarchy, new_name_id_map = worms_utils.WORMS_tree_to_name_hierarchy([tree])
                
                self.hierarchy_tree.update(new_hierarchy)
                self.name_to_id.update(new_name_id_map)
                
                # Safely remove it from deferred now that it's successfully resolved
                self._deferred.remove(orphan_name)
            except ValueError:
                # Disambiguation failed (0 or >1 matches); leave it in deferred.
                pass

    def _log_true_orphans(self) -> None:
        """Prints loud warnings for any names that survived both contextual and rescue passes."""
        if self._deferred:
            print("\n" + "="*70)
            print("WARNING: TRUE TAXONOMIC ORPHANS DETECTED")
            print("="*70)
            print("The following names returned ambiguous results or errors from WoRMS")
            print("and could not be resolved contextually or via strict disambiguation.")
            print("These require manual curation:")
            for orphan in sorted(self._deferred):
                print(f" - {orphan}")
            print("="*70 + "\n")

    def _compile_dummy_dataset(self) -> None:
        """Sorts the final name mapping and builds the master dummy COCO categories block."""
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
        # Deepcopy ONLY the small master dummy to protect it across multiple 
        # dataset alignments, while intentionally leaving target_coco to be 
        # mutated in-place to conserve memory on massive datasets.
        dummy_copy = copy.deepcopy(self.master_coco_dummy)

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
        aligned = pycocowriter.cocomerge.coco_merge(safe_target, dummy_copy)
        aligned = pycocowriter.cocomerge.coco_collapse_categories(aligned)
        aligned = pycocowriter.cocomerge.coco_reindex_categories(aligned)
        
        return aligned



def verify_alignment(orig_mapping: dict, new_coco: dict, split_name: str) -> None:
    """
    Verifies that the aligned annotations mathematically mapped to the correct taxonomic names.

    Parameters
    ----------
    orig_mapping : dict
        A dictionary mapping the original `annotation_id` to the original `category_name`.
    new_coco : dict
        The newly aligned COCO dictionary to test.
    split_name : str
        The string identifier for the dataset split (used for logging output).

    Raises
    ------
    AssertionError
        If an annotation's taxonomic name changed during the alignment mapping.
    """
    category_map_new = {cat['id']: cat for cat in new_coco['categories']}
    
    for ann in new_coco['annotations']:
        old_cat_name = orig_mapping[ann['id']]
        new_cat_name = category_map_new[ann['category_id']]['name']
        assert old_cat_name == new_cat_name, f"Category mapping failed: {old_cat_name} != {new_cat_name}"
        
    print(f"  -> Data quality assertions passed for {split_name} split!")


def expand_and_align_dataset(data_dir: str, coco_sources: list[str]) -> None:
    """
    Orchestrates the fetching, taxonomy expansion, and dataset alignment of marine imagery datasets.

    This acts as the primary data entrypoint. It reads local paths or URLs, categorizes 
    them into train/val/test splits, queries WoRMS to construct a unified master taxonomic 
    hierarchy, mutates all input datasets to share a contiguous ID space based on that 
    hierarchy, and ultimately outputs a YOLO-ready directory structure.

    Parameters
    ----------
    data_dir : str
        The root destination directory where the processed datasets and artifacts will live.
    coco_sources : list[str]
        A list of URLs (http://...) or local file paths to the raw COCO JSON files.

    Returns
    -------
    None
    """
    print(f"\n{'='*50}\nInitializing Dataset Expansion & Alignment\n{'='*50}")
    print(f"Target Directory: {data_dir}")
    
    # 1. Setup Directories
    raw_data_dir = os.path.join(data_dir, 'raw_data')
    hierarchy_dir = os.path.join(data_dir, 'hierarchy_data')
    
    for directory in [data_dir, raw_data_dir, hierarchy_dir]:
        os.makedirs(directory, exist_ok=True)
        
    # 2. Fetch/Copy Sources and categorize by split
    raw_datasets = []
    
    for idx, source in enumerate(coco_sources):
        source_lower = source.lower()
        if 'val' in source_lower:
            split = 'val'
        elif 'test' in source_lower:
            split = 'test'
        else:
            split = 'train'
            
        raw_dest = os.path.join(raw_data_dir, f"{split}_{idx}_raw.json")
        
        if not os.path.exists(raw_dest):
            if source.startswith('http://') or source.startswith('https://'):
                print(f"Downloading {split} data from {source}...")
                urllib.request.urlretrieve(source, raw_dest)
            else:
                print(f"Copying {split} data from {source}...")
                shutil.copyfile(source, raw_dest)
                
        with open(raw_dest, 'r') as f:
            coco_dict = json.load(f)
            
            # Cache original mappings for downstream assertions
            cat_map = {cat['id']: cat['name'] for cat in coco_dict.get('categories', [])}
            orig_map = {ann['id']: cat_map[ann['category_id']] for ann in coco_dict.get('annotations', [])}
            
            raw_datasets.append({
                'split': split,
                'idx': idx,
                'dict': coco_dict,
                'orig_map': orig_map
            })

    # 3. Expand & Align the Categories
    print("\nInitializing Expander and fetching/building WoRMS hierarchy...")
    expander = WormsCocoExpander(use_cache=True)
    
    # Build the master tree unifying all taxa found across all discovered datasets
    all_dicts = [ds['dict'] for ds in raw_datasets]
    expander.build_master_hierarchy(*all_dicts)
    
    aligned_paths = {'train': [], 'val': [], 'test': []}
    
    print("\nAligning independent datasets to master taxonomy...")
    for ds in raw_datasets:
        split = ds['split']
        idx = ds['idx']
        print(f"  -> Processing {split.capitalize()} subset ({idx})...")
        
        aligned_coco = expander.align_dataset(ds['dict'])
        
        # 4. Data Quality Assertions
        verify_alignment(ds['orig_map'], aligned_coco, f"{split}_{idx}")
        
        # Save aligned dataset to main directory for YOLO conversion
        aligned_path = os.path.join(data_dir, f"{split}_{idx}_aligned.json")
        with open(aligned_path, 'w') as f:
            json.dump(aligned_coco, f)
            
        aligned_paths[split].append(aligned_path)

    # 5. Save Global Artifacts
    print("\nSaving aligned COCO files and hierarchy artifact...")
    hierarchy_json_path = os.path.join(hierarchy_dir, 'hierarchy.json')
    with open(hierarchy_json_path, 'w') as f:
        json.dump(expander.hierarchy_tree, f, indent=4)

    # 6. Convert to YOLO
    print("\nConverting aligned datasets to YOLO format...")
    # Filter out empty splits to prevent pycocowriter loops
    active_splits = {k: v for k, v in aligned_paths.items() if v}
    pycocowriter.coco2yolo.coco2yolo(active_splits, data_dir)
    
    print(f"\nPipeline Complete. Master datasets and models are ready in: {data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch, expand, and align heterogeneous COCO datasets to a WoRMS taxonomy.")
    parser.add_argument(
        '--data_dir', 
        type=str, 
        required=True,
        help="Path to the target root dataset directory (e.g., ~/datasets/gfisher)"
    )
    parser.add_argument(
        '--coco_sources', 
        type=str, 
        nargs='+',
        required=True,
        help="List of URLs or local file paths to the raw COCO JSON files. Split is inferred from filename (e.g., train/val/test)."
    )
    args = parser.parse_args()
    
    expand_and_align_dataset(args.data_dir, args.coco_sources)
