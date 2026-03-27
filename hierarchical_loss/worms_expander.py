#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from . import hierarchy_expander
from . import worms_utils

class WormsTaxonomyProvider:
    """
    A TaxonomyProvider that standardizes and expands species names to a unified 
    WoRMS taxonomy hierarchy via the official API.

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
        A mapping of taxonomic names to their official contiguous model ID.
    """

    def __init__(self, use_cache: bool = True) -> None:
        if use_cache:
            import requests_cache
            # Cache expires after 30 days; helps significantly with API rate limiting/flakiness
            requests_cache.install_cache('worms_api_cache', backend='sqlite', expire_after=2592000)
            
        self.hierarchy_tree: dict[str, str] = {}
        self.name_to_id: dict[str, int] = {}
        
        # Internal state for tracking resolution queues
        self._to_lookup: set[str] = set()
        self._deferred: set[str] = set()

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

    def _extract_unique_names(self, *coco_dicts: dict) -> None:
        """
        Collects all unique taxonomic names from the datasets and initializes state.

        Parameters
        ----------
        *coco_dicts : dict
            The datasets to extract category names from.
        """
        self._to_lookup.clear()
        self._deferred.clear()
        self.hierarchy_tree.clear()
        self.name_to_id.clear()

        for c_dict in coco_dicts:
            for cat in c_dict.get('categories', []):
                self._to_lookup.add(cat['name'])

    def _process_graph_discovery(self) -> None:
        """
        Pops names, performs optimistic ID fetches via the WoRMS API, and 
        retroactively resolves local caches using tree context.
        """
        while self._to_lookup:
            name = self._to_lookup.pop()
            
            # Optimistic ID fetch
            try:
                aphia_id = worms_utils.get_WORMS_id(name)
            except ValueError:
                self._deferred.add(name)
                continue

            if aphia_id < 0:
                self._deferred.add(name)
                continue
                
            # Tree Harvest (If ID succeeded)
            try:
                tree = worms_utils.get_WORMS_tree(aphia_id)
            except ValueError as e:
                raise RuntimeError(
                    f"CRITICAL: Failed to fetch tree for successfully resolved ID {aphia_id} (Name: {name}). "
                    f"Underlying error: {e}"
                )

            # Walk the tree and extract all nodes
            new_hierarchy, new_name_id_map = worms_utils.WORMS_tree_to_name_hierarchy([tree])
            
            self.hierarchy_tree.update(new_hierarchy)
            self.name_to_id.update(new_name_id_map)
            
            # Retroactively clear all discovered contextual nodes from our sets
            for discovered_name in new_name_id_map.keys():
                self._to_lookup.discard(discovered_name)
                self._deferred.discard(discovered_name)

    def _rescue_orphans(self) -> None:
        """
        Attempts to rescue true orphans using strict exact-match accepted-status 
        disambiguation via the WoRMS records API.
        """
        for orphan_name in list(self._deferred):
            try:
                rescued_id = worms_utils.disambiguate_taxon(orphan_name)
                
                # If rescue succeeds, we must fetch its tree to capture its parent hierarchy
                tree = worms_utils.get_WORMS_tree(rescued_id)
                new_hierarchy, new_name_id_map = worms_utils.WORMS_tree_to_name_hierarchy([tree])
                
                self.hierarchy_tree.update(new_hierarchy)
                self.name_to_id.update(new_name_id_map)
                
                self._deferred.remove(orphan_name)
            except ValueError:
                pass

    def _log_true_orphans(self) -> None:
        """
        Prints warnings for any names that survived both contextual and rescue passes.
        """
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


def expand_and_align_dataset(data_dir: str, coco_sources: list[str]) -> None:
    """
    Maintains API compatibility with external orchestrator scripts. 
    Injects the WormsTaxonomyProvider into the universal hierarchical dataset processor.

    Parameters
    ----------
    data_dir : str
        The root destination directory where the processed datasets and artifacts will live.
    coco_sources : list[str]
        A list of URLs (http://...) or local file paths to the raw COCO JSON files.

    Examples
    --------
    >>> expand_and_align_dataset(
    ...     data_dir="./output/gfisher",
    ...     coco_sources=["http://example.com/train.json"]
    ... )
    """
    provider = WormsTaxonomyProvider(use_cache=True)
    hierarchy_expander.process_hierarchical_dataset(
        data_dir=data_dir, 
        coco_sources=coco_sources, 
        taxonomy_provider=provider
    )


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
