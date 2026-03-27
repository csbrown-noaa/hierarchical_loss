import copy
import os
import urllib.request
import json
import shutil
import pycocowriter.cocomerge
from abc import ABC, abstractmethod

class TaxonomyProvider(ABC):
    """
    Abstract base class for taxonomy providers.
    
    Any class inheriting from this must implement the `build_master_hierarchy` 
    method and populate the `hierarchy_tree` and `name_to_id` attributes.

    Attributes
    ----------
    hierarchy_tree : dict[str, str]
        A mapping of {child_node_name: parent_node_name} representing the tree.
    name_to_id : dict[str, int]
        A mapping of {node_name: unique_integer_id} for all elements in the tree.
    """
    def __init__(self) -> None:
        self.hierarchy_tree: dict[str, str] = {}
        self.name_to_id: dict[str, int] = {}

    @abstractmethod
    def build_master_hierarchy(self, *coco_dicts: dict) -> None:
        """
        Parses inputs (and potentially external APIs) to construct a global 
        taxonomic tree and a master ID mapping.

        Parameters
        ----------
        *coco_dicts : dict
            An arbitrary number of COCO-formatted dictionaries to scan.
        """
        pass


class StaticTaxonomyProvider(TaxonomyProvider):
    """
    A TaxonomyProvider that loads a contrived, pre-computed hierarchy from disk.
    
    Ideal for local datasets like COCO where the tree structure is already known
    and static.

    Parameters
    ----------
    hierarchy_json_path : str
        The absolute or relative path to the pre-computed hierarchy JSON file.
        The JSON should represent a flat dictionary mapping {child: parent}.
    """
    def __init__(self, hierarchy_json_path: str) -> None:
        super().__init__()
        self.hierarchy_json_path: str = hierarchy_json_path

    def build_master_hierarchy(self, *coco_dicts: dict) -> None:
        """
        Loads the static tree, validates it for circular references, and maps 
        all unique nodes to a contiguous 1-to-N ID space.

        Parameters
        ----------
        *coco_dicts : dict
            An arbitrary number of COCO-formatted dictionaries to scan for 
            category validation against the static tree.

        Raises
        ------
        FileNotFoundError
            If the specified hierarchy JSON file does not exist.
        ValueError
            If a circular reference is detected in the provided taxonomy tree.
        """
        if not os.path.exists(self.hierarchy_json_path):
            raise FileNotFoundError(f"Static hierarchy file not found: {self.hierarchy_json_path}")
            
        with open(self.hierarchy_json_path, 'r') as f:
            self.hierarchy_tree = json.load(f)

        # 1. Validate the tree (check for circular references)
        for node in self.hierarchy_tree:
            visited = set()
            current = node
            while current in self.hierarchy_tree:
                if current in visited:
                    raise ValueError(f"CRITICAL: Circular reference detected in static hierarchy at '{current}'")
                visited.add(current)
                parent = self.hierarchy_tree[current]
                if parent == current:  # Safely handle explicit root self-loops
                    break
                current = parent

        # 2. Extract all unique nodes from the tree
        all_taxa = set(self.hierarchy_tree.keys()).union(set(self.hierarchy_tree.values()))
        
        # 3. Validation: Ensure all categories in the datasets actually exist in the tree
        for c_dict in coco_dicts:
            for cat in c_dict.get('categories', []):
                cat_name = cat['name']
                if cat_name not in all_taxa:
                    print(f"WARNING: Dataset category '{cat_name}' is missing from the static hierarchy. Adding as an orphaned root node.")
                    all_taxa.add(cat_name)

        # 4. Generate the master contiguous name-to-ID mapping alphabetically
        self.name_to_id = {name: i + 1 for i, name in enumerate(sorted(all_taxa))}
        print(f"Static taxonomy loaded successfully: {len(self.name_to_id)} total unique nodes.")


class HierarchicalCocoAligner:
    """
    Universally aligns any COCO dataset to a provided taxonomic hierarchy.

    This class relies on a pre-computed taxonomy tree and name-to-id mapping 
    (provided by a TaxonomyProvider). It injects this master tree into the 
    dataset's categories, merges duplicate taxa names into unified IDs, and 
    assigns perfectly contiguous (1 to N) IDs alphabetically to satisfy downstream 
    computer vision frameworks.

    Parameters
    ----------
    hierarchy_tree : dict[str, str]
        A mapping of {child_node: parent_node} representing the taxonomy.
    name_to_id : dict[str, int]
        A mapping of {node_name: unique_integer_id} for all elements in the tree.
        
    Attributes
    ----------
    master_coco_dummy : dict
        An internal empty COCO structure containing the superset of all taxonomic 
        categories discovered, sorted alphabetically.
    """
    def __init__(self, hierarchy_tree: dict[str, str], name_to_id: dict[str, int]) -> None:
        self.hierarchy_tree: dict[str, str] = hierarchy_tree
        self.name_to_id: dict[str, int] = name_to_id
        
        self.master_coco_dummy: dict = {
            "images": [], 
            "annotations": [], 
            "licenses": [], 
            "categories": [
                {'id': i + 1, 'name': name} 
                for i, name in enumerate(sorted(self.name_to_id.keys()))
            ]
        }

    def align_dataset(self, target_coco: dict) -> dict:
        """
        Aligns a single COCO dataset to the unified master hierarchy.

        Parameters
        ----------
        target_coco : dict
            The target COCO dictionary to be aligned.

        Returns
        -------
        dict
            A new, fully aligned COCO dictionary.

        Examples
        --------
        >>> aligner = HierarchicalCocoAligner(
        ...     hierarchy_tree={'dog': 'animal', 'animal': 'animal'},
        ...     name_to_id={'animal': 1, 'dog': 2}
        ... )
        >>> raw_dataset = {
        ...     'categories': [{'id': 99, 'name': 'dog'}],
        ...     'annotations': [{'id': 1, 'category_id': 99, 'image_id': 1}]
        ... }
        >>> aligned = aligner.align_dataset(raw_dataset)
        >>> aligned['annotations'][0]['category_id']
        2
        """
        # Deepcopy ONLY the small master dummy to protect it across multiple dataset alignments
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


def verify_alignment(orig_mapping: list[str], new_coco: dict, split_name: str) -> None:
    """
    Verifies that the aligned annotations mathematically mapped to the correct 
    taxonomic names, using sequential order rather than IDs.

    Parameters
    ----------
    orig_mapping : list[str]
        A list containing the original `category_name` for each annotation, in sequence.
    new_coco : dict
        The newly aligned COCO dictionary to be verified.
    split_name : str
        The string identifier for the dataset split (used for logging output).

    Raises
    ------
    AssertionError
        If any annotation's mapped category name differs from its original name,
        or if the total number of annotations changes.

    Examples
    --------
    >>> orig_map = ["dog", "cat"]
    >>> aligned_coco = {
    ...     "categories": [{"id": 10, "name": "cat"}, {"id": 20, "name": "dog"}],
    ...     "annotations": [
    ...         {"id": 1, "category_id": 20},
    ...         {"id": 2, "category_id": 10}
    ...     ]
    ... }
    >>> verify_alignment(orig_map, aligned_coco, "test_split")
      -> Data quality assertions passed for test_split split!
    """
    category_map_new = {cat['id']: cat for cat in new_coco.get('categories', [])}
    new_annotations = new_coco.get('annotations', [])
    
    assert len(orig_mapping) == len(new_annotations), f"Annotation count mismatch in {split_name} split!"
    
    for i, ann in enumerate(new_annotations):
        old_cat_name = orig_mapping[i]
        new_cat_name = category_map_new[ann['category_id']]['name']
        assert old_cat_name == new_cat_name, f"Category mapping failed at index {i}: {old_cat_name} != {new_cat_name}"
        
    print(f"  -> Data quality assertions passed for {split_name} split!")


def process_hierarchical_dataset(data_dir: str, coco_sources: list[str], taxonomy_provider: TaxonomyProvider) -> None:
    """
    The universal orchestration pipeline for fetching, taxonomy expansion, 
    and dataset alignment of hierarchical imagery datasets.

    Parameters
    ----------
    data_dir : str
        The root destination directory where the processed datasets and artifacts will live.
    coco_sources : list[str]
        A list of URLs (http://...) or local file paths to the raw COCO JSON files.
    taxonomy_provider : TaxonomyProvider
        An instantiated TaxonomyProvider that exposes a `build_master_hierarchy(*dicts)` 
        method, and `hierarchy_tree` / `name_to_id` attributes.
    """
    print(f"\n{'='*50}\nInitializing Dataset Expansion & Alignment\n{'='*50}")
    print(f"Target Directory: {data_dir}")
    print(f"Taxonomy Provider: {taxonomy_provider.__class__.__name__}")
    
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
            orig_map = [cat_map[ann['category_id']] for ann in coco_dict.get('annotations', [])]
            
            raw_datasets.append({
                'split': split,
                'idx': idx,
                'dict': coco_dict,
                'orig_map': orig_map
            })

    # 3. Build the Master Taxonomy
    print(f"\nBuilding Master Taxonomy using {taxonomy_provider.__class__.__name__}...")
    all_dicts = [ds['dict'] for ds in raw_datasets]
    taxonomy_provider.build_master_hierarchy(*all_dicts)
    
    # 4. Initialize Universal Aligner
    aligner = HierarchicalCocoAligner(
        hierarchy_tree=taxonomy_provider.hierarchy_tree,
        name_to_id=taxonomy_provider.name_to_id
    )
    
    aligned_paths = {'train': [], 'val': [], 'test': []}
    
    print("\nAligning independent datasets to master taxonomy...")
    for ds in raw_datasets:
        split = ds['split']
        idx = ds['idx']
        print(f"  -> Processing {split.capitalize()} subset ({idx})...")
        
        aligned_coco = aligner.align_dataset(ds['dict'])
        
        # 5. Data Quality Assertions
        verify_alignment(ds['orig_map'], aligned_coco, f"{split}_{idx}")
        
        # Save aligned dataset to main directory for YOLO conversion
        aligned_path = os.path.join(data_dir, f"{split}_{idx}_aligned.json")
        with open(aligned_path, 'w') as f:
            json.dump(aligned_coco, f)
            
        aligned_paths[split].append(aligned_path)

    # 6. Save Global Artifacts
    print("\nSaving aligned COCO files and hierarchy artifact...")
    hierarchy_json_path = os.path.join(hierarchy_dir, 'hierarchy.json')
    with open(hierarchy_json_path, 'w') as f:
        json.dump(aligner.hierarchy_tree, f, indent=4)
    
    print(f"\nPipeline Complete. Master datasets and models are ready in: {data_dir}")
