import copy
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
    A TaxonomyProvider that loads a contrived, pre-computed hierarchy dictionary.
    
    Ideal for local datasets like COCO where the tree structure is already known
    and static.

    Parameters
    ----------
    hierarchy_tree : dict[str, str]
        A flat dictionary mapping {child: parent} representing the taxonomy.
    """
    def __init__(self, hierarchy_tree: dict[str, str]) -> None:
        super().__init__()
        self.hierarchy_tree = hierarchy_tree

    def build_master_hierarchy(self, *coco_dicts: dict) -> None:
        """
        Validates the static tree for circular references, and maps 
        all unique nodes to a contiguous 1-to-N ID space.

        Parameters
        ----------
        *coco_dicts : dict
            An arbitrary number of COCO-formatted dictionaries to scan for 
            category validation against the static tree.

        Raises
        ------
        ValueError
            If a circular reference is detected in the provided taxonomy tree.
        """
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


def verify_alignment(orig_mapping: list[str], new_coco: dict, dataset_name: str) -> None:
    """
    Verifies that the aligned annotations mathematically mapped to the correct 
    taxonomic names, using sequential order rather than IDs.

    Parameters
    ----------
    orig_mapping : list[str]
        A list containing the original `category_name` for each annotation, in sequence.
    new_coco : dict
        The newly aligned COCO dictionary to be verified.
    dataset_name : str
        The string identifier for the dataset (used for logging output).

    Raises
    ------
    AssertionError
        If any annotation's mapped category name differs from its original name,
        or if the total number of annotations changes.
    """
    category_map_new = {cat['id']: cat for cat in new_coco.get('categories', [])}
    new_annotations = new_coco.get('annotations', [])
    
    assert len(orig_mapping) == len(new_annotations), f"Annotation count mismatch in {dataset_name}!"
    
    for i, ann in enumerate(new_annotations):
        old_cat_name = orig_mapping[i]
        new_cat_name = category_map_new[ann['category_id']]['name']
        assert old_cat_name == new_cat_name, f"Category mapping failed at index {i}: {old_cat_name} != {new_cat_name}"
        
    print(f"  -> Data quality assertions passed for dataset '{dataset_name}'!")


def align_coco_dictionaries(coco_dicts: list[dict], taxonomy_provider: TaxonomyProvider) -> tuple[list[dict], dict[str, str]]:
    """
    The universal orchestration pipeline for taxonomy expansion and dataset 
    alignment of hierarchical imagery datasets in pure memory.

    Parameters
    ----------
    coco_dicts : list[dict]
        A list of COCO-formatted dictionaries to be aligned to a single taxonomy.
    taxonomy_provider : TaxonomyProvider
        An instantiated TaxonomyProvider that exposes a `build_master_hierarchy(*dicts)` 
        method, and `hierarchy_tree` / `name_to_id` attributes.
        
    Returns
    -------
    tuple[list[dict], dict[str, str]]
        A tuple containing:
        1. The list of newly aligned COCO dictionaries (in the same order).
        2. The unified master taxonomic hierarchy tree mapping.
    """
    print(f"\n{'='*50}\nInitializing Dataset Expansion & Alignment\n{'='*50}")
    print(f"Taxonomy Provider: {taxonomy_provider.__class__.__name__}")
    
    # 1. Cache original mappings for downstream assertions
    orig_maps = []
    for c_dict in coco_dicts:
        cat_map = {cat['id']: cat['name'] for cat in c_dict.get('categories', [])}
        orig_maps.append([cat_map[ann['category_id']] for ann in c_dict.get('annotations', [])])

    # 2. Build the Master Taxonomy
    print(f"\nBuilding Master Taxonomy using {taxonomy_provider.__class__.__name__}...")
    taxonomy_provider.build_master_hierarchy(*coco_dicts)
    
    # 3. Initialize Universal Aligner
    aligner = HierarchicalCocoAligner(
        hierarchy_tree=taxonomy_provider.hierarchy_tree,
        name_to_id=taxonomy_provider.name_to_id
    )
    
    print("\nAligning independent datasets to master taxonomy...")
    aligned_dicts = []
    for idx, (c_dict, orig_map) in enumerate(zip(coco_dicts, orig_maps)):
        print(f"  -> Processing dictionary subset ({idx})...")
        
        aligned_coco = aligner.align_dataset(c_dict)
        verify_alignment(orig_map, aligned_coco, f"subset_{idx}")
        aligned_dicts.append(aligned_coco)

    print(f"\nAlignment Complete. Returned {len(aligned_dicts)} aligned dictionaries.")
    return aligned_dicts, aligner.hierarchy_tree
