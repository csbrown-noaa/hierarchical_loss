import torch
from collections.abc import Hashable
from .tree_utils import get_roots
from .hierarchy_tensor_utils import (
    build_parent_tensor,
    build_hierarchy_index_tensor,
    build_hierarchy_sibling_mask,
    build_ancestor_sibling_mask,
    build_ancestor_mask
)
from .path_utils import construct_parent_childtensor_tree
from .utils import dict_keyvalue_replace

class Hierarchy:
    """
    Centralizes all hierarchy logic, mapping, and tensor creation.
    """
    def __init__(self, 
                 raw_tree: dict[Hashable, Hashable], 
                 node_to_idx_map: dict[Hashable, int] | None = None,
                 device: torch.device | str | None = None):
        
        all_nodes = set(raw_tree.keys()) | set(raw_tree.values())
        self.raw_tree = raw_tree

        # 1. Build the translation maps
        if node_to_idx_map:
            # Use the provided map
            self.node_to_idx = node_to_idx_map
            # Verify all nodes are accounted for
            for node in all_nodes:
                if node not in self.node_to_idx:
                    raise ValueError(f"Node '{node}' from raw_tree is missing from the provided node_to_idx_map.")
            # Verify node indices are sequential and dense
            idx_vals = node_to_idx_map.values()
            min_idx, max_idx, n_idx = min(idx_vals), max(idx_vals), len(idx_vals)
            if min_idx !=0 or (max_idx != n_idx-1):
                raise ValueError(f"node_to_idx_map must have contiguous sequential indices")
        else:
            # Auto-generate a dense map
            self.node_to_idx = {node: i for i, node in enumerate(all_nodes)}
        
        self.idx_to_node = {i: n for n, i in self.node_to_idx.items()}
        self.num_classes = len(self.node_to_idx)

        # 2. Create the core index-based tree
        self.index_tree = dict_keyvalue_replace(raw_tree, self.node_to_idx)

        # 3. Pre-compute all tensor and dict representations
        # These are now cached for the lifetime of the object.
        self.parent_tensor = build_parent_tensor(self.index_tree, device=device)
        self.index_tensor = build_hierarchy_index_tensor(self.index_tree, device=device)
        self.hierarchy_mask = self.index_tensor == -1
        self.sibling_mask = build_hierarchy_sibling_mask(self.parent_tensor, device=device)
        self.roots = torch.tensor(get_roots(self.index_tree), device=device)
        self.root_mask = torch.zeros(self.num_classes, dtype=torch.bool, device=device).scatter_(0, self.roots, True)
        self.parent_child_tensor_tree = construct_parent_childtensor_tree(self.index_tree, device=device)
        self.ancestor_sibling_mask = build_ancestor_sibling_mask(self.parent_tensor, self.index_tensor, device=device)
        self.ancestor_mask = build_ancestor_mask(self.index_tensor, device=device)

        valid_mask = self.index_tensor != -1
        last_valid_idx = valid_mask.sum(dim=1) - 1
        # Gather the root index for every node
        self.node_to_root = self.index_tensor[torch.arange(self.num_classes, device=device), last_valid_idx]

    def to(self, device: torch.device | str):
        """Moves all computed tensors to the specified device."""
        self.parent_tensor = self.parent_tensor.to(device)
        self.index_tensor = self.index_tensor.to(device)
        self.hierarchy_mask = self.hierarchy_mask.to(device)
        self.sibling_mask = self.sibling_mask.to(device)
        self.roots = self.roots.to(device) 
        self.root_mask = self.root_mask.to(device) 
        self.parent_child_tensor_tree = {k: v.to(device) for k, v in self.parent_child_tensor_tree.items()}
        self.ancestor_sibling_mask = self.ancestor_sibling_mask.to(device)
        self.ancestor_mask = self.ancestor_mask.to(device)
        self.node_to_root = self.node_to_root.to(device)
        return self
