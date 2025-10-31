import utils
import functools
import unittest

import hierarchical_loss.utils
import hierarchical_loss.coco_utils
import hierarchical_loss.hierarchical_loss
import hierarchical_loss.hierarchy_tensor_utils
import hierarchical_loss.path_utils
import hierarchical_loss.tree_utils
import hierarchical_loss.viz_utils
import hierarchical_loss.worms_utils

MODULES = [
    hierarchical_loss.utils,
    hierarchical_loss.coco_utils,
    hierarchical_loss.hierarchical_loss,
    hierarchical_loss.hierarchy_tensor_utils,
    hierarchical_loss.path_utils,
    hierarchical_loss.tree_utils,
    hierarchical_loss.viz_utils,
    hierarchical_loss.worms_utils,
]

class TestEmpty(unittest.TestCase):
    def test_empty(self):
        pass

def load_tests(loader, tests, ignore):
    return functools.reduce(lambda tests_so_far, module: utils.doctests(module, tests_so_far), MODULES, tests)

if __name__ == '__main__':
    unittest.main()
