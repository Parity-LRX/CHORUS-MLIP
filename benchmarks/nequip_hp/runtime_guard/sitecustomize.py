"""Keep old optional PyG ABI wheels out of the Torch 2.11 OEQ runtime."""
import importlib.abc
import sys


class _OldABIBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.',1)[0] in {
            'torch_cluster','torch_scatter','torch_sparse','torch_spline_conv','pyg_lib',
            'cuequivariance','cuequivariance_torch',
        }:
            raise ImportError('Optional old-ABI/excluded extension: '+fullname)
        return None


sys.meta_path.insert(0,_OldABIBlocker())
