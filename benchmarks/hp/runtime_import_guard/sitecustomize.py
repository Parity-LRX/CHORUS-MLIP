"""Exclude optional wheels compiled for a different Torch ABI in the overlay.

The inference boundary receives complete ASE cutoff edges. Optional neighbour
search/scatter wheels are not needed; the reference uses its existing PyTorch
fallback. This file is enabled only by the isolated remote batch environment.
"""
import importlib.abc
import os
import sys


class _IncompatibleOptionalWheels(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        roots=('torch_cluster','torch_scatter','torch_sparse','torch_spline_conv','pyg_lib',
               'cuequivariance','cuequivariance_torch','openequivariance')
        if fullname.split('.')[0] in roots:
            raise ModuleNotFoundError('Optional external/old-ABI wheel excluded from the isolated inference runtime',name=fullname)
        return None


if os.environ.get('CHORUS_DISABLE_ABI_EXTENSIONS')=='1':
    sys.meta_path.insert(0,_IncompatibleOptionalWheels())
