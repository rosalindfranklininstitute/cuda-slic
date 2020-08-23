import logging
import os.path as op

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from pycuda.compiler import SourceModule

from ._ccl import _merge_small3d, _relabel2d, _relabel3d, _remap
from .cuda import asgpuarray, grid_kernel_config
from .types import int3


__dirname__ = op.dirname(__file__)


# @gpuregion
def ccl3d(labels, remap=True):
    assert labels.ndim == 3
    assert labels.dtype == np.uint32

    with open(op.join(__dirname__, "kernels", "ccl3d.cu"), "r") as f:
        _mod_conv = SourceModule(f.read())
        gpu_ccl_local = _mod_conv.get_function("uf_local")
        gpu_ccl_global = _mod_conv.get_function("uf_global")
        gpu_ccl_final = _mod_conv.get_function("uf_final")

    labels_gpu = asgpuarray(labels, dtype=np.uint32)
    result_gpu = gpuarray.zeros_like(labels_gpu)
    shape = np.asarray(tuple(labels.shape[::-1]), dtype=int3)

    block, grid = grid_kernel_config(gpu_ccl_local, labels.shape)
    shared = int(np.prod(block) * 8)

    gpu_ccl_local(
        labels_gpu, result_gpu, shape, block=block, grid=grid, shared=shared
    )
    gpu_ccl_global(labels_gpu, result_gpu, shape, block=block, grid=grid)
    gpu_ccl_final(result_gpu, shape, block=block, grid=grid)

    if remap:
        return remap_labels(result_gpu.get())

    return result_gpu


def remap_labels(labels):
    assert labels.dtype == np.uint32
    new_labels = _remap(labels.ravel())
    new_labels.shape = labels.shape
    return new_labels


def relabel(labels):
    assert labels.dtype == np.uint32

    if labels.ndim == 2:
        new_labels = _relabel2d(labels.ravel(), labels.shape[1])
    elif labels.ndim == 3:
        new_labels = _relabel3d(
            labels.ravel(), labels.shape[1], labels.shape[2]
        )
    else:
        raise ValueError(
            "Input array has to be 2 or 3 dimensional: {}".format(labels.ndim)
        )

    new_labels.shape = labels.shape
    return new_labels


# @cpuregion
def merge_small(data, labels, min_size=1, **kwargs):
    if data.ndim != labels.ndim + 1:
        data = data[..., None]
    assert data.ndim == labels.ndim + 1
    return _merge_small3d(data, labels, labels.max() + 1, min_size)
