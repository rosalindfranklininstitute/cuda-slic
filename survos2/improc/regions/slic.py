

import os.path as op

import numpy as np

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from .ccl import ccl3d, merge_small
from ..types import int3, float3
from ..utils import asnparray, gpuregion
from ..cuda import asgpuarray, grid_kernel_config, flat_kernel_config
from ..features import gaussian


__dirname__ = op.dirname(__file__)


@gpuregion
def slic3d(data, nsp=None, sp_shape=None, compactness=30, sigma=None,
           spacing=(1,1,1), max_iter=5, postprocess=True):
    """

    """
    assert data.ndim == 3 or data.ndim == 4
    dshape = np.asarray(data.shape[-3:], int)

    with open(op.join(__dirname__, 'kernels', 'slic3d.cu'), 'r') as f:
        _mod_conv = SourceModule(f.read())
        gpu_slic_init = _mod_conv.get_function('init_clusters')
        gpu_slic_expectation = _mod_conv.get_function('expectation')
        gpu_slic_maximization = _mod_conv.get_function('maximization')

    if sp_shape is not None:
        _sp_shape = list(sp_shape)
        if len(_sp_shape) == 3:
            _sp_grid = (dshape + _sp_shape - 1) // _sp_shape
        else:
            raise ValueError('Incorrect `sp_shape`: {}'.format(sp_shape))
    elif nsp is not None:
        sp_size = int(round((np.prod(data.shape) / nsp)**(1./3.)))
        _sp_shape = list((sp_size, sp_size, sp_size))
        _sp_grid = (dshape + _sp_shape - 1) // _sp_shape
    else:
        raise ValueError('`nsp` or `sp_shape` has to be provided.')

    sp_shape = np.asarray(tuple(_sp_shape[::-1]), int3)
    sp_grid = np.asarray(tuple(_sp_grid[::-1]), int3)

    m = np.float32(compactness)
    S = np.float32(np.prod(_sp_shape))

    n_centers = np.int32(np.prod(_sp_grid))
    n_features = np.int32(data.shape[0] if data.ndim == 4 else 1)
    im_shape = np.asarray(tuple(dshape[::-1]), int3)
    spacing = np.asarray(tuple(spacing[::-1]), float3)

    data_gpu = asgpuarray(data, np.float32)
    centers_gpu = gpuarray.zeros((n_centers, n_features + 3), np.float32)
    labels_gpu = gpuarray.zeros(dshape, np.uint32)

    vblock, vgrid = flat_kernel_config(gpu_slic_init, dshape)
    cblock, cgrid = flat_kernel_config(gpu_slic_init, _sp_grid)

    gpu_slic_init(data_gpu, centers_gpu, n_centers, n_features,
        sp_grid, sp_shape, im_shape, block=cblock, grid=cgrid)

    for _ in range(max_iter):
        gpu_slic_expectation(data_gpu, centers_gpu, labels_gpu, m, S,
            n_centers, n_features, spacing, sp_grid, sp_shape, im_shape,
            block=vblock, grid=vgrid)

        gpu_slic_maximization(data_gpu, labels_gpu, centers_gpu,
            n_centers, n_features, sp_grid, sp_shape, im_shape,
            block=cblock, grid=cgrid)

    r = ccl3d(labels_gpu, remap=True)

    labels = labels_gpu.get()
    binlab = np.bincount(labels.ravel())
    binlab = np.bincount(r.ravel())

    if postprocess:
        min_size = int(np.prod(_sp_shape) / 10.)
        r = merge_small(asnparray(data), r, min_size)
        binlab = np.bincount(r.ravel())

    return r