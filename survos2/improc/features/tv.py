

import os.path as op

import numpy as np

import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from ..types import int3
from ..utils import gpufeature
from ..cuda import asgpuarray, flat_kernel_config


__dirname__ = op.dirname(__file__)


@gpufeature
def tvdenoising3d(data, lamda=15, max_iter=100):
    assert data.ndim == 3

    with open(op.join(__dirname__, 'kernels', 'tv.cu'), 'r') as f:
        _mod_tv = SourceModule(f.read())
        gpu_tv_u = _mod_tv.get_function('update_u')
        gpu_tv_p = _mod_tv.get_function('update_p')

    dsize = np.prod(data.shape)

    f_gpu = asgpuarray(data)
    u_gpu = f_gpu.copy()
    z_gpu = gpuarray.zeros_like(f_gpu)
    y_gpu = gpuarray.zeros_like(f_gpu)
    x_gpu = gpuarray.zeros_like(f_gpu)

    lamda = np.float32(1.0 / lamda)
    #z, y, x = map(np.int32, data.shape)
    shape = np.asarray(data.shape[::-1], dtype=int3)
    mtpb = gpu_tv_u.max_threads_per_block
    block, grid = flat_kernel_config(gpu_tv_u, data.shape)

    for i in range(max_iter):
        tau2 = np.float32(0.3 + 0.02 * i)
        tau1 = np.float32((1. / tau2) * ((1. / 6.) - (5. / (15. + i))))

        gpu_tv_u(f_gpu, z_gpu, y_gpu, x_gpu, u_gpu, tau1, lamda, shape,
            block=block, grid=grid)
        gpu_tv_p(u_gpu, z_gpu, y_gpu, x_gpu, tau2, shape,
            block=block, grid=grid)

    return u_gpu