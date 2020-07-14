

import os
import sys
import os.path as op
from math import ceil, sqrt

import numpy as np

# CUDA
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# Decomposition
from io import StringIO
sys.stderr = StringIO()
os.environ['TENSORLY_BACKEND'] = 'numpy'
from tensorly import tensor, to_numpy
from tensorly.decomposition import non_negative_parafac
sys.stderr = sys.__stderr__

from ..types import int3
from ..utils import gpufeature
from ..cuda import asgpuarray, to_tex3d, grid_kernel_config


__cudafile__ = op.join(op.dirname(__file__), 'kernels', 'conv.cu')


##############################################################################
# Convolution Kernels
##############################################################################

@gpufeature
def conv(data, kernel=None, separate=True):
    assert data.ndim == kernel.ndim

    if data.ndim == 2:
        raise NotImplementedError('2D convolutions not implemented in the GPU')
    elif data.ndim == 3:
        return _conv3d(data, kernel, separate=separate)
    else:
        raise ValueError('Input dataset has to be 2 or 3 dimensional: {}'
                         .format(data.ndim))


def _conv3d(data, kernel, separate=True):
    if separate:
        kernels = separate_kernel(kernel)
        return conv3d_multiple(data, kernels)

    return conv3d_tex(data, kernel)


@gpufeature
def conv3d_tex(data, kernel=None):
    assert data.ndim == 3 and kernel.ndim == 3

    with open(__cudafile__, 'r') as f:
        _mod_conv = SourceModule(f.read())
        gpu_conv3d_t = _mod_conv.get_function('conv3d_tex')
        gpu_conv3d_tex1 = _mod_conv.get_texref('texSrc')
        gpu_conv3d_tex2 = _mod_conv.get_texref('texK')

    im_shape = np.asarray(data.shape[::-1], dtype=int3)
    k_radius = np.asarray(tuple(k//2 for k in kernel.shape[::-1]), dtype=int3)

    data_tex = to_tex3d(data)
    gpu_conv3d_tex1.set_array(data_tex)
    gpu_conv3d_tex1.set_address_mode(0, cuda.address_mode.WRAP)
    gpu_conv3d_tex1.set_address_mode(1, cuda.address_mode.WRAP)
    gpu_conv3d_tex1.set_address_mode(2, cuda.address_mode.WRAP)

    kernel_tex = to_tex3d(kernel)
    gpu_conv3d_tex2.set_array(kernel_tex)

    r_gpu = gpuarray.zeros(data.shape, np.float32)

    block, grid = grid_kernel_config(gpu_conv3d_t, data.shape,
                                     isotropic=kernel.shape)

    gpu_conv3d_t(r_gpu, im_shape, k_radius,
        block=block, grid=grid, texrefs=[gpu_conv3d_tex1, gpu_conv3d_tex2])

    return r_gpu


@gpufeature
def conv_sep(data, kernels=None):
    assert data.ndim == len(kernels)
    if data.ndim == 2:
        pass  # TODO
    elif data.ndim == 3:
        return _conv3d_sep(data, *kernels)
    else:
        raise ValueError('Input data has to be 2 or 3 dimensional: {}'
                         .format(data.ndim))


def _conv3d_sep(data, kz, ky, kx):
    assert data.ndim == 3

    with open(__cudafile__, 'r') as f:
        _mod_conv = SourceModule(f.read())
        gpu_conv3d_0 = _mod_conv.get_function('conv3d_axis0')
        gpu_conv3d_1 = _mod_conv.get_function('conv3d_axis1')
        gpu_conv3d_2 = _mod_conv.get_function('conv3d_axis2')

    d_gpu = asgpuarray(data)
    kz_gpu = asgpuarray(kz, np.float32)
    ky_gpu = asgpuarray(ky, np.float32)
    kx_gpu = asgpuarray(kx, np.float32)
    r1_gpu = gpuarray.zeros_like(d_gpu)
    r2_gpu = gpuarray.zeros_like(d_gpu)

    shape = np.asarray(data.shape[::-1], dtype=int3)
    block, grid = grid_kernel_config(gpu_conv3d_0, data.shape)

    gpu_conv3d_0(d_gpu, kz_gpu, r1_gpu, shape, np.int32(kz.size//2),
        block=block, grid=grid, shared=(kz.size * kz.itemsize))
    gpu_conv3d_1(r1_gpu, ky_gpu, r2_gpu, shape, np.int32(ky.size//2),
        block=block, grid=grid, shared=(ky.size * ky.itemsize))
    gpu_conv3d_2(r2_gpu, kx_gpu, r1_gpu, shape, np.int32(kx.size//2),
        block=block, grid=grid, shared=(kx.size * kx.itemsize))

    return r1_gpu


@gpufeature
def conv3d_multiple(data, kernels=None):
    d_gpu = asgpuarray(data)
    r_gpu = gpuarray.zeros_like(d_gpu)

    Kz, Ky, Kx = kernels

    for i in range(Kz.shape[0]):
        kz = to_numpy(Kz[i]).astype(np.float32)
        ky = to_numpy(Ky[i]).astype(np.float32)
        kx = to_numpy(Kx[i]).astype(np.float32)
        r_gpu += conv_sep(d_gpu, [kz, ky, kx])

    return r_gpu


##############################################################################
# Auxiliary Kernel generation functions
##############################################################################

def _make_rotated_grid(shape, orient):
    r = [s//2 for s in shape]
    r = [(-r[i], r[i]+1) if 2 * r[i] + 1 == shape[i] else (-r[i], r[i])
         for i in range(len(r))]
    z, y, x = np.mgrid[r[0][0]:r[0][1], r[1][0]:r[1][1], r[2][0]:r[2][1]]

    orgpts = np.vstack([z.ravel(), y.ravel(), x.ravel()])
    a, b = orient
    rotmz = [
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a), np.cos(a)]
    ]
    rotmy = [
        [np.cos(b), 0, np.sin(b)],
        [0, 1, 0],
        [-np.sin(b), 0, np.cos(b)]
    ]
    rotpts = np.dot(rotmy, np.dot(rotmz, orgpts))
    return [rotpts[i, :].reshape(z.shape) for i in range(3)]


def make_gaussian_1d(sigma=1., size=None, order=0, trunc=3):
    if size is None:
        size = sigma * trunc * 2 + 1
    x = np.arange(-(size//2), (size//2)+1)
    if order > 2:
        raise ValueError("Only orders up to 2 are supported")
    # compute unnormalized Gaussian response
    response = np.exp(-x ** 2 / (2. * sigma ** 2))
    if order == 1:
        response = -response * x
    elif order == 2:
        response = response * (x ** 2 - sigma ** 2)
    # normalize
    response /= np.abs(response).sum()
    return response.astype(np.float32, copy=False)


def make_gaussian_3d(sigma, size=None, order=(0, 0, 0), ori=(0, 0), trunc=3):
    if type(sigma) == tuple or type(sigma) == list:
        sz, sy, sx = sigma
    else:
        sz, sy, sx = (sigma * (1. + 2. * o) for o in order)

    if size is None:
        shape = sz * trunc * 2 + 1, sy * trunc * 2 + 1, sx * trunc * 2 + 1
    else:
        shape = size, size, size

    rotz, roty, rotx = _make_rotated_grid(shape, ori)

    g = np.exp(-0.5 * (rotx ** 2 / sx ** 2 + roty ** 2 / sy ** 2 +
                       rotz ** 2 / sz ** 2))
    g /= 2 * np.pi * sx * sy * sz

    for o, s, x in zip(order, (sz, sy, sx), (rotz, roty, rotx)):
        if o == 1:
            g *= -x
        elif o == 2:
            g *= (x**2 - s**2)

    g /= np.abs(g).sum()

    return g.astype(np.float32, copy=False)


def make_gabor_3d(shape, sigmas, frequency, offset=0, orient=(0, 0),
                  return_real=True):
    sz, sy, sx = sigmas
    rotz, roty, rotx = _make_rotated_grid(shape, orient)

    g = np.zeros(shape, dtype=np.complex)
    g = np.exp(-0.5 * (rotx ** 2 / sx ** 2 + roty ** 2 / sy ** 2 +
                       rotz ** 2 / sz ** 2))
    g /= 2 * np.pi * sx * sy * sz
    g *= np.exp(1j * (2 * np.pi * frequency * rotx + offset))

    if return_real:
        return np.real(g)

    return g.astype(np.float32, copy=False)


def _rec_2d(D, weights=None):
    if weights is None:
        weights = np.ones(len(D), D[0][0].dtype)
    R = np.zeros((D[0][0].shape[0], D[0][1].shape[0]))
    for i in range(len(D)):
        R += D[i][0][:, None] * D[i][1][None, :] * weights[i]
    return R


def _rec_3d(D, weights=None):
    if weights is None:
        weights = np.ones(len(D), D[0][0].dtype)
    R = np.zeros((D[0][0].shape[0], D[0][1].shape[0], D[0][2].shape[0]))
    for i in range(len(D)):
        R += D[i][0][:, None, None] * D[i][1][None, :, None] \
             * D[i][2][None, None, :] * weights[i]
    return R


def _rec_error(X, R, mean=True):
    if mean:
        return np.mean((X - R)**2)
    else:
        return np.sum((X - R)**2)


def separate_kernel(kernel, max_rank='sqrt'):
    if max_rank is None:
        max_rank = min(*kernel.shape)
    elif max_rank == 'sqrt':
        max_rank = int(ceil(sqrt(max(*kernel.shape))))
    else:
        max_rank = min(*kernel.shape) // 3

    if kernel.ndim == 1:
        return kernel
    else:
        P = non_negative_parafac(tensor(kernel), rank=max_rank)
        return [D.T for D in P]
