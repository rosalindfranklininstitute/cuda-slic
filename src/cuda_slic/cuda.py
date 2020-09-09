import atexit
import logging

from functools import wraps

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from .utils import asnparray


__cuda_started__ = False


def init_cuda():
    global __cuda_started__
    if not __cuda_started__:
        cuda.init()
        __cuda_started__ = True


def create_context(gpu_id=0):
    init_cuda()
    device = cuda.Device(gpu_id)
    return device.make_context()


def gpu_argument_wrapper(func, dtype=None):
    """
    The wrapped function will get an extra argument `keep_gpu`. If `True`
    the result of the function will be kept in the gpu, or returned as a
    numpy array otherwise.
    """

    @wraps(func)
    def wrapper(*args, keep_gpu=False, **kwargs):
        r = func(*args, **kwargs)
        return asgpuarray(r, dtype) if keep_gpu else asnparray(r, dtype)

    return wrapper


def gpu_context_wrapper(func):
    """
    The wrapped will generate its own context to be runned in the GPU,
    allowing chunking operations to run simultaneously in the GPU.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        gpucontext = create_context(kwargs.pop("gpu_id", 0))
        r = func(*args, **kwargs)
        if kwargs.get("keep_gpu", False):
            atexit.register(gpucontext.detach)
        else:
            gpucontext.detach()
        return r

    return wrapper


def normalize(array):
    array = asgpuarray(array)
    array -= gpuarray.min(array)
    array /= gpuarray.max(array)
    return array


def asgpuarray(data, dtype=None):
    dtype = dtype or data.dtype

    if isinstance(data, gpuarray.GPUArray):
        if np.dtype(data.dtype) != np.dtype(dtype):
            raise ValueError(
                "Data type `{}` does not match expected type `{}`".format(
                    np.dtype(data.dtype).name, np.dtype(dtype).name
                )
            )
        return data
    if np.dtype(data.dtype) != np.dtype(dtype):
        logging.warn(
            "Probably unsafe type casting to CUDA array: {} to {}".format(
                np.dtype(data.dtype).name, np.dtype(dtype).name
            )
        )

    return gpuarray.to_gpu(asnparray(data, dtype=dtype))


def flat_kernel_config(kernel, shape):
    data_size = int(np.prod(shape))
    # max_threads = kernel.max_threads_per_block
    max_threads = 128

    block = (int(max_threads), 1, 1)
    grid = ((data_size + max_threads - 1) // max_threads, 1, 1)

    return block, grid


def grid_kernel_config(kernel, shape, isotropic=False):
    if isotropic in [True, False]:
        block = np.asarray(shape[::-1])  # z, y, x -> x, y, z
    else:
        iso = np.asarray(isotropic[::-1], np.float32)
        iso /= np.abs(iso).sum()
        block = max(*shape) * iso

    if block.size == 2:
        block = np.r_[block, 1]

    # max_threads = kernel.max_threads_per_block
    max_threads = 128

    if isotropic is not True:
        while np.prod(block) > max_threads:
            block = np.maximum(1, np.round(block - 0.1 * block)).astype(int)
    else:
        max_axis = np.floor(max_threads ** (1.0 / len(shape)))
        block = [max_axis] * len(shape) + [1] * (3 - len(shape))

    block = tuple(map(int, block))
    grid = tuple(
        map(
            int,
            (
                (shape[2] + block[0] - 1) // block[0],
                (shape[1] + block[1] - 1) // block[1],
                (shape[0] + block[2] - 1) // block[2],
            ),
        )
    )

    return block, grid


def to_tex3d(data):
    """
    Source: https://wiki.tiker.net/PyCUDA/Examples/Demo3DSurface
    """
    d, h, w = data.shape
    descr = cuda.ArrayDescriptor3D()
    descr.width = w
    descr.height = h
    descr.depth = d
    descr.format = cuda.dtype_to_array_format(data.dtype)
    descr.num_channels = 1
    descr.flags = 0

    if isinstance(data, gpuarray.GPUArray):
        data = data.get()

    device_array = cuda.Array(descr)
    copy = cuda.Memcpy3D()
    copy.set_src_host(data)
    copy.set_dst_array(device_array)
    copy.width_in_bytes = copy.src_pitch = data.strides[1]
    copy.src_height = copy.height = h
    copy.depth = d
    copy()

    return device_array
