import os.path as op

import numpy as np
from jinja2 import Template

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule

from .types import int3, float3

from skimage.segmentation.slic_superpixels import _enforce_label_connectivity_cython


def flat_kernel_config(threads_total, block_size=128):
    block = (block_size, 1, 1)
    grid = ((threads_total + block_size - 1) // block_size, 1, 1)
    return block, grid


def slic3d(
    image,
    n_segments=100,
    sp_shape=None,
    compactness=1.0,
    sigma=None,
    spacing=(1, 1, 1),
    max_iter=5,
    postprocess=True,
):
    """

    """
    if image.ndim not in [3, 4]:
        raise ValueError(
            (
                "input image must be either 3, or 4 dimention."
                "the image.ndim provided is {}".format(image.ndim)
            )
        )
    dshape = np.array(image.shape[-3:])

    if sp_shape:
        if isinstance(sp_shape, int):
            _sp_shape = np.array([sp_shape, sp_shape, sp_shape])

        elif len(sp_shape) == 3 and isinstance(sp_shape, tuple):
            _sp_shape = np.array(sp_shape)
        else:
            raise ValueError(("sp_shape must be scalar int or tuple of length 3"))

        _sp_grid = (dshape + _sp_shape - 1) // _sp_shape

    else:
        sp_size = int(np.ceil((np.prod(dshape) / n_segments) ** (1.0 / 3.0)))
        _sp_shape = np.array([sp_size, sp_size, sp_size])
        _sp_grid = (dshape + _sp_shape - 1) // _sp_shape

    sp_shape = np.asarray(tuple(_sp_shape[::-1]), int3)
    sp_grid = np.asarray(tuple(_sp_grid[::-1]), int3)

    m = np.float32(compactness)
    S = np.float32(np.max(_sp_shape))

    n_centers = np.int32(np.prod(_sp_grid))
    n_features = np.int32(image.shape[0] if image.ndim == 4 else 1)
    im_shape = np.asarray(tuple(dshape[::-1]), int3)
    spacing = np.asarray(tuple(spacing[::-1]), float3)

    data_gpu = gpuarray.to_gpu(np.float32(image))
    centers_gpu = gpuarray.zeros((n_centers, n_features + 3), np.float32)
    labels_gpu = gpuarray.zeros(dshape, np.uint32)

    __dirname__ = op.dirname(__file__)
    with open(op.join(__dirname__, "kernels", "slic3d_template.cu"), "r") as f:
        template = Template(f.read()).render(n_features=n_features)
        _mod_conv = SourceModule(template)
        gpu_slic_init = _mod_conv.get_function("init_clusters")
        gpu_slic_expectation = _mod_conv.get_function("expectation")
        gpu_slic_maximization = _mod_conv.get_function("maximization")

    vblock, vgrid = flat_kernel_config(int(np.prod(dshape)))
    cblock, cgrid = flat_kernel_config(int(np.prod(_sp_grid)))

    gpu_slic_init(
        data_gpu, centers_gpu, n_centers, sp_grid, sp_shape, block=cblock, grid=cgrid
    )
    cuda.Context.synchronize()

    for _ in range(max_iter):
        gpu_slic_expectation(
            data_gpu,
            centers_gpu,
            labels_gpu,
            m,
            S,
            n_centers,
            spacing,
            sp_grid,
            sp_shape,
            im_shape,
            block=vblock,
            grid=vgrid,
        )
        cuda.Context.synchronize()

        gpu_slic_maximization(
            data_gpu,
            labels_gpu,
            centers_gpu,
            n_centers,
            sp_grid,
            sp_shape,
            im_shape,
            block=cblock,
            grid=cgrid,
        )
        cuda.Context.synchronize()

    labels = np.asarray(labels_gpu.get(), dtype=np.int)
    if postprocess:
        segment_size = np.prod(dshape) / n_centers
        min_size = int(0.4 * segment_size)
        max_size = int(10 * segment_size)
        labels = _enforce_label_connectivity_cython(
            labels, min_size, max_size, start_label=0
        )

    return labels
