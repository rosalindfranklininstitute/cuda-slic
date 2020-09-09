import os

import numpy as np

from jinja2 import Template
from skimage.color import rgb2lab
from skimage.segmentation.slic_superpixels import (
    _enforce_label_connectivity_cython,
)

from .types import float3, int3


def line_kernel_config(threads_total, block_size=128):
    block = (block_size, 1, 1)
    grid = ((threads_total + block_size - 1) // block_size, 1, 1)
    return block, grid


def box_kernel_config(im_shape, block=(2, 4, 32)):
    """
    block = (z=2,y=4,x=32) was hand tested to be very fast
    on the Quadro P2000, might not be the fastest config for other
    cards
    """
    grid = (
        (im_shape[0] + block[0] - 1) // block[0],
        (im_shape[1] + block[1] - 1) // block[1],
        (im_shape[2] + block[2] - 1) // block[2],
    )
    return block, grid


def _slic_pycuda(
    image,
    n_features,
    n_centers,
    max_iter,
    template,
    center_block,
    center_grid,
    image_block,
    image_grid,
):

    import pycuda.autoinit
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray

    from pycuda.compiler import SourceModule

    module = SourceModule(
        template,
        options=[
            "-std=c++11",
        ],
    )
    gpu_slic_init = module.get_function("init_clusters")
    gpu_slic_expectation = module.get_function("expectation")
    gpu_slic_maximization = module.get_function("maximization")

    data_gpu = gpuarray.to_gpu(image)
    centers_gpu = gpuarray.zeros((n_centers, n_features + 3), np.float32)
    labels_gpu = gpuarray.zeros(image.shape[:3], np.uint32)

    gpu_slic_init(
        data_gpu,
        centers_gpu,
        block=center_block,
        grid=center_grid,
    )
    cuda.Context.synchronize()

    for _ in range(max_iter):
        gpu_slic_expectation(
            data_gpu,
            centers_gpu,
            labels_gpu,
            block=image_block,
            grid=image_grid,
        )
        cuda.Context.synchronize()

        gpu_slic_maximization(
            data_gpu,
            labels_gpu,
            centers_gpu,
            block=center_block,
            grid=center_grid,
        )
        cuda.Context.synchronize()

    labels = np.asarray(labels_gpu.get(), dtype=np.intp)
    return labels


def _slic_cupy(
    image,
    n_features,
    n_centers,
    max_iter,
    template,
    center_block,
    center_grid,
    image_block,
    image_grid,
):

    import cupy as cp

    template = 'extern "C" { ' + template + " }"
    module = cp.RawModule(code=template, options=("-std=c++11",))
    gpu_slic_init = module.get_function("init_clusters")
    gpu_slic_expectation = module.get_function("expectation")
    gpu_slic_maximization = module.get_function("maximization")

    data_gpu = cp.asarray(image)
    centers_gpu = cp.zeros((n_centers, n_features + 3), dtype=cp.float32)
    labels_gpu = cp.zeros(image.shape[:3], dtype=cp.uint32)

    gpu_slic_init(
        center_grid,
        center_block,
        (
            data_gpu,
            centers_gpu,
        ),
    )
    cp.cuda.runtime.deviceSynchronize()

    for _ in range(max_iter):
        gpu_slic_expectation(
            image_grid,
            image_block,
            (
                data_gpu,
                centers_gpu,
                labels_gpu,
            ),
        )
        cp.cuda.runtime.deviceSynchronize()

        gpu_slic_maximization(
            center_grid,
            center_block,
            (
                data_gpu,
                labels_gpu,
                centers_gpu,
            ),
        )
        cp.cuda.runtime.deviceSynchronize()

    labels = np.asarray(labels_gpu.get(), dtype=np.intp)
    return labels


def _slic(image, sp_shape, sp_grid, spacing, compactness, max_iter):
    im_shape_zyx = image.shape[:3]

    im_shape_xyz = im_shape_zyx[::-1]
    sp_grid_xyz = sp_grid[::-1]
    sp_shape_xyz = sp_shape[::-1]
    spacing_xyz = spacing[::-1]

    spacial_weight = float(np.max(sp_shape_xyz))
    n_centers = int(np.prod(sp_grid_xyz))
    n_features = image.shape[-1]

    image = np.float32(image)
    image *= 1 / compactness  # do color scaling once outside of kernel

    __dirname__ = os.path.dirname(__file__)
    module_path = os.path.join(__dirname__, "kernels", "slic3d_template.cu")
    with open(module_path, "r") as f:
        template = Template(f.read()).render(
            n_features=n_features,
            n_clusters=n_centers,
            sp_shape=sp_shape_xyz,
            sp_grid=sp_grid_xyz,
            im_shape=im_shape_xyz,
            spacing=spacing_xyz,
            SS=spacial_weight * spacial_weight,
        )

    center_block, center_grid = line_kernel_config(int(np.prod(sp_grid)))
    image_block, image_grid = box_kernel_config(im_shape_zyx)

    try:
        import pycuda

        labels = _slic_pycuda(
            image,
            n_features,
            n_centers,
            max_iter,
            template,
            center_block,
            center_grid,
            image_block,
            image_grid,
        )
    except ImportError:
        labels = _slic_cupy(
            image,
            n_features,
            n_centers,
            max_iter,
            template,
            center_block,
            center_grid,
            image_block,
            image_grid,
        )
    except:
        raise

    return labels


def slic(
    image,
    n_segments=100,
    compactness=1.0,
    spacing=(1, 1, 1),
    max_iter=5,
    multichannel=True,
    convert2lab=None,
    enforce_connectivity=True,
    min_size_factor=0.4,
    max_size_factor=10.0,
):
    """Segments image using k-means clustering in Color-(x,y,z) space.
    Parameters
    ----------
    image : 2D, 3D or 4D ndarray
        Input image, which can be 2D or 3D, and grayscale or multichannel
        (see `multichannel` parameter).
    n_segments : int, optional
        The (approximate) number of labels in the segmented output image.
    compactness : float, optional
        Balances color proximity and space proximity. Higher values give
        more weight to space proximity, making superpixel shapes more
        square/cubic.
        This parameter depends strongly on image contrast and on the
        shapes of objects in the image. We recommend exploring possible
        values on a log scale, e.g., 0.01, 0.1, 1, 10, 100, before
        refining around a chosen value.
    max_iter : int, optional
        Maximum number of iterations of k-means.
    spacing : (3,) array-like of floats, optional
        The voxel spacing along each image dimension. By default, `slic`
        assumes uniform spacing (same voxel resolution along z, y and x).
        This parameter controls the weights of the distances along z, y,
        and x during k-means clustering.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.
    convert2lab : bool, optional
        Whether the input should be converted to Lab colorspace prior to
        segmentation. The input image *must* be RGB. Highly recommended.
        This option defaults to ``True`` when ``multichannel=True`` *and*
        ``image.shape[-1] == 3``.
    enforce_connectivity : bool, optional
        Whether the generated segments are connected or not
    min_size_factor : float, optional
        Proportion of the minimum segment size to be removed with respect
        to the supposed segment size ```depth*width*height/n_segments```
    max_size_factor : float, optional
        Proportion of the maximum connected segment size. A value of 3 works
        in most of the cases.
    Returns
    -------
    labels : 2D or 3D array
        Integer mask indicating segment labels.
    Raises
    ------
    ValueError
        If ``convert2lab`` is set to ``True`` but the last array
        dimension is not of length 3.
    ValueError
        If ``image.ndim`` is not 2, 3 or 4.
    Notes
    -----
    * Images of shape (M, N, 3) are interpreted as 2D RGB images by default. To
      interpret them as 3D with the last dimension having length 3, use
      `multichannel=False`.

    References
    ----------
    .. [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,
        Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to
        State-of-the-art Superpixel Methods, TPAMI, May 2012.
        :DOI:`10.1109/TPAMI.2012.120`
    Examples
    --------
    >>> from cuda_slic import slic
    >>> from skimage import data
    >>> img = data.astronaut() # 2D RGB image
    >>> labels = slic(img, n_segments=100, compactness=10)
    To segment single channel 3D volumes
    >>> vol = data.binary_blobs(length=50, n_dim=3, seed=2)
    >>> labels = slic(vol, n_segments=100, multichannel=False, compactness=0.1)
    """

    if image.ndim not in [2, 3, 4]:
        raise ValueError(
            (
                "input image must be either 2, 3, or 4 dimentional.\n"
                "The input image.ndim is {}".format(image.ndim)
            )
        )

    is_2d = False

    if image.ndim == 2:
        # 2D grayscale image
        image = image[np.newaxis, ..., np.newaxis]
        is_2d = True
    elif image.ndim == 3 and multichannel:
        # Make 2D multichannel image 3D with depth = 1
        image = image[np.newaxis, ...]
        is_2d = True
    elif image.ndim == 3 and not multichannel:
        # Add channel as single last dimension
        image = image[..., np.newaxis]

    if multichannel and (convert2lab or convert2lab is None):
        if image.shape[-1] != 3 and convert2lab:
            raise ValueError("Lab colorspace conversion requires a RGB image.")
        elif image.shape[-1] == 3:
            image = rgb2lab(image)

    depth, height, width = image.shape[:3]
    im_shape_zyx = np.array([depth, height, width])

    power = 1 / 2 if is_2d else 1 / 3
    sp_size = int(np.ceil((np.prod(im_shape_zyx) / n_segments) ** power))
    sp_shape = np.array(
        [
            # don't allow sp_shape to be larger than image sides
            min(depth, sp_size),
            min(height, sp_size),
            min(width, sp_size),
        ]
    )
    sp_grid = (im_shape_zyx + sp_shape - 1) // sp_shape
    n_centers = np.prod(sp_grid)

    labels = _slic(image, sp_shape, sp_grid, spacing, compactness, max_iter)

    if enforce_connectivity:
        segment_size = np.prod(im_shape_zyx) / n_centers
        min_size = int(min_size_factor * segment_size)
        max_size = int(max_size_factor * segment_size)
        labels = _enforce_label_connectivity_cython(
            labels, min_size, max_size, start_label=0
        )

    if is_2d:
        labels = labels[0]

    return labels
