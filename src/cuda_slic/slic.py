import os.path as op

# import pycuda.autoinit
# import pycuda.driver as cuda
# import pycuda.gpuarray as gpuarray
import cupy as cp
import numpy as np

from jinja2 import Template

# from pycuda.compiler import SourceModule
from skimage.color import rgb2lab
from skimage.segmentation.slic_superpixels import (
    _enforce_label_connectivity_cython,
)

from .types import float3, int3


def flat_kernel_config(threads_total, block_size=128):
    block = (block_size, 1, 1)
    grid = ((threads_total + block_size - 1) // block_size, 1, 1)
    return block, grid


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
    dshape = np.array([depth, height, width])

    power = 1 / 2 if is_2d else 1 / 3
    sp_size = int(np.ceil((np.prod(dshape) / n_segments) ** power))
    _sp_shape = np.array(
        [
            # don't allow sp_shape to be larger than image sides
            min(depth, sp_size),
            min(height, sp_size),
            min(width, sp_size),
        ]
    )
    _sp_grid = (dshape + _sp_shape - 1) // _sp_shape

    sp_shape = np.asarray(tuple(_sp_shape[::-1]), np.int32)
    sp_grid = np.asarray(tuple(_sp_grid[::-1]), np.int32)

    m = np.float32(compactness)
    S = np.float32(np.max(_sp_shape))

    n_centers = np.int32(np.prod(_sp_grid))
    n_features = image.shape[-1]
    im_shape = np.asarray(tuple(dshape[::-1]), np.int32)
    spacing = np.asarray(tuple(spacing[::-1]), np.float32)

    image = np.float32(image)
    image *= 1 / m  # Do color scaling outside of kernel
    data_gpu = cp.asarray(image)
    centers_gpu = cp.zeros((n_centers, n_features + 3), dtype=cp.float32)
    labels_gpu = cp.zeros(dshape, dtype=cp.uint32)

    __dirname__ = op.dirname(__file__)
    with open(op.join(__dirname__, "kernels", "slic3d_template.cu"), "r") as f:
        template = Template(f.read()).render(
            n_features=n_features,
            n_clusters=n_centers,
            sp_shape=sp_shape,
            sp_grid=sp_grid,
            im_shape=im_shape,
            spacing=spacing,
            S=S,
        )
        template = 'extern "C" { ' + template + " }"
        _mod_conv = cp.RawModule(code=template, options=("-std=c++11",))
        gpu_slic_init = _mod_conv.get_function("init_clusters")
        gpu_slic_expectation = _mod_conv.get_function("expectation")
        gpu_slic_maximization = _mod_conv.get_function("maximization")

    vblock, vgrid = flat_kernel_config(int(np.prod(dshape)))
    cblock, cgrid = flat_kernel_config(int(np.prod(_sp_grid)))

    gpu_slic_init(
        cgrid,
        cblock,
        (
            data_gpu,
            centers_gpu,
        ),
    )
    cp.cuda.runtime.deviceSynchronize()

    for _ in range(max_iter):
        gpu_slic_expectation(
            vgrid,
            vblock,
            (
                data_gpu,
                centers_gpu,
                labels_gpu,
            ),
        )
        cp.cuda.runtime.deviceSynchronize()

        gpu_slic_maximization(
            cgrid,
            cblock,
            (
                data_gpu,
                labels_gpu,
                centers_gpu,
            ),
        )
        cp.cuda.runtime.deviceSynchronize()

    labels = np.asarray(labels_gpu.get(), dtype=np.intp)
    if enforce_connectivity:
        segment_size = np.prod(dshape) / n_centers
        min_size = int(min_size_factor * segment_size)
        max_size = int(max_size_factor * segment_size)
        labels = _enforce_label_connectivity_cython(
            labels, min_size, max_size, start_label=0
        )

    if is_2d:
        labels = labels[0]

    return labels
