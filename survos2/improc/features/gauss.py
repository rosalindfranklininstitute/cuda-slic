

import numpy as np
import logging

from pycuda import cumath, gpuarray

from ..cuda import asgpuarray
from ..utils import gpufeature

from .conv import conv_sep, make_gaussian_1d


@gpufeature
def gaussian(data, sigma=0.5, **kwargs):
    """
    Computes Gaussian filter/convolution in the GPU.
    Accepts 2 and 3 dimesional input.

    Parameters
    ----------
    data : 2 or 3 dimensional array
        The data to be filtered
    sigma : float or array of floats
        The standard deviation of the Gaussian filter. Controls the
        radius and strength of the filter. If an array is given, it has to
        satisfy `len(sigma) = data.ndim`. Default: 0.5
    **kwargs : other named parameters
        Parameters are passed to `conv.make_gaussian_1d`

    Returns
    -------
    result : 2 or 3 dimensional filtered `GPUArray`
        The result of the filtering resulting from PyCuda. Use `.get()` to
        retrieve the corresponding Numpy array.
    """
    if data.ndim == 2:
        result = _gaussian2d(data, sigma=sigma, **kwargs)
    elif data.ndim == 3:
        result = _gaussian3d(data, sigma=sigma, **kwargs)
    else:
        raise ValueError("Input dataset has to be 2 or 3 dimensional: {}"
                         .format(data.ndim))
    return result


def _gaussian2d(data, sigma=0.5, size=None, **kwargs):
    """
    Computes Gaussian filter/convolution in the GPU.

    Parameters
    ----------
    data : 2 dimensional array
        The data to be filtered
    sigma : float or array of floats
        The standard deviation of the Gaussian filter. Controls the
        radius and strength of the filter. If an array is given, it has to
        satisfy `len(sigma) = data.ndim = 2`. Default: 0.5
    **kwargs : other named parameters
        Parameters are passed to `conv.make_gaussian_1d`

    Returns
    -------
    result : 2 dimensional filtered `GPUArray`
        The result of the filtering resulting from PyCuda. Use `.get()` to
        retrieve the corresponding Numpy array.
    """
    assert data.ndim == 2

    if np.isscalar(sigma):
        ky = kx = make_gaussian_1d(sigma, size=size, **kwargs)
    elif len(sigma) == data.ndim:
        kz = make_gaussian_1d(sigma[0], size=size, **kwargs)
        ky = make_gaussian_1d(sigma[1], size=size, **kwargs)
    else:
        raise ValueError('Incorrect parameter `sigma`: a scalar or '
                         '`(sy, sx)` vector is expected.')
    return conv_sep(data, [ky, kx])


def _gaussian3d(data, sigma=0.5, **kwargs):
    """
    Computes Gaussian filter/convolution in the GPU.

    Parameters
    ----------
    data : 3 dimensional array
        The data to be filtered
    sigma : float or array of floats
        The standard deviation of the Gaussian filter. Controls the
        radius and strength of the filter. If an array is given, it has to
        satisfy `len(sigma) = data.ndim = 3`. Default: 0.5
    **kwargs : other named parameters
        Parameters are passed to `conv.make_gaussian_1d`

    Returns
    -------
    result : 3 dimensional filtered `GPUArray`
        The result of the filtering resulting from PyCuda. Use `.get()` to
        retrieve the corresponding Numpy array.
    """
    assert data.ndim == 3

    if np.isscalar(sigma):
        kz = ky = kx = make_gaussian_1d(sigma, **kwargs)
    elif len(sigma) == data.ndim:
        kz = make_gaussian_1d(sigma[0], **kwargs)
        ky = make_gaussian_1d(sigma[1], **kwargs)
        kx = make_gaussian_1d(sigma[2], **kwargs)
    else:
        raise ValueError('Incorrect parameter `sigma`: a scalar or '
                         '`(sz, sy, sx)` vector is expected.')
    return conv_sep(data, [kz, ky, kx])


@gpufeature
def gaussian_center(data, sigma=0.5, **kwargs):
    """
    Performs Gaussian centering, where the mean in a Gaussian neighbourhood is
    substracted to every voxel.

    Parameters
    ----------
    data : 2 or 3 dimensional array
        The data to be filtered
    sigma : float or array of floats
        The standard deviation of the Gaussian filter used to calculate the
        mean. Controls the radius and strength of the filter.
        If an array is given, it has to satisfy `len(sigma) = data.ndim`.
        Default: 0.5
    **kwargs : other named parameters
        Parameters are passed to `conv.make_gaussian_1d`

    Returns
    -------
    result : 2 or 3 dimensional filtered `GPUArray`
        The result of the filtering resulting from PyCuda. Use `.get()` to
        retrieve the corresponding Numpy array.
    """
    kwargs['keep_gpu'] = True
    data = asgpuarray(data)
    result = data - gaussian(data, sigma=sigma, **kwargs)
    return result


@gpufeature
def gaussian_norm(data, sigma=0.5, **kwargs):
    """
    Performs Gaussian normalization to an input dataset. This is, every voxel
    is normalized by substracting the mean and dividing it by the standard
    deviation in a Gaussian neighbourhood around it.

    Parameters
    ----------
    data : 2 or 3 dimensional array
        The data to be filtered
    sigma : float or array of floats
        The standard deviation of the Gaussian filter used to estimate the mean
        and standard deviation of the kernel. Controls the radius and strength
        of the filter. If an array is given, it has to satisfy
        `len(sigma) = data.ndim`. Default: 0.5
    **kwargs : other named parameters
        Parameters are passed to `conv.make_gaussian_1d`

    Returns
    -------
    result : 2 or 3 dimensional filtered `GPUArray`
        The result of the filtering resulting from PyCuda. Use `.get()` to
        retrieve the corresponding Numpy array.
    """
    kwargs['keep_gpu'] = True
    num = gaussian_center(data, sigma=sigma, **kwargs)
    den = cumath.sqrt(gaussian(num**2, sigma=sigma, **kwargs))
    # TODO numerical precision ignore den < 1e-7
    num /= den
    return num
