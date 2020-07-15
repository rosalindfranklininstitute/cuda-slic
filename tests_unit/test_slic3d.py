import numpy as np

from survos2.improc.regions.slic import slic3d

from skimage import data, color, filters
from skimage import img_as_float32

def test_slic3d_grayscale_runs():
    blob = data.binary_blobs(length=50, n_dim=3, seed=2)
    blob = img_as_float32(blob)
    labels = slic3d(blob, nsp=100, compactness=300)
    assert isinstance(labels, np.ndarray)
    assert labels.ndim == 3

def test_slic3d_3channels_runs():
    blob = data.binary_blobs(length=50, n_dim=4, seed=2)
    blob = blob[:3]
    blob = filters.gaussian(blob)
    blob = img_as_float32(blob)
    labels = slic3d(blob, nsp=100, compactness=300)
    assert isinstance(labels, np.ndarray)
    assert labels.ndim == 3

def test_slic3d_grayscale_give_good_number_of_superpixels():
    blob = data.binary_blobs(length=50, n_dim=3, seed=2)
    blob = img_as_float32(blob)
    labels = slic3d(blob, nsp=100, compactness=300)
    assert len(np.unique(labels)) < 200

def test_slic3d_3channels_give_good_number_of_superpixels():
    blob = data.binary_blobs(length=100, n_dim=4, seed=2)
    blob = blob[:3]
    blob = filters.gaussian(blob)
    blob = img_as_float32(blob)
    labels = slic3d(blob, nsp=100, compactness=300)
    assert len(np.unique(labels)) < 500

# def test_slic3d_with_1GB_array():
#     blob = data.binary_blobs(length=600, n_dim=3, seed=2)
#     blob = img_as_float32(blob)
#     labels = slic3d(blob, nsp=100, compactness=300)
#     assert len(np.unique(labels)) < 500

# def test_slic3d_with_1GB_array_and_500_000_superpixels():
#     blob = data.binary_blobs(length=500, n_dim=3, seed=2)
#     blob = img_as_float32(blob)
#     labels = slic3d(blob, nsp=500000, compactness=300)
#     assert len(np.unique(labels)) < 1000000
