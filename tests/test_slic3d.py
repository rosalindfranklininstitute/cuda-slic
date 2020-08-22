import pytest
import numpy as np

from cuda_slic.slic import slic3d

from skimage import data, color, filters
from skimage import img_as_float32


def test_slic3d_grayscale_runs():
    blob = data.binary_blobs(length=50, n_dim=3, seed=2)
    blob = img_as_float32(blob)
    labels = slic3d(blob, n_segments=100, compactness=3)
    assert isinstance(labels, np.ndarray)
    assert labels.ndim == 3


def test_slic3d_3channels_runs():
    blob = data.binary_blobs(length=50, n_dim=4, seed=2)
    blob = blob[:3]
    blob = filters.gaussian(blob)
    blob = img_as_float32(blob)
    labels = slic3d(blob, n_segments=100, compactness=3)
    assert isinstance(labels, np.ndarray)
    assert labels.ndim == 3


def test_slic3d_grayscale_give_good_number_of_superpixels():
    blob = data.binary_blobs(length=50, n_dim=3, seed=2)
    blob = img_as_float32(blob)
    labels = slic3d(blob, n_segments=100, compactness=3)
    assert len(np.unique(labels)) < 200


def test_slic3d_3channels_give_good_number_of_superpixels():
    blob = data.binary_blobs(length=33, n_dim=4, seed=2)
    blob = blob[:3]
    blob = filters.gaussian(blob)
    blob = img_as_float32(blob)
    labels = slic3d(blob, n_segments=100, compactness=3)
    assert len(np.unique(labels)) < 500


def test_slic3d_with_sp_shape_tuple():
    blob = data.binary_blobs(length=50, n_dim=3, seed=2)
    blob = img_as_float32(blob)
    labels = slic3d(blob, sp_shape=(5, 5, 5), compactness=3)
    upper_bound = 50 ** 3 / 5 ** 3 + 500
    assert len(np.unique(labels)) < upper_bound


def test_slic3d_with_sp_shape_int():
    blob = data.binary_blobs(length=50, n_dim=3, seed=2)
    blob = img_as_float32(blob)
    labels = slic3d(blob, sp_shape=5, compactness=3)
    upper_bound = 50 ** 3 / 5 ** 3 + 500
    assert len(np.unique(labels)) < upper_bound


def test_slic3d_raises_value_error_when_input_dimention_less_than_3():
    blob = data.binary_blobs(length=33, n_dim=2, seed=2)
    blob = np.float32(blob)
    with pytest.raises(ValueError):
        labels = slic3d(blob, n_segments=100, compactness=3)


def test_slic3d_raises_value_error_when_input_dimention_more_than_4():

    blob = data.binary_blobs(length=33, n_dim=5, seed=2)
    blob = np.float32(blob)

    with pytest.raises(ValueError):
        labels = slic3d(blob, n_segments=100, compactness=3)


def test_cherry_picked_results_with_postprocess():
    # fmt: off
    plane = [[1,1,0,0],
             [1,1,0,0],
             [1,1,1,1],
             [1,1,1,1]]

    vol = np.asarray([plane, plane, plane])
    expected = np.array([[[0, 0, 1, 1],
                          [0, 0, 1, 1],
                          [0, 0, 2, 2],
                          [0, 0, 2, 2]],

                          [[0, 0, 1, 1],
                           [0, 0, 1, 1],
                           [0, 0, 2, 2],
                           [0, 0, 2, 2]],

                          [[0, 0, 1, 1],
                           [0, 0, 1, 1],
                           [0, 0, 2, 2],
                           [0, 0, 2, 2]]])
    # fmt: on
    labels = slic3d(vol, n_segments=2, compactness=0.1, postprocess=True)
    assert (labels == expected).all()


def test_cherry_picked_results_without_postprocess():
    # fmt: off
    plane = [[1,1,0,0],
             [1,1,0,0],
             [1,1,1,1],
             [1,1,1,1]]

    vol = np.asarray([plane, plane, plane])
    expected = np.array([[[3, 3, 2, 2],
                          [3, 3, 2, 2],
                          [3, 3, 4, 4],
                          [3, 3, 4, 4]],

                          [[3, 3, 2, 2],
                           [3, 3, 2, 2],
                           [3, 3, 4, 4],
                           [3, 3, 4, 4]],

                          [[3, 3, 2, 2],
                           [3, 3, 2, 2],
                           [3, 3, 4, 4],
                           [3, 3, 4, 4]]])
    # fmt: on
    labels = slic3d(vol, n_segments=2, compactness=0.1, postprocess=False)
    assert (labels == expected).all()


def test_stress_test_slic3d():
    lengths = range(100, 200, 3)
    for i in range(len(lengths)):
        n_segments = lengths[i] ** 3 / 10 ** 3
        blob = data.binary_blobs(length=lengths[i], n_dim=3, seed=2)
        blob = np.float32(blob)
        slic3d(blob, n_segments=n_segments, compactness=2)
