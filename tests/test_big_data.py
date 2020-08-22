import pytest
import numpy as np

from cuda_slic.slic import slic3d

from skimage import data, color, filters


def test_slic3d_with_saved_results():
    n = 200
    blob = data.binary_blobs(length=n, n_dim=3, seed=2)
    blob = np.float32(blob)
    n_segments = n ** 3 // 5 ** 3  # 5x5x5 initial segment size
    labels = slic3d(blob, n_segments=n_segments, compactness=0.01, postprocess=True)
    expected = np.load("tests/test_big_data.npy")
    assert (labels == expected).all()


if __name__ == "__main__":
    test_slic3d_with_saved_results()
