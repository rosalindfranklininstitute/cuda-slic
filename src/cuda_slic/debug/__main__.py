import numpy as np
import pycuda

from skimage import color, data, filters

from ..slic import slic


def test_slic_grayscale_runs(n=200, sp_size=5):
    blob = data.binary_blobs(length=n, n_dim=3, seed=2)
    blob = np.float32(blob)
    n_segments = n ** 3 / sp_size ** 3
    labels = slic(blob, n_segments=n_segments, compactness=3)


if __name__ == "__main__":
    pycuda.driver.start_profiler()
    test_slic_grayscale_runs(n=200)
    pycuda.driver.stop_profiler()
