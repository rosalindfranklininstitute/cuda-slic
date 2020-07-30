from ..slic import slic3d

from skimage import data, color, filters
from skimage import img_as_float32

def test_slic3d_grayscale_runs():
    blob = data.binary_blobs(length=5, n_dim=3, seed=2)
    blob = img_as_float32(blob)
    labels = slic3d(blob, n_segments=10, compactness=3)

if __name__ == "__main__":
    test_slic3d_grayscale_runs()
