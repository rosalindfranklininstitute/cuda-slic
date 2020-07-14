

import os
import logging
import pytest

from survos2.data import embrain
from survos2.improc.features import gaussian
from survos2.improc.regions.slic import slic3d


from skimage import data
from skimage import img_as_float32
from skimage import color

blob = data.binary_blobs(leng   th=512, n_dim=3, seed=2)
blob = img_as_float32(blob)
labels = slic3d(blob, sp_shape=(10, 10, 10), compactness=300)
print(labels.shape)
print(labels[:2, :2])
