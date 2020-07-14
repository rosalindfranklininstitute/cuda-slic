

import pytest
import logging
import numpy as np

from survos2.io import dataset_from_uri
from survos2.data import embrain, mrbrain
from survos2.improc.utils import asnparray


@pytest.mark.parametrize("data_func", [
    mrbrain, embrain
])
def test_data(data_func):
    dataset = dataset_from_uri(data_func())
    data = asnparray(dataset)
    assert data.shape == data_func.__shape__
    assert data.dtype == np.float32
    assert data.max() == 1.0
    assert data.min() == 0.0
    dataset.close()


if __name__ == '__main__':
    pytest.main(args=['-s', __file__, '--loglevel', logging.DEBUG])
