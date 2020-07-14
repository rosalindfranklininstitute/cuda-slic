

import os
import logging
import pytest

from survos2.data import embrain
from survos2.improc.regions.slic import slic3d
from survos2.improc.features import gaussian
from survos2.improc.utils import map_blocks, map_pipeline
from survos2.utils import Timer, get_logger


gauss_params = dict(sigma=2)
slic_params = dict(sp_shape=(10, 10, 10), compactness=30)


@pytest.yield_fixture()
def data():
    data = embrain()
    yield data
    data.close()


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Travis doesn't support CUDA tests")
def test_compare_supervoxels(data, logger):
    data_gauss = map_blocks(gaussian, data, **gauss_params, timeit=True)
    reg1 = map_blocks(slic3d, data_gauss, **slic_params, timeit=True)

    pipeline = [(gaussian, gauss_params), (slic3d, slic_params)]
    reg2 = map_pipeline(data, pipeline=pipeline, timeit=True)

    with Timer('raw slic3d'):
        reg3 = slic3d(data_gauss, **slic_params)

    nreg1 = int(reg1.max())
    nreg2 = int(reg2.max())
    nreg3 = int(reg3.max())
    assert reg1.shape == reg2.shape
    assert reg1.dtype == reg2.dtype
    assert reg1.dtype == reg3.dtype
    assert reg1.shape == reg3.shape

    logger.info('Total regions: {} vs {} vs {}'.format(nreg1, nreg2, nreg3))


if __name__ == '__main__':
    pytest.main(args=['-s', __file__, '--loglevel', logging.DEBUG])
