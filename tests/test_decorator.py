

import pytest
import logging

import numpy as np
import dask.array as da

from survos2.improc.utils import survosify, map_blocks, optimal_chunksize


@survosify(dtype=np.int32)
def func_a(arr):
    return func_b(arr) + 1


@survosify(dtype=np.float32)
def func_b(arr):
    return func_c(arr) + 1


@survosify(dtype=np.uint8)
def func_c(arr):
    return arr + 1


def func_d(arr):
    return arr + 1


def test_wrapping(logger):
    data = np.random.randint(0, 10, size=(100, 946, 946))
    chunks = optimal_chunksize(data, 100)

    # Test full inception
    r1 = map_blocks(func_a, data, pad=False, chunk_size=chunks, timeit=True)
    np.testing.assert_allclose(r1, data + 3)
    assert r1.dtype == func_a.__out_dtype__

    # Test simple wrapping
    r1 = map_blocks(func_b, data, pad=False, chunk_size=chunks, timeit=True)
    r2 = map_blocks(func_c, data, pad=False, chunk_size=chunks, timeit=True)
    r3 = da.from_array(data, chunks=chunks).map_blocks(func_d).compute()

    np.testing.assert_allclose(r1, data + 2)
    np.testing.assert_allclose(r2, data + 1)
    np.testing.assert_allclose(r3, data + 1)

    assert r1.dtype == func_b.__out_dtype__
    assert r2.dtype == func_c.__out_dtype__
    assert r3.dtype == data.dtype


if __name__ == '__main__':
    pytest.main(args=['-s', __file__, '--loglevel', logging.DEBUG])