from cuda_slic.slic import flat_kernel_config


def test_flat_kernel_config_1():
    n_threads = 33
    block_size = 32
    block, grid = flat_kernel_config(n_threads, block_size=block_size)
    assert block == (32, 1, 1)
    assert grid == (2, 1, 1)


def test_flat_kernel_config_2():
    n_threads = 64
    block_size = 32
    block, grid = flat_kernel_config(n_threads, block_size=block_size)
    assert block == (32, 1, 1)
    assert grid == (2, 1, 1)
