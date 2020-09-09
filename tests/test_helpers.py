from cuda_slic.slic import box_kernel_config, line_kernel_config


def test_line_kernel_config_when_not_tiles():
    n_threads = 33
    block_size = 32
    block, grid = line_kernel_config(n_threads, block_size=block_size)
    assert block == (32, 1, 1)
    assert grid == (2, 1, 1)


def test_line_kernel_config_when_tiles():
    n_threads = 64
    block_size = 32
    block, grid = line_kernel_config(n_threads, block_size=block_size)
    assert block == (32, 1, 1)
    assert grid == (2, 1, 1)


def test_box_kernel_config_when_not_tiles():
    im_shape = (33, 33, 33)
    block_shape = (32, 32, 32)
    block, grid = box_kernel_config(im_shape, block=block_shape)
    assert block == (32, 32, 32)
    assert grid == (2, 2, 2)


def test_box_kernel_config_when_tiles():
    im_shape = (64, 64, 64)
    block_shape = (32, 32, 32)
    block, grid = box_kernel_config(im_shape, block=block_shape)
    assert block == (32, 32, 32)
    assert grid == (2, 2, 2)
