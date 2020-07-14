

import numpy as np


_MaskSize   = 4   # 4 bits per history label
_MaskCopy   = 15  # 0000 1111
_MaskPrev   = 240 # 1111 0000


def annotate_voxels(dataset, slice_idx=0, yy=None, xx=None, label=0):
    """

    """
    mbit = 2 ** (np.dtype(dataset.dtype).itemsize * 8 // _MaskSize) - 1

    def tobounds(slices):
        zs, ys, xs = slices
        if slice_idx < zs.start or slice_idx >= zs.stop:
            return False
        yp, xp = [], []
        for y, x in zip(yy, xx):
            if ys.start <= y < ys.stop and xs.start <= x < xs.stop:
                yp.append(y - ys.start)
                xp.append(x - xs.start)
        if len(yp) > 0:
            return slice_idx - zs.start, yp, xp
        return False

    if label >= 16 or label < 0 or type(label) != int:
        raise ValueError('Label has to be in bounds [0, 15]')

    modified = dataset.get_attr('modified')
    for i in range(dataset.total_chunks):
        idx = dataset.unravel_chunk_index(i)
        chunk_slices = dataset.global_chunk_bounds(idx)
        result = tobounds(chunk_slices)
        if result is False:
            modified[i] = (modified[i] << 1) & mbit
            continue
        idx, yp, xp = result
        data_chunk = dataset[chunk_slices]
        data_chunk = (data_chunk & _MaskCopy) | (data_chunk << _MaskSize)
        data_chunk[idx, yp, xp] = (data_chunk[idx, yp, xp] & _MaskPrev) | label
        dataset[chunk_slices] = data_chunk
        modified[i] = (modified[i] << 1) & mbit | 1

    dataset.set_attr('modified', modified)



def annotate_regions(dataset, region, r=None, label=0):
    if label >= 16 or label < 0 or type(label) != int:
        raise ValueError('Label has to be in bounds [0, 15]')
    if r is None or len(r) == 0:
        return

    mbit = 2 ** (np.dtype(dataset.dtype).itemsize * 8 // _MaskSize) - 1

    rmax = np.max(r)
    modified = dataset.get_attr('modified')
    for i in range(dataset.total_chunks):
        idx = dataset.unravel_chunk_index(i)
        chunk_slices = dataset.global_chunk_bounds(idx)
        reg_chunk = region[chunk_slices]
        total = max(rmax + 1, np.max(reg_chunk) + 1)
        mask = np.zeros(total, np.bool)
        mask[r] = True
        mask = mask[reg_chunk]
        if not np.any(mask):
            modified[i] = (modified[i] << 1) & mbit
            continue
        data_chunk = dataset[chunk_slices]
        data_chunk = (data_chunk & _MaskCopy) | (data_chunk << _MaskSize)
        data_chunk[mask] = (data_chunk[mask] & _MaskPrev) | label
        dataset[chunk_slices] = data_chunk
        modified[i] = (modified[i] << 1) & mbit | 1

    dataset.set_attr('modified', modified)


def undo_annotation(dataset):
    modified = dataset.get_attr('modified')
    for i in range(dataset.total_chunks):
        if modified[i] & 1 == 0:
            continue
        idx = dataset.unravel_chunk_index(i)
        chunk_slices = dataset.global_chunk_bounds(idx)
        data = dataset[chunk_slices]
        data = (data << _MaskSize) | (data >> _MaskSize)
        dataset[chunk_slices] = data
    dataset.set_attr('modified', modified)


def erase_label(dataset, label=0):
    if label >= 16 or label < 0 or type(label) != int:
        raise ValueError('Label has to be in bounds [0, 15]')
    lmask = _MaskCopy - label
    # remove label from all history
    nbit = np.dtype(dataset.dtype).itemsize * 8
    btop = 2**nbit - 1
    for i in range(dataset.total_chunks):
        idx = dataset.unravel_chunk_index(i)
        chunk_slices = dataset.global_chunk_bounds(idx)
        data_chunk = dataset[chunk_slices]
        modified = False
        for s in range(nbit // _MaskSize):
            shift = (s * _MaskSize)
            cmask = _MaskCopy << shift
            rmask = (data_chunk & cmask == label << shift) # Check presence of label
            if np.any(rmask): # Delete label
                modified = True
                hmask = btop - (label << shift)
                data_chunk[rmask] &= hmask
        if modified:
            dataset[chunk_slices] = data_chunk