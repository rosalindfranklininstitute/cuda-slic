

import os
import os.path as op
import tempfile
import requests
import wget
import tarfile
import glob

import numpy as np
import h5py as h5
from skimage import io

from functools import partial

from .io import dataset_from_uri


__all__ = ["embrain", ]


__dirname__ = op.dirname(__file__)
__datapath__ = op.realpath(op.join(__dirname__, '..', 'data'))


def data_shape(*shape):
    def wrapper(func):
        func.__shape__ = shape
        return func
    return wrapper


@data_shape(165, 768, 1024)
def embrain(dataset='train', force=False):
    """
    Electron Microscopy Dataset - EPFL

    The dataset available for download on this webpage* represents a 5x5x5µm
    section taken from the CA1 hippocampus region of the brain, corresponding
    to a 1065x2048x1536 volume. The resolution of each voxel is approximately
    5x5x5nm.

    * https://cvlab.epfl.ch/data/em

    Returns
    -------
    f : file object
        To close it when no longer using it.
    data : numpy array-like
        Dataset of shape `(165, 768, 1024)` of dtype `float32` normalized to
        `[0, 1]` range.
    """
    if dataset not in ['train', 'test']:
        raise ValueError('Invalid value for dataset: train or test expected.')

    source = op.join(__datapath__, 'eplf_embrain_{}.h5'.format(dataset))
    if op.isfile(source) and not force:
        return source

    dsurl = 'https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/'
    if dataset == 'train':
        dsurl += ' ElectronMicroscopy_Hippocampus/training.tif'
    else:
        dsurl += ' ElectronMicroscopy_Hippocampus/testing.tif'

    os.makedirs(__datapath__, exist_ok=True)
    print('Downloading {} - {}:'.format('EPFL EM Dataset', dataset))
    output = tempfile.mkdtemp()
    filename = wget.download(dsurl, out=output)
    data = io.imread(filename).astype(np.float32, copy=False)
    data -= data.min()
    data /= data.max()
    ds = dataset_from_uri(source, mode='w', dtype=np.float32, shape=data.shape)
    ds[...] = data
    ds.close()
    print()

    return source



@data_shape(99, 256, 256)
def mrbrain(force=False):
    """
    MR Dataset of a Head - Standford

    Description:    MR study of head with skull partially removed to reveal brain
    Dimensions:     99 slices of 256 x 256 pixels,
                    voxel grid is rectangular, and
                    X:Y:Z aspect ratio of each voxel is 1:1:2
    Data source:    acquired on a Siemens Magnetom and provided courtesy of
                    Siemens Medical Systems, Inc., Iselin, NJ.  Data edited
                    (skull removed) by Dr. Julian Rosenman, North Carolina
                    Memorial Hospital

    * https://graphics.stanford.edu/data/voldata/

    Returns
    -------
    f : file object
        To close it when no longer using it.
    data : numpy array-like
        Dataset of shape `(109, 256, 256)` of dtype `float32` normalized to
        `[0, 1]` range.
    """
    source = op.join(__datapath__, 'standford_mrbrain.h5')
    if op.isfile(source) and not force:
        return source

    dsurl = 'https://graphics.stanford.edu/data/voldata/mrbrain-16bit.tar.gz'
    print('Downloading {}:'.format('Standford MRBrain Dataset'))
    output = tempfile.mkdtemp()
    filename = wget.download(dsurl, out=output)
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(output)
    path = str(op.join(output, 'mrbrain-*.tif'))
    files = sorted(glob.glob(path))
    images = [io.imread(f) for f in files]
    data = np.stack(images)
    data = data.astype(np.float32, copy=False)
    data -= data.min()
    data /= data.max()
    ds = dataset_from_uri(source, mode='w', dtype=np.float32, shape=data.shape)
    ds[...] = data
    ds.close()
    print()

    return source

