

import hug

import re
import os
import os.path as op
import copy

import logging
import h5py as h5
import numpy as np
import mrcfile as mrc
import dask.array as da

from .model import Workspace, Dataset, DatasetWrapper
from .improc.utils import asnparray, optimal_chunksize


##############################################################################
# DTYPE and SHAPE utils
##############################################################################


def dtype2str(dtype):
    return np.dtype(dtype).name


def samedtype(d1, d2):
    return dtype2str(d1) == dtype2str(d2)


def sameshape(s1, s2):
    return np.all(np.asarray(s1, int) == np.asarray(s2, int))


def dtype2size(dtype):
    return np.dtype(dtype).itemsize


##############################################################################
# Retrieve datasets from URI
##############################################################################

MRC_REGEXP = r'^(mrc://)?(?P<fpath>.+(.rec|.mrc))$'
HDF5_REGEXP = r'^((hdf5|h5)://)?(?P<wspath>.+(.h5|.hdf5))(:(?P<dspath>[^:]+))?$'
SURVOS_REGEXP = r'^(survos://)?(((?P<session>[^:@]+)@)?(?P<workspace>[^:@]+):)?(?P<dataset>[^:@.]+)$'


def is_dataset_uri(uri):
    if not isinstance(uri, str):
        return False
    for (regexp, _) in __dataset_loaders__.values():
        if re.match(regexp, uri) is not None:
            return True
    return False


def supports_metadata(uri):
    return is_dataset_uri(uri) and dataset_from_uri(uri).supports_metadata()


def dataset_loader(uri):
    for (regexp, loader) in __dataset_loaders__.values():
        if re.match(regexp, uri) is not None:
            return loader
    return None


def dataset_from_uri(uri, mode='a', shape=None, dtype='float32', fill=0):
    """
    Loads datasets from uris.

    Parameters
    ----------
    uri: string
        A file URI for a dataset in MRC, HDF5 or SuRVoS format.
    mode: one of ['a', 'r', 'r+', 'w']
        The opening mode. See `open` for more details.
    shape: tuple
        Shape of the ND-Array. Only valid if `mode='w'`.
    dtype: numpy-accepted data type
        Data type of the ND-Array. Only valid if `mode='a'`. Default: float32.
    fill: int or float
        Value used to fill the array. Default: 0.

    Check `hdf5_from_uri`, `survos_from_uri` and `mrc_from_uri` for
    respective URI format specifications.
    """
    loader = dataset_loader(uri)
    if loader is not None:
        return loader(uri, mode=mode, shape=shape, dtype=dtype, fill=fill)
    else:
        raise ValueError('Worng URI. Only hdf5:// and survos:// uris are supported.')


def hdf5_from_uri(uri, mode='a', shape=None, dtype='float32', fill=0):
    """
    Loads an HDF5 dataset from an uri of the form:

        `[(hdf5|h5)://]file[:dataset]`

    Where `file` is the path to the `.h5` or `.hdf5` file and
    `dataset` is the internal path to the dataset within the HDF5
    structure. If `dataset` is not given `/data` is assumed. Protocol
    description is optional, being either `hdf5://` or `h5://` valid.

    Parameters
    ----------
    uri: string
        A file URI for a dataset in HDF5 format.
    mode: one of ['a', 'r', 'r+', 'w']
        The opening mode. See `open` for more details.
    shape: tuple
        Shape of the ND-Array. Only valid if `mode='a'`.
    dtype: numpy-accepted data type
        Data type of the ND-Array. Only valid if `mode='a'`. Default: float32.
    fill: int or float
        Value used to fill the array. Default: 0.

    """
    match = re.match(HDF5_REGEXP, uri)
    if match is None:
        raise ValueError('Invalid hdf5:// uri: `{}`'.format(uri))

    fpath = match.group('wspath')
    dpath = match.group('dspath') or '/data'

    if mode in ['r', 'r+', 'a']:
        f = h5.File(fpath, mode)
        ds = HDF5DatasetWrapper(f, f[dpath])
    else:
        try:
            f = h5.File(fpath, mode)
            d = f.create_dataset(dpath, shape=shape, dtype=dtype,
                                 fillvalue=fill)
            ds = HDF5DatasetWrapper(f, d)
        except OSError:
            raise ValueError('Output file cannot be created. It is likely that '
                             'the file is already open (e.g. the input and output '
                             'files for the filter are the same.')
        except Exception:
            if f:
                f.close()
            raise

    return ds


def survos_from_uri(uri, mode='a', shape=None, dtype='float32', fill=0):
    """
    Loads a SuRVoS dataset from an uri of the form:

        `survos://[session@workspace:]dataset`

    `dataset` can be a relative path if `workspace` is given
    or a full path to a SuRVoS Dataset on disk.

    `workspace` can be the name of the workspace if `config.chroot` is enabled
    or the path to an existing folder on disk otherwise.

    `session` is the user session on the workspace. Is set to 'default' if missing
    and `workspace` is given.

    Parameters
    ----------
    uri: string
        A file URI for a dataset in SuRVoS format.
    mode: one of ['a', 'r', 'r+', 'w']
        The opening mode. See `open` for more details.
    shape: tuple
        Shape of the ND-Array. Only valid if `mode='a'`.
    dtype: numpy-accepted data type
        Data type of the ND-Array. Only valid if `mode='a'`. Default: float32.
    fill: int or float
        Value used to fill the array. Default: 0.
    """
    match = re.match(SURVOS_REGEXP, uri)
    if not match:
        raise ValueError('Invalid survos:// uri: {}'.format(uri))
    dataset = match.group('dataset')
    workspace = match.group('workspace')
    session = match.group('session') or 'default'

    if workspace:
        ws = Workspace(workspace)
        if dataset == Workspace.__dsname__:
            if mode != 'r':
                raise ValueError('Invalid URI. Dataset {} cannot be oppened in write mode.'
                                 .format(Workspace.__dsname__))
            ds = ws.get_data(readonly=True)
        elif mode == 'r':
            ds = ws.get_dataset(dataset, session=session, readonly=True)
        else:
            if ws.has_dataset(dataset, session=session):
                ds = ws.get_dataset(dataset, session=session)
            elif mode == 'w':
                ds = ws.add_dataset(dataset, dtype, session=session, fillvalue=fill)
            else:
                raise ValueError('Workspace has no dataset: {}'.format(uri))
    else:
        if mode != 'r':
            if Dataset.exists(dataset):
                ds = Dataset(dataset)
            elif mode == 'w':
                ds = Dataset.create(path, shape=shape, dtype=dtype,
                                    fillvalue=fillvalue)
            else:
                raise ValueError('Dataset does not exist: {}'.format(uri))
        else:
            ds = Dataset(dataset, readonly=True)

    return ds



def mrc_from_uri(uri, mode='a', shape=None, dtype='float32', fill=0):
    """
    Loads a MRC dataset from an uri of the form:

        `[mrc://]file`

    Where `file` is the path to the `.mrc` or `.rec` file.

    Parameters
    ----------
    uri: string
        A file URI for a dataset in MRC format.
    mode: one of ['a', 'r', 'r+', 'w']
        The opening mode. See `open` for more details.
    shape: tuple
        Shape of the ND-Array. Only valid if `mode='a'`.
    dtype: numpy-accepted data type
        Data type of the ND-Array. Only valid if `mode='a'`. Default: float32.
    fill: int or float
        Value used to fill the array. Default: 0.
    """
    match = re.match(MRC_REGEXP, uri)
    if match is None:
        raise ValueError('Invalid mrc:// uri: `{}`'.format(uri))
    fpath = match.group('fpath')
    if mode != 'r':
        if mode in ['a', 'r+'] and  op.isfile(fpath):
            f = mrc.mmap(fpath, 'r+')
            if not sameshape(f.data.shape, shape) or not samedtype(f.data.dtype, dtype):
                msg = 'Dataset already exists with different' \
                      ' shape or dype. Expected {} and {}, got {} and {}' \
                      .format(shape, dtype, f.data.shape, f.data.dtype)
                raise ValueError(msg)
        else:
            f = mrc.MrcMemmap(fpath, mode='w+', overwrite=True)
            f._open_memmap(dtype, shape)
            f.update_header_from_data()
            f.data[...] = fill
    else:
        f = mrc.mmap(fpath, 'r')

    return DatasetWrapper(f, f.data)



class HDF5DatasetWrapper(DatasetWrapper):

    def __init__(self, fileobj, dataset, strlength=300):
        super().__init__(fileobj, dataset)
        self.strdtype = 'S' + str(int(strlength))

    def supports_metadata(self):
        return True

    def has_attr(self, key):
        return key in self.dataset.attrs

    def get_attr(self, key, default=None):
        if self.has_attr(key):
            return self.attrs[key]
        elif default is not None:
            return default
        raise KeyError('Dataset has no `{}` metadata.'.format(key))

    def set_attr(self, key, value):
        if type(value) == str:
            value = np.asarray(value, dtype=self.strdtype)
        self.dataset.attrs[key] = value

    def metadata(self):
        return copy.deepcopy(dict(self.dataset.attrs))


__dataset_loaders__ = dict(
    mrc=(MRC_REGEXP, mrc_from_uri),
    hdf5=(HDF5_REGEXP, hdf5_from_uri),
    survos=(SURVOS_REGEXP, survos_from_uri)
)