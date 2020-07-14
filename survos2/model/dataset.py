

import os
import os.path as op
import shutil

import copy

import h5py as h5
import numpy as np
import dask.array as da

import logging as log

import collections
import itertools
import numbers

from survos2.config import Config
from survos2.utils import AttributeDB, get_logger
from survos2.improc.utils import optimal_chunksize


logger = get_logger()


CHUNKS = Config['computing.chunk_size'] if Config['computing.chunks'] else None
CHUNKS_SPARSE = Config['computing.chunk_size_sparse'] if Config['computing.chunks'] else None


class DatasetException(Exception):
    pass


class BaseDataset(object):

    def close(self):
        pass

    def supports_metadata(self):
        return False

    def has_attr(self, key, default=None):
        raise NotImplementedError()

    def get_attr(self, key, value):
        raise NotImplementedError()

    def set_attr(self, key, value):
        raise NotImplementedError()

    def metadata(self):
        raise NotImplementedError()


class DatasetWrapper(BaseDataset):

    def __init__(self, fileobj, dataset):
        self.fileobj = fileobj
        self.dataset = dataset

    @property
    def id(self):
        for prop in ['id', 'name', 'path']:
            if hasattr(self.dataset, prop):
                return getattr(self.dataset, prop)
        return 'dataset'

    def close(self):
        if self.fileobj:
            self.fileobj.close()

    def __getattr__(self, attr):
        """
        Attributes/Functions that do not exist in the extended class
        are going to be passed to the instance being wrapped
        """
        return self.dataset.__getattribute__(attr)

    def __getitem__(self, slices):
        return self.dataset.__getitem__(slices)

    def __setitem__(self, slices, values):
        return self.dataset.__setitem__(slices, values)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def tojson(self):
        if hasattr(self.dataset, 'tojson'):
            return self.dataset.tojson()
        return dict(shape=self.shape, dtype=np.dtype(self.dtype).name)



class Dataset(BaseDataset):

    __dbname__ = 'dataset'
    __dsname__ = '__data__'

    def __init__(self, path, readonly=False):
        if not op.isdir(path):
            raise DatasetException('Dataset \'%s\' does not exist.' % path)
        self._load(path)
        self._readonly = readonly

    @property
    def id(self):
        return self._id

    def _load(self, path):
        self._id = op.basename(path)
        self._path = path
        dbpath = op.join(path, self.__dbname__)
        if op.isfile(dbpath + '.yaml'):
            self._db = db = AttributeDB(dbpath, dbtype='yaml')
        elif op.isfile(dbpath + '.json'):
            self._db = db = AttributeDB(dbpath, dbtype='json')
        else:
            raise DatasetException('DB not found: \'%s\' is not a valid dataset.' % path)

        try:
            self._shape = tuple(db[self.__dsname__]['shape'])
            self._dtype = db[self.__dsname__]['dtype']
            self._chunk_grid = tuple(db[self.__dsname__]['chunk_grid'])
            self._chunk_size = tuple(db[self.__dsname__]['chunk_size'])
            self._fillvalue = db[self.__dsname__]['fillvalue']
        except:
            raise DatasetException('Unable to load dataset attributes: \'%s\'' % path)
        self._total_chunks = np.prod(self._chunk_grid)
        self._ndim = len(self._shape)

        if not (len(self.shape) == len(self.chunk_grid) == len(self.chunk_size)):
            raise DatasetException('Data shape and chunk layout do not match: {}, {}, {}'
                                   .format(self.shape, self.chunk_grid, self.chunk_size))

    def tojson(self):
        db = copy.deepcopy(self._db)
        metadata = db.pop(self.__dsname__)
        metadata['id'] = self._id
        metadata['path'] = self._path
        #metadata['metadata'] = db
        metadata.setdefault('name', self._id)
        return metadata

    # Properties

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def chunk_grid(self):
        return self._chunk_grid

    @property
    def chunk_size(self):
        return self._chunk_size

    @property
    def fillvalue(self):
        return self._fillvalue

    @property
    def total_chunks(self):
        return self._total_chunks

    @property
    def ndim(self):
        return self._ndim

    @property
    def readonly(self):
        return self._readonly

    # Access / Edit metadata

    def supports_metadata(self):
        return True

    def metadata(self):
        return self.get_metadata()

    def has_attr(self, key):
        return self.get_metadata(key) is not None

    def get_attr(self, key, default=None):
        value = self.get_metadata(key, default=default)
        if value is None:
            raise KeyError('Dataset has no `{}` metadata.'.format(key))
        return value

    def set_attr(self, key, value):
        self.set_metadata(key, value)

    def get_metadata(self, key=None, default=None):
        if key is None:
            return copy.deepcopy(self._db)
        elif key in self._db:
            return self._db[key]
        return default

    def set_metadata(self, key, value):
        if key == self.__dsname__:
            raise DatasetException('Dataset metadata cannot me changed.')
        elif not self._db.isserializable(value):
            raise DatasetException('Metadata `{}` is not serializable'.format(value))
        self._db[key] = value
        self._db.save()

    def update_metadata(self, key, value):
        if key == self.__dsname__:
            raise DatasetException('Dataset metadata cannot me changed.')
        elif not self._db.isserializable(value):
            raise DatasetException('Metadata `{}` is not serializable'.format(value))
        elif key in self._db:
            self._db.update(value)
            self._db.save()
        else:
            raise DatasetException('Metadata \'%s\' does not exist.' % key)

    # Create

    @staticmethod
    def create(path, shape=None, dtype=None, data=None, fillvalue=0, chunks=CHUNKS, **kwargs):
        database = kwargs.pop('database', 'yaml')
        readonly = kwargs.pop('readonly', False)

        if Dataset.exists(path):
            raise DatasetException('Dataset \'%s\' already exists.' % path)
        if op.isfile(path):
            raise DatasetException('Path \'%s\' is not a valid directory path.' % path)
        elif op.isdir(path) and os.listdir(dir):  # non-empty dir
            raise DatasetException('Path \'%s\' already exists.' % path)

        if data is not None:
            shape = data.shape
            dtype = data.dtype

        if shape is None or dtype is None:
            raise DatasetException('Not valid `shape` and `dtype` was provided.')

        shape = list(shape)
        dtype = np.dtype(dtype).name
        isize = np.dtype(dtype).itemsize

        if chunks is None:
            chunk_size = list(shape)
        elif isinstance(chunks, collections.Iterable) and len(chunks) == len(shape):
            chunk_size = list(chunks)
        elif isinstance(chunks, numbers.Number):
            chunk_size = list(optimal_chunksize(shape, chunks, item_size=isize, **kwargs))
        chunk_grid = (np.ceil(np.asarray(shape, 'f4') / chunk_size)).astype('i2').tolist()

        metadata = {
            Dataset.__dsname__ : dict(
                shape=shape, dtype=dtype, fillvalue=fillvalue,
                chunk_grid=chunk_grid, chunk_size=chunk_size,
            )
        }

        # Filesystem
        if not op.isdir(path):
            os.makedirs(path)

        dbpath = op.join(path, Dataset.__dbname__)
        # Database
        db = AttributeDB.create(dbpath, dbtype=database)
        db.update(metadata)
        db.save()

        ds = Dataset(path, readonly=readonly)
        if data is not None:
            log.info('Loading data into dataset: {}'.format(shape))
            ds.load(data)
        return ds

    @staticmethod
    def exists(path):
        try:
            Dataset(path)
        except Exception as e:
            return False
        return True

    @staticmethod
    def remove(path):
        Dataset(path).delete()

    def delete(self):
        shutil.rmtree(self._path)

    def _idx2name(self, idx):
        if not all([type(i) == int for i in idx]) or len(idx) != self.ndim:
            raise DatasetException('Invalid chunk idx: {}'.format(idx))
        return os.path.join(self._path, 'chunk_%s.h5' % 'x'.join(map(str, idx)))

    def create_chunk(self, idx, data=None, cslices=None):
        if self.readonly:
            raise DatasetException('Dataset is in readonly mode. Cannot create chunk.')
        if self.has_chunk(idx):
            raise DatasetException('DataChunk {} already exists'.format(idx))
        path = self._idx2name(idx)

        subchunk_size = optimal_chunksize(self.chunk_size, 8)

        with h5.File(path, 'w') as f:
            chunks = optimal_chunksize(self.chunk_size, 1)
            f.create_dataset('data', shape=self.chunk_size, dtype=self.dtype,
                             fillvalue=self.fillvalue, chunks=chunks)
            if data is not None:
                slices = cslices or slice(None)
                f['data'][slices] = data
        return DataChunk(idx, path, self.chunk_size, self.dtype, self.fillvalue)

    def get_chunk(self, idx):
        if self.has_chunk(idx):
            path = self._idx2name(idx)
            return DataChunk(idx, path, self.chunk_size, self.dtype, self.fillvalue)
        return self.create_chunk(idx)

    def has_chunk(self, idx):
        return op.isfile(self._idx2name(idx))

    def del_chunk(self, idx):
        if self.readonly:
            raise DatasetException('Dataset is in readonly mode. Cannot delete chunk.')
        if self.has_chunk(idx):
            os.remove(self._idx2name(idx))

    def get_chunk_data(self, idx, slices=None):
        if self.has_chunk(idx):
            return self.get_chunk(idx)[slices]
        return self._fillvalue

    def set_chunk_data(self, idx, values, slices=None):
        if self.readonly:
            raise DatasetException('Dataset is in readonly mode. Cannot modify chunk data.')
        self.get_chunk(idx)[slices] = values

    # Data setter/getters

    def __getitem__(self, slices):
        return self.get_data(slices=slices)

    def __setitem__(self, slices, values):
        return self.set_data(values, slices=slices)

    def get_data(self, slices=None):
        slices, squeeze_axis = self._process_slices(slices, squeeze=True)
        tshape = tuple(x.stop - x.start for x in slices)
        chunk_iterator = self._chunk_slice_iterator(slices, self.ndim)

        output = np.empty(tshape, dtype=self.dtype)
        for idx, cslice, gslice in chunk_iterator:
            output[gslice] = self.get_chunk_data(idx, slices=cslice)

        if len(squeeze_axis) > 0:
            output = np.squeeze(output, axis=squeeze_axis)
        return output

    def set_data(self, values, slices=None):
        if self.readonly:
            raise DatasetException('Dataset is in readonly mode. Cannot modify data.')

        if slices is None:
            return self.load(values)

        if np.dtype(self.dtype) != np.asarray(values).dtype:
            log.warn('Performing automatic data casting from \'{}\' to \'{}\''
                     .format(np.asarray(values).dtype.name, self.dtype))

        isscalar = np.isscalar(values)
        ndim = self.ndim if isscalar else values.ndim
        slices, squeeze_axis = self._process_slices(slices, squeeze=True)
        chunk_iterator = self._chunk_slice_iterator(slices, ndim)

        for idx, cslice, gslice in chunk_iterator:
            if isscalar:
                self.set_chunk_data(idx, values, slices=cslice)
            else:
                self.set_chunk_data(idx, values[gslice], slices=cslice)

    def load(self, data):
        if tuple(data.shape) != tuple(self.shape):
            raise Exception('Data shape does not match: {} expected {}'.format(self.shape, data.shape))
        if isinstance(data, da.Array):
            data.store(self)
        else:
            for idx in range(self.total_chunks):
                idx = self.unravel_chunk_index(idx)
                gslices = self.global_chunk_bounds(idx)
                lslices = self.local_chunk_bounds(idx)
                self.set_chunk_data(idx, data[gslices], slices=lslices)

    def local_chunk_bounds(self, idx):
        return tuple((slice(0, min((i + 1) * s, self.shape[j]) - i * s)
                      for j, (i, s) in enumerate(zip(idx, self.chunk_size))))

    def global_chunk_bounds(self, idx):
        return tuple((slice(i * s, min((i + 1) * s, self.shape[j]))
                      for j, (i, s) in enumerate(zip(idx, self.chunk_size))))

    def unravel_chunk_index(self, flat_idx):
        return tuple(map(int, np.unravel_index(flat_idx, self.chunk_grid)))

    def ravel_chunk_index(self, idx):
        return tuple(map(int, np.ravel_multi_index(idx, self.chunk_grid)))

    def _process_slices(self, slices, squeeze=False):
        if type(slices) in [slice, int]:
            slices = [slices]
        elif slices is Ellipsis:
            slices = [slice(None)]
        elif np.isscalar(slices):
            slices = [int(slices)]
        elif type(slices) not in [list, tuple]:
            raise Exception('Invalid Slicing with index of type `{}`'
                            .format(type(slices)))
        else:
            slices = list(slices)

        if len(slices) <= self.ndim:
            nmiss = self.ndim - len(slices)
            while Ellipsis in slices:
                idx = slices.index(Ellipsis)
                slices = slices[:idx] + ([slice(None)] * (nmiss + 1)) + slices[idx + 1:]
            if len(slices) < self.ndim:
                slices = list(slices) + ([slice(None)] * nmiss)
        elif len(slices) > self.ndim:
            raise Exception('Invalid slicing of dataset of dimension `{}`'
                            ' with {}-dimensional slicing'
                            .format(self.ndim, len(slices)))
        final_slices = []
        shape = self.shape
        squeeze_axis = []
        for i, s in enumerate(slices):
            if type(s) == int:
                final_slices.append(slice(s, s + 1))
                squeeze_axis.append(i)
            elif type(s) == slice:
                start = s.start
                stop = s.stop
                if start is None:
                    start = 0
                if stop is None:
                    stop = shape[i]
                elif stop < 0:
                    stop = self.shape[i] + stop
                if start < 0 or start >= self.shape[i]:
                    raise Exception('Only possitive and in-bounds slicing supported: `{}`'
                                           .format(slices))
                if stop < 0 or stop > self.shape[i] or stop < start:
                    raise Exception('Only possitive and in-bounds slicing supported: `{}`'
                                           .format(slices))
                if s.step is not None and s.step != 1:
                    raise Exception('Only slicing with step 1 supported')
                final_slices.append(slice(start, stop))
            else:
                raise Exception('Invalid type `{}` in slicing, only integer or'
                                ' slices are supported'.format(type(s)))

        if squeeze:
            return final_slices, squeeze_axis
        return final_slices

    def _ndindex(self, dims):
        return itertools.product(*(range(d) for d in dims))

    def _chunk_slice_iterator(self, slices, ndim):
        indexes = []
        nchunks = []
        cslices = []
        gslices = []

        chunk_size = self.chunk_size
        chunks = self.chunk_grid

        for n, slc in enumerate(slices):
            sstart = slc.start // chunk_size[n]
            sstop = min((slc.stop - 1) // chunk_size[n], chunks[n] - 1)
            if sstop < 0:
                sstop = 0

            pad_start = slc.start - sstart * chunk_size[n]
            pad_stop = slc.stop - sstop * chunk_size[n]

            _i = []  # index
            _c = []  # chunk slices in current dimension
            _g = []  # global slices in current dimension

            for i in range(sstart, sstop + 1):
                start = pad_start if i == sstart else 0
                stop = pad_stop if i == sstop else chunk_size[n]
                gchunk = i * chunk_size[n] - slc.start
                _i += [i]
                _c += [slice(start, stop)]
                _g += [slice(gchunk + start, gchunk + stop)]

            nchunks += [sstop - sstart + 1]
            indexes += [_i]
            cslices += [_c]
            gslices += [_g]

        return (
            zip(*
                (
                    (
                        indexes[n][i],
                        cslices[n][i],
                        (n < ndim or None) and gslices[n][i],
                    )
                    for n, i in enumerate(idx)
                )
            )
            for idx in self._ndindex(nchunks)
        )


class DataChunk(object):

    def __init__(self, idx, path, shape, dtype, fillvalue):
        if not op.isfile(path):
            raise Exception('Wrong initialization of a DataChunk({}): {}'.format(idx, path))
        self._idx = idx
        self._path = path
        self._shape = shape
        self._size = np.prod(shape)
        self._dtype = dtype
        self._fillvalue = fillvalue
        self._ndim = len(shape)

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

    @property
    def dtype(self):
        return self._dtype

    @property
    def fillvalue(self):
        return self._fillvalue

    @property
    def ndim(self):
        return self._ndim

    def get_data(self, slices=None):
        if slices is None:
            slices = slice(None)
        with h5.File(self._path, 'r') as f:
            data = f['data'][slices]
        return data

    def set_data(self, values, slices=None):
        if slices is None:
            slices = slice(None)
        with h5.File(self._path, 'a') as f:
            f['data'][slices] = values

    def __getitem__(self, slices):
        return self.get_data(slices=slices)

    def __setitem__(self, slices, values):
        self.set_data(values, slices=slices)
