

import os
import os.path as op
import shutil

import numpy as np
import tempfile
import logging as log

from survos2.config import Config
from survos2.utils import check_relpath
from survos2.model.dataset import Dataset


CHROOT = Config['model.chroot']
DATABASE = Config['model.dbtype']
CHUNK_DATA = Config['computing.chunks']
CHUNK_SIZE = Config['computing.chunk_size']

if CHROOT in ['tmp', 'temp']:
    tmp = tempfile.gettempdir()
    CHROOT = op.join(tmp, 'tmp_survos_chroot')
    os.makedirs(CHROOT, exist_ok=True)


class WorkspaceException(Exception):
    pass


class Workspace(object):

    __dsname__ = '__data__'

    def __init__(self, path):
        path = self._validate_path(path)
        if not op.isdir(path):
            raise WorkspaceException('Workspace \'{}\' does not exist.'.format(path))
        self._path = path

    # JSON representation

    @property
    def path(self):
        return self._path

    def tojson(self):
        ws = {}
        for session_name in self.available_sessions():
            ws[session_name] = []
            for ds in self.available_datasets(session_name):
                ws[session_name].append(ds)
        return dict(path=self._path, data=ws)

    @staticmethod
    def _validate_path(path):
        if not CHROOT and op.realpath(path) != path:
            raise WorkspaceException('\'{}\' is not a valid workspace path without CHROOT'
                                     .format(path))
        elif CHROOT:
            path2 = check_relpath(CHROOT, path, exception=False)
            if path2 is False:
                raise WorkspaceException('Invalid workspace path: {}. '
                                         'Workspace is in chroot mode at {}'
                                         .format(path, CHROOT))
            path = path2
        return path

    # Creation + deletion

    @staticmethod
    def create(path):
        path = Workspace._validate_path(path)
        if op.isdir(path) and os.listdir(path):
            raise WorkspaceException('Directory \'%s\' is not empty.' % path)
        elif not op.isdir(path):
            os.makedirs(path)
        return Workspace(path)

    @staticmethod
    def remove(path):
        Workspace(path).delete()


    def delete(self):
        shutil.rmtree(self._path)


    @staticmethod
    def exists(path):
        try:
            Workspace(path)
        except:
            return False
        return True

    # Auxiliary

    def genpath(self, *args):
        for i in range(len(args)-1):
            check_relpath(args[i], args[i+1])
        kpath = op.normpath(op.join(*args))
        path = op.realpath(op.join(self._path, kpath))
        check_relpath(self._path, path)
        return path

    # Inspect workspace

    def available_sessions(self):
        if not self.has_data():
            return []
        path = self.genpath()
        return [sess for sess in os.listdir(path) if self.has_session(sess)]

    def available_datasets(self, session='default', group=None):
        if not self.has_session(session):
            return []
        path = self.genpath(session, group) if group else self.genpath(session)
        if not op.isdir(path):
            return []
        datasets = [ds for ds in os.listdir(path)]
        if group:
            datasets = [op.sep.join([group, ds]) for ds in datasets]
        return [ds for ds in datasets if self.has_dataset(ds, session=session)]

    # Data

    def add_data(self, data):
        if self.has_data():
            raise WorkspaceException('Workspace has already been initialized with data.')
        path = self.genpath(self.__dsname__)
        chunks = CHUNK_SIZE if CHUNK_DATA else None
        Dataset.create(path, data=data, chunks=chunks)
        self.add_session('default')
        return self.get_data()

    def has_data(self):
        path = self.genpath(self.__dsname__)
        return Dataset.exists(path)

    def metadata(self):
        if not self.has_data():
            raise WorkspaceException('Workspace data has not been initialized')
        path = self.genpath(self.__dsname__)
        return Dataset(path).get_metadata(Dataset.__dsname__)

    def get_data(self, **kwargs):
        kwargs.setdefault('readonly', True)
        if not self.has_data():
            raise WorkspaceException('Workspace data has not been initialized')
        path = self.genpath(self.__dsname__)
        return Dataset(path, **kwargs)

    # Session

    def add_session(self, session):
        if self.has_session(session):
            raise WorkspaceException('Session \'%s\' already exists.' % (session))
        elif session == self.__dsname__:
            raise WorkspaceException('Invalid session name: \'%s\'' % session)
        # Update filesystem
        path = self.genpath(session)
        if not op.isdir(path):
            os.makedirs(path)

    def remove_session(self, session):
        if not self.has_session(session):
            raise WorkspaceException('Session \'{}\' does not exists.'
                                     .format(session))
        # Update filesystem
        path = self.genpath(session)
        shutil.rmtree(path)

    def has_session(self, session):
        if not self.has_data():
            raise WorkspaceException('Workspace data has not been initialized')
        path = self.genpath(session)
        return session != self.__dsname__ and op.isdir(path)

    # Datasets

    def add_dataset(self, dataset_name, dtype, session='default',
                    fillvalue=0, chunks=None, shape=None):
        dataset_name = dataset_name.replace('/', op.sep)
        if self.has_dataset(dataset_name, session=session):
            raise WorkspaceException('Dataset \'{}::{}\' already exists.'
                                     .format(session, dataset_name))

        metadata = self.metadata()
        shape = shape or metadata['shape']
        chunk_size = chunks or metadata['chunk_size']
        dtype = np.dtype(dtype).name

        path = self.genpath(session, dataset_name)
        return Dataset.create(path, shape=shape, dtype=dtype, chunks=chunk_size,
                              fillvalue=fillvalue, database=DATABASE)

    def remove_dataset(self, dataset_name, session='default'):
        dataset_name = dataset_name.replace('/', op.sep)
        if not self.has_dataset(dataset_name, session=session):
            raise WorkspaceException('Dataset \'{}::{}\' does not exist.'
                                     .format(session, dataset_name))
        path = self.genpath(session, dataset_name)
        shutil.rmtree(path)

    def has_dataset(self, dataset_name, session='default'):
        dataset_name = dataset_name.replace('/', op.sep)
        if self.has_session(session):
            path = self.genpath(session, dataset_name)
            return Dataset.exists(path)
        return False

    def get_dataset(self, dataset_name, session='default', **kwargs):
        dataset_name = dataset_name.replace('/', op.sep)
        if not self.has_dataset(dataset_name, session=session):
            raise WorkspaceException('Dataset \'{}::{}\' does not exist.'
                                     .format(session, dataset_name))
        path = self.genpath(session, dataset_name)
        ds = Dataset(path, **kwargs)
        if tuple(ds.shape) != tuple(self.metadata()['shape']):
            raise WorkspaceException('Dataset \'{}::{}\' has incorrect `shape`.'
                                     'Got {}, expected {}.'
                                     .format(session, dataset_name,
                                             tuple(ds.shape),
                                             tuple(self.metadata()['shape'])))
        return ds


