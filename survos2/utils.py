

import os
import sys
import os.path as op
import yaml
import json

import numpy as np
import base64

import time
import logging

__loggers__ = {}


def encode_numpy(ndarray):
    dtype = np.dtype(ndarray.dtype).name
    data = base64.b64encode(ndarray).decode()
    return dict(data=data, dtype=dtype, shape=ndarray.shape)


def decode_numpy(dictarray):
    data = base64.b64decode(dictarray.pop('data'))
    data = np.fromstring(data, dtype=dictarray['dtype'])
    data.shape = dictarray['shape']
    return data


def find_library(libname):
    libname, _ = op.splitext(libname)
    lib_paths = os.environ['LD_LIBRARY_PATH'].split(os.pathsep)
    for folder in lib_paths + sys.path:
        if op.isdir(folder):
            if any([f.startswith(libname) for f in os.listdir(folder)]):
                return True
    return False


def get_logger(name=None, level=None):
    if name not in __loggers__:
        logger = setup_logger(name=name, level=level)
    else:
        logger = __loggers__[name]
    if level is not None:
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
    return logger


def setup_logger(name=None, level=None):
    """
    Returns a logger formatted as specified in the SuRVoS config file.
    """
    from .config import Config
    logger = logging.getLogger(name)
    logger.handlers = []
    level = level or Config['logging.level'].upper() or logging.ERROR
    if Config['logging.std']:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        fmt = logging.Formatter(Config['logging.std_format'])
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    if Config['logging.file']:
        handler = logging.FileHandler(Config['logging.file'])
        handler.setLevel(level)
        fmt = logging.Formatter(Config['logging.file_format'])
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


class Timer(object):
    """
    Context manager to time blocks of code.

    Usage:

        with Timer('message to show'):
            # long executions of code

    When the context manager exists it will print:

        message to show - elapsed: 0.0000 seconds.
    """
    def __init__(self, name, *args):
        self.name = name
        self.args = list(args)

    def __enter__(self):
        self.tstart = time.time()
        return self

    def push(self, *args):
        self.args.extend(args)

    def __exit__(self, type, value, traceback):
        self.tend = (time.time() - self.tstart)
        if len(self.args):
            logging.info('{0}: {1:.4f} seconds, Args: {2}'
                         .format(self.name, self.tend, tuple(self.args)))
        else:
            logging.info('{0}: {1:.4f} seconds'
                         .format(self.name, self.tend))


def check_relpath(path1, path2, exception=True):
    """
    Checks that the real path of `path2` is inside `path1`.

    Parameters
    ----------
    path1: str
        Path-like string. It can be relative or absolute.
    path2: str
        Path-like string. It can be relative or absolute.
    exception: bool (optional)
        Whether to return `False` or raise an exception in case
        `path2` is not relative to `path1`.

    Returns
    -------
    flag : str or bool
        Returns the full path from `path1` to `path2` or False
        if `path2` is not relative to `path1` (if `exception=False`).
    """
    p1 = op.normpath(path1)
    p2 = op.normpath(op.join(path1, path2))
    if op.relpath(p1, p2).endswith(op.basename(p1)):
        if exception:
            raise ValueError('Invalid path \'%s\'' % path2)
        return False
    return p2


class AttributeDB(dict):
    """
    Extends a dictionary to add `read` and `save` functionality. It allows
    to dump or load its contents to either JSON or YAML files.

    Parameters
    ----------
    filename : string
        Path of the filename where to read/write its contents.
    dbtype : string
        The writing backend to choose. Values are 'yaml' or 'json'.
    """

    def __init__(self, filename, dbtype='yaml'):
        super(AttributeDB, self).__init__()
        self.use_yaml = dbtype == 'yaml'
        self.filename = AttributeDB.dbpath(filename, dbtype)
        self.read(self.filename)

    @staticmethod
    def dbpath(filename, dbtype):
        if not filename.endswith('.' + dbtype):
            filename += '.' + dbtype
        return filename

    @staticmethod
    def create(filename, dbtype='yaml'):
        filename = AttributeDB.dbpath(filename, dbtype)
        if os.path.isfile(filename):
            raise FileExistsError('Database file \'{}\' already exists.'
                                  .format(filename))
        with open(filename, 'w') as f:
            if dbtype == 'yaml':
                os.utime(filename, None)
            elif dbtype == 'json':
                f.write('{}')
        return AttributeDB(filename, dbtype='yaml')

    def read(self, filename=None, exists_ok=False):
        self.clear()
        filename = filename or self.filename
        data = None
        with open(filename) as handle:
            if self.use_yaml:
                data = yaml.safe_load(handle.read())
            else:
                data = json.load(handle)
        self.update(data or [])

    def isserializable(self, value):
        try:
            if self.use_yaml:
                yaml.dump(value)
            else:
                json.dumps(value)
            return True
        except:
            return False

    def save(self, filename=None):
        filename = filename or self.filename
        with open(filename, 'w') as handle:
            if self.use_yaml:
                yaml.dump(dict(self), handle, indent=4,
                          explicit_start=True, explicit_end=True)
            else:
                json.dump(dict(self), handle, sort_keys=True, indent=4)


def _canpickle(obj):
    import pickle
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False

def _transform_params(data):
    result = dict()
    for k, v in data.items():
        if not _canpickle(v):
            continue
        if type(v) == tuple:
            v = list(v)
        elif hasattr(v, 'tolist'):
            v = v.tolist()
        if type(v) == bytes:
            v = str(v)
        result[k] = v
    return result

def parse_params(data):
    d = _transform_params(data)
    if 'pipeline' in data:
        d.update({f.__name__: parse_params(p) for f, p in data['pipeline']})
        del d['pipeline']
    return d


def format_yaml(data, flow=None, **kwargs):
    data = parse_params(data)
    kwargs.setdefault('explicit_start', True)
    kwargs.setdefault('explicit_end', True)
    kwargs.update(dict(default_flow_style=flow))
    return yaml.dump(data, **kwargs)[:-1]