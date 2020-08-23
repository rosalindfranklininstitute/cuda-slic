import logging
import time

from functools import partial, wraps
from itertools import product
from math import ceil

import dask.array as da
import numpy as np
import yaml


# from dask.array.core import slices_from_chunks

# from ..config import Config
# from ..utils import format_yaml, parse_params, get_logger


# CHUNK = Config['computing.chunks']
# CHUNK_SIZE = Config['computing.chunk_size']
# CHUNK_PAD = Config['computing.chunk_padding']
# SCALE = Config['computing.scale']
# STRETCH = Config['computing.stretch']


# logger = get_logger()


def asnparray(data, dtype=None, contiguous=True):
    """
    Transforms an input array to a numpy array. The known input
    formats are `pycuda.gpuarray.GPUArray`, `dask.array.Array` and
    `numpy.ndarray`. If a different type is given a conversion
    will be attempted assuming that the input accepts numpy-like
    slicing.

    Parameters
    ----------
    data: numpy-like array
        The input array to be transformed. If a numpy.ndarray is given
    """
    newdata = None
    dtype = dtype or data.dtype

    if isinstance(data, da.Array):
        newdata = data.compute()
    elif hasattr(data, "get"):  # is a pycuda.gpuarray.GPUArray
        newdata = data.get()
    elif isinstance(data, np.ndarray):
        newdata = data

    if newdata is not None:
        if newdata.dtype != dtype:
            logger.warn(
                "Transforming data type from {} to {}:".format(
                    np.dtype(newdata.dtype).name, np.dtype(dtype).name
                )
            )
            newdata = newdata.astype(dtype, copy=False)
        elif newdata.flags.c_contiguous == False and contiguous:
            newdata = newdata.copy()
        return newdata
    else:
        logger.warn(
            "Transforming unkown data to a numpy array. "
            "Expected pycuda.gpuarray.GPUArray, dask.array.Array "
            "or numpy.ndarray."
        )
        return np.asarray(data[...], dtype=dtype)


# def optimal_chunksize(source, max_size, item_size=4, delta=0.1, axis_weight=None):
#     """
#     Obtain the optimal chunk size to split a large dataset in.

#     Parameters
#     ----------
#     source : iterable of integers (list, tuple) or Numpy array-like
#         The shape of the whole dataset being chunked. If an iterable is provided,
#         it is assumed to contain the shape of the object. Otherwise, an object with
#         `shape` and `dtype` attributes is expected.
#     max_size : number
#         The maximum size (in MegaBytes) of the desired chunks.
#     item_size : int
#         The size (in bytes) of the data type. Only used if `source` does not contain
#         a `dtype` attribute.
#     delta: float
#         The amount by which to decrease the total shape iteratively. Smaller amounts
#         would allow for better optimal chunk sizes at the expenses of slower
#         computations.
#     axis_weight : None or array-like
#         Controls the weight of each of the axes. If `None` the importance of the axes
#         will be calculated from `source`'s shape, otherwise an array-like is
#         expected with a float value for each dimension.

#     Returns
#     -------
#     chunk_size: tuple
#         Optimal chunk size.
#     """
#     if hasattr(source, 'shape') and hasattr(source, 'dtype'):
#         item_size = np.dtype(source.dtype).itemsize
#         shape = np.asarray(source.shape, np.int16)
#     else:
#         shape = np.asarray(source, np.int16)

#     sizeMB = max_size * (2**20)
#     if axis_weight is None:
#         axis_weight = shape / float(np.min(shape))
#     axis_weight = 1 + np.asarray(axis_weight, float) / np.max(axis_weight)
#     total_chunks = int(ceil(np.prod(shape) * item_size / sizeMB))
#     total_chunks_axis = [int(ceil(total_chunks / p)) for p in axis_weight]
#     max_chunk_iter = [range(1, total+1) for total in total_chunks_axis]

#     best_chunk = shape
#     best_chunk_err = np.inf
#     for nchunks in product(*max_chunk_iter):
#         chunks = np.ceil(shape / nchunks).astype(int)
#         chunk_size = np.prod(chunks) * item_size
#         chunk_size_err = (sizeMB - chunk_size) / (2**20)
#         chunk_axis_err = np.abs(
#             chunks / float(chunks.min()) - axis_weight).sum()
#         chunk_err = chunk_size_err + chunk_axis_err
#         if chunk_size < sizeMB and chunk_err < best_chunk_err:
#             best_chunk = chunks
#             best_chunk_err = chunk_err

#     return tuple(map(int, best_chunk))


# def dask_relabel_chunks(A):
#     """
#     Relabel all the the chunks of an input array. It is assumed
#     that a `map_blocks` or `map_overlap`, has been previously applied
#     over `A`, and each of the chunks of `A` contains a local labelling.

#     The function labels all the chunks so that all the labels all
#     globally independent.

#     E.g. for an input dataset with 3 different chunks of local labels:

#     Chunk1: [1, 0, 0] # Maxlabel = 1
#     Chunk2: [0, 2, 1] # Maxlabel = 2
#     Chunk3: [2, 0, 1] # Maxlabel = 2

#     The relabelling of the chunks would look like:

#     newChunk1: [1, 0, 0]
#     newChunk2: [2, 4, 3] # Chunk2 + Maxlabel(newChunk1) + 1
#     newChunk3: [7, 5, 6] # Chunk3 + Maxlabel(newChunk2) + 1

#     Parameters
#     ----------
#     A: dask.Array
#         An input array to be relabeled

#     Returns
#     -------
#     B: dask.Array
#         Dask array of the same shape, with chunks relabelled.
#     """
#     inds = tuple(range(A.ndim))
#     max_per_block = da.blockwise(np.max, inds, A, inds, axis=inds,
#                             keepdims=True, dtype=A.dtype,
#                             adjust_chunks={i: 1 for i in inds})
#     block_index_global = da.cumsum(max_per_block.ravel() + 1)

#     def relabel(a, block_id=None):
#         bid = int(np.ravel_multi_index(block_id, A.numblocks))
#         if bid == 0:
#             return a
#         return a + block_index_global[bid-1]

#     return A.map_blocks(relabel, dtype=np.float64)


# def _chunk_datasets(datasets, chunk=CHUNK, chunk_size=CHUNK_SIZE, stack=False):
#     """
#     Process a set of input datasets and returns chunked `Dask.Array`s
#     if `chunk = True`. The optimal chunk size is estimated from the
#     maximum `chunk_size`. All datasets are assumed to have the same
#     shape.

#     Parameters
#     ----------
#     datasets: list of numpy array-like
#         List of input datasets
#     chunk: boolean
#         Wether to chunk the data. If `False` `datasets` is returned
#         with no modification. Otherwise, a list with each of the
#         datasets mapped to `Dask.Array`s is returned.
#     chunk_size: int or iterable
#         If a tuple or list is given with length equals to the number of
#         dimensions of the input datasets, it is used as the chunk size.
#         If a number is given, the maximum size in MB is assumed, and
#         optimal chunking is estimated. See `optimal_chunksize` for more
#         details.
#     stack: bool
#         If datasets are going to be stacked or not.

#     Returns
#     -------
#     datasets: list of numpy array-like
#         If `chunk = True` a list of `Dask.Array` is returned, otherwise
#         the input is returned with no modification.
#     """
#     newds = []
#     if chunk:
#         if isinstance(datasets[0], da.Array):  # override chunk size
#             chunk_size = tuple(a[0] for a in datasets[0].chunks)
#         elif hasattr(datasets[0], 'chunk_size'):
#             chunk_size = datasets[0].chunk_size
#         elif np.isscalar(chunk_size):
#             #if stack:
#             #    chunk_size = chunk_size / float(len(datasets))
#             chunk_size = optimal_chunksize(datasets[0], chunk_size)
#         elif len(chunk_size) != datasets[0].ndim:
#             raise ValueError('Chunk size has different dimension than '
#                              'source volume.')
#         for i in range(len(datasets)):
#             if isinstance(datasets[i], da.Array):
#                 newds.append(datasets[i])
#             else:
#                 chunk_size = tuple(map(int, chunk_size))
#                 newds.append(da.from_array(datasets[i], chunks=chunk_size))
#     else:
#         newds = datasets

#     return newds


# def _preprocess_datasets(datasets, chunk=CHUNK, scale=SCALE, stretch=STRETCH):
#     """
#     Stretch and/or scale input datasets. If `chunk = True` dask will be used
#     for stretch and scaling, otherwise numpy.

#     Parameters
#     ----------
#     datasets: list of numpy array-like
#         List of input datasets
#     chunk: boolean
#         If `True` use dask to preprocess datasets lazily, otherwise numpy.
#     scale: boolean
#         If `True` normalize the datasets to `[0, 1]` range.
#     stretch: boolean
#         If `True` clip the intensity of the dataset to 1 and 99 percentiles.

#     Returns
#     -------
#     datasets: list of numpy array-like
#         Preprocessed datasets according to flags.
#     """
#     proc = da if chunk else np

#     newds = []

#     for i in range(len(datasets)):
#         d = datasets[i]
#         if stretch:
#             a, b = proc.percentile(proc.ravel(d), (1, 99))
#             d = proc.clip(d, a, b)
#         if scale:
#             dmin = proc.min(d)
#             d = (d - dmin) / (proc.max(d) - dmin)
#         newds.append(d)

#     return newds


# def _apply(func, datasets, chunk=CHUNK, pad=None, relabel=False,
#            stack=False, compute=True, out=None, normalize=False,
#            **kwargs):
#     """
#     Appplies a function to a given set of datasets. Wraps a standard
#     function call of the form:

#         func(*datasets, **kwargs)

#     Named parameters gives extra functionality.

#     Parameters
#     ----------
#     func: callable
#         Function to be mapped across datasets.
#     datasets: list of numpy array-like
#         Input datasets.
#     chunk: boolean
#         If `True` then input datasets will be assumed tobe `Dask.Array`s and
#         the function will be mapped across arrays blocks.
#     pad: None, int or iterable
#         The padding to apply (only if `chunk = True`). If `pad != None` then
#         `dask.array.overlap.map_overlap` will be used to map the function across
#         overlapping blocks, otherwise `dask.array.map_blocks` will be used.
#     relabel: boolean
#         Some of the labelling functions will yield local labelling if `chunk=True`.
#         If `func` is a labelling function, set `relabel = True` to map the result
#         for global consistency. See `survos2.improc.utils.dask_relabel_chunks` for
#         more details.
#     compute: boolean
#         If `True` the result will be computed and returned in numpy array form,
#         otherwise a `dask.delayed` will be returned if `chunk = True`.
#     out: None or numpy array-like
#         if `out != None` then the result will be stored in there.
#     **kwargs: other keyword arguments
#         Arguments to be passed to `func`.

#     Returns
#     -------
#     result: numpy array-like
#         The computed result if `compute = True` or `chunk = False`, the result
#         of the lazy wrapping otherwise.
#     """
#     if stack and len(datasets) > 1:
#         dataset = da.stack(datasets, axis=0)
#         dataset = da.rechunk(dataset, chunks=(
#             dataset.shape[0],) + dataset.chunks[1:])
#         datasets = [dataset]

#     if chunk == True:
#         kwargs.setdefault('dtype', out.dtype if out else datasets[0].dtype)
#         kwargs.setdefault('drop_axis', 0 if stack else None)
#         if pad is None or pad == False:
#             result = da.map_blocks(func, *datasets, **kwargs)
#         elif len(datasets) == 1:
#             if np.isscalar(pad):
#                 pad = [pad] * datasets[0].ndim

#             if stack:
#                 pad[0] = 0  # don't pad feature channel
#                 depth = {i: d for i, d in enumerate(pad)}
#                 trim = {i: d for i, d in enumerate(pad[1:])}
#             else:
#                 depth = trim = {i: d for i, d in enumerate(pad)}
#             g = da.overlap.overlap(
#                 datasets[0], depth=depth, boundary='reflect')
#             #g = da.ghost.ghost(datasets[0], depth=depth, boundary='reflect')
#             r = g.map_blocks(func, **kwargs)
#             #result = da.ghost.trim_internal(r, trim)
#             result = da.overlap.trim_internal(r, trim)
#         else:
#             raise ValueError('`pad` only works with single')

#         rchunks = result.chunks

#         if not relabel and normalize:
#             result = result / da.nanmax(da.fabs(result))

#         if out is not None:
#             result.store(out, compute=True)
#         elif compute:
#             result = result.compute()

#         if relabel:
#             if out is not None:
#                 result = dask_relabel_chunks(
#                     da.from_array(out, chunks=rchunks))
#                 result.store(out, compute=True)
#             else:
#                 result = dask_relabel_chunks(
#                     da.from_array(result, chunks=rchunks))
#                 if compute:
#                     result = result.compute()
#     else:
#         result = func(*datasets, **kwargs)
#         if out is not None:
#             out[...] = result

#     if out is None:
#         return result


# def map_blocks(func, *args, chunk=CHUNK, chunk_size=CHUNK_SIZE, pad=CHUNK_PAD,
#                compute=True, scale=SCALE, stretch=STRETCH, relabel=False,
#                out=None, out_dtype=None, out_fillvalue=None, uses_gpu=False,
#                timeit=False, stack=False, normalize=False, **kwargs):
#     """
#     Computes a function over a set of input datasets of the same size, optionally
#     distributing it across blocks of `chunk_size` if `chunk = True`.

#     Parameters:
#     -----------
#     func: function
#         The function to map across blocks.
#     *args: list of numpy-like arrays
#         All the source arrays in `args` have to have the same shape.
#     chunk: boolean
#         If `True` Dask will be used to map a function across chunks in parallel.
#         Default: `computing.chunks` in the config file.
#     chunk_size: int or tuple
#         If `chunks=True` this will specify the size of the chunks. If a scalar is
#         given a size in MB is asssumed and optimal chunks are calculated.
#         Default: `computing.chunk_size` in the config file.
#     stretch: boolean
#         If `True` the contrast of the volume will be stretched by cliping the
#         data to the 1% and 99% percentiles. See `np.percentile` and histogram
#         stretching for more information.
#         Default: `computing.stretch` in the config file.
#     scale: boolean
#         If `True` all the input datasets will be scald to the `[0, 1]` range.
#         Default: `computing.scale` in the config file. Note, if both `scale`
#         and `stretch` are True, `stretch` happens first and data is rescaled
#         afterwards.
#     pad: int, tuple or None
#         The padding between blocks when using `chunk=True`. `pad` only works with
#         single input functions. Default: None
#     compute: bool
#         If `True` the result of the dask computation (if `chunk=True`) will be
#         calculted. If `False` the corresponding lazy Dask array will be returned.
#         Default: True
#     out: numpy-like array
#         if `out` is given, the result of the computation will be stored on it. If
#         `chunk=True` and `out` is provided the value of `compute` will always be `True`.
#     out_dtype: data type
#         The data type of the output, in case `out` is a string URI. Default: `'float32'`
#     out_fillvalue: number
#         The fill value for the output, in case `out` is an URI. Defualt: `0`
#     uses_gpu: boolean
#         If True a new PyCUDA context will be created to run the function. NOTE: The
#         function has to be a CUDA function, this argument does not magically convert
#         the function to run in the GPU. Default: False
#     timeit: boolean
#         If `True` the function call will be timed and logged. Default: `False`.
#     stack: boolean
#         Whether to stack all the input datasets into a single ndim+1 dataset.
#     **kwargs:
#         other keyword arguments for the specific function being mapped.
#     """
#     uses_gpu = uses_gpu or getattr(func, '__uses_gpu__', False)
#     out_dtype = out_dtype or getattr(func, '__out_dtype__', np.float64)
#     out_fillvalue = out_fillvalue or getattr(func, '__out_fillvalue__', 0) or 0
#     relabel = relabel or getattr(func, '__requires_relabel__', False)

#     params = dict()
#     params['params'] = parse_params(kwargs)
#     params['blocks'] = parse_params(
#         dict(chunk=chunk, chunk_size=chunk_size, pad=pad))
#     params['preprocess'] = dict(scale=scale, stretch=stretch)
#     params['postprocess'] = dict(relabel=relabel, compute=compute)
#     params['misc'] = dict(uses_gpu=uses_gpu, timeit=timeit)
#     params['out'] = dict(dtype=np.dtype(out_dtype).name, fill=out_fillvalue)

#     func_showname = '({}::{})'.format(func.__name__, np.dtype(out_dtype).name)
#     logger.info('Launching {}'.format(func_showname))
#     logger.debug('\n{}'.format(format_yaml(params)))

#     if uses_gpu:
#         from .cuda import gpu_context_wrapper
#         func = gpu_context_wrapper(func)

#     if timeit:
#         t0 = time.time()

#     with DatasetManager(*args, out=out, dtype=out_dtype, fillvalue=out_fillvalue) as DM:

#         datasets = _chunk_datasets(DM.sources, chunk=chunk,
#                                    chunk_size=chunk_size, stack=stack)
#         datasets = _preprocess_datasets(datasets, chunk=chunk, scale=scale,
#                                         stretch=stretch)
#         result = _apply(func, datasets, chunk=chunk, pad=pad, relabel=relabel,
#                         stack=stack, compute=compute, out=DM.out,
#                         normalize=normalize, **kwargs, dtype=out_dtype)

#     if timeit:
#         t1 = time.time()
#         logger.info('{0} - elapsed: {1:.4f}'.format(func.__name__, t1 - t0))

#     if out is None:
#         return result


# class DatasetManager(object):
#     """
#     In an out dataset manager. Uses `survos2.io.dataset_from_uri` to allow
#     functions to receive input and output datasets from strings. Implements
#     a context manager that closes and opens datasets automatically.
#     """

#     def __init__(self, *args, out=None, dtype=None, fillvalue=0, src_mode='r'):
#         from ..io import dataset_from_uri, is_dataset_uri

#         self._src_mode = src_mode
#         self._closed = False
#         self._sources = []
#         self._out = out

#         for source in args:
#             if is_dataset_uri(source):
#                 source = dataset_from_uri(source, mode=src_mode)
#             self._sources.append(source)

#         if is_dataset_uri(out):
#             shape = self._sources[0].shape
#             dtype = dtype or self._sources[0].dtype
#             out = dataset_from_uri(out, mode='w', shape=shape,
#                                    dtype=dtype, fill=fillvalue)
#             self._out = out

#     def __enter__(self):
#         if self._closed:
#             raise RuntimeError('DatasetManager has already been closed.')
#         return self

#     def __exit__(self, *args):
#         for f in self._sources + [self._out]:
#             if f is not None and hasattr(f, 'close'):
#                 f.close()
#         self._closed = True

#     @property
#     def sources(self):
#         return self._sources

#     @property
#     def out(self):
#         return self._out


# def cpu_argument_wrapper(func, dtype):
#     """
#     The result of the wrapepd function will satisfy a data type (if given).
#     """
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         r = func(*args, **kwargs)
#         return r is None or asnparray(r, dtype=dtype)
#     return wrapper


# def survosify(func=None, dtype=None, fillvalue=0, uses_gpu=False, relabel=False):
#     """
#     Wraps a **mapping function** to add extra parameters and common functionality.
#     A **mapping function** has the following form: `mapping(*args, **kwargs)`
#     where `*args` are source datasets and keyword arguments define options for the
#     function. All the source datasets and the output dataset have to have the same
#     shape.

#     The arguments of the function that will result in a decorator:

#     Arguments:
#     ----------
#     dtype: data type
#         The data type of the returning datasets. Default: `'float32'`
#     fillvalue: number
#         The default value to fill the resulting datasets. Default: `0`
#     uses_gpu: bool
#         If `True` the function to wrap will generate its own context for the GPU,
#         allowing chunking operations to run in parallel in the GPU. Default: `False`.
#     relabel: bool
#         Indicates if the function returns labelling. If so, when applied with
#         `compute_blocks` the result would yield local labels, which have to be
#         recalculated globally. See `survos2.improc.utils.dask_relabel_chunks`
#         for more details. Default: `False`.

#     See `compute` for more information.
#     """
#     if func is None:
#         return partial(survosify, dtype=dtype, fillvalue=fillvalue, uses_gpu=uses_gpu)

#     if uses_gpu:
#         from .cuda import gpu_argument_wrapper
#         func = gpu_argument_wrapper(func)
#     elif dtype is not None:
#         func = cpu_argument_wrapper(func, dtype=dtype)

#     @wraps(func)
#     def wrapper(*args, out=None, src_mode='r', **kwargs):
#         dm_params = dict(out=out, dtype=dtype,
#                          fillvalue=fillvalue, src_mode=src_mode)
#         with DatasetManager(*args, **dm_params) as DM:
#             result = func(*DM.sources, **kwargs)
#             if out is not None:
#                 DM.out[...] = result
#             else:
#                 return result

#     wrapper.__uses_gpu__ = uses_gpu
#     wrapper.__out_dtype__ = dtype
#     wrapper.__out_fillvalue__ = fillvalue
#     wrapper.__requires_relabel__ = relabel

#     return wrapper


# def gpufeature(func):
#     params = dict(uses_gpu=True, dtype='float32', fillvalue=0, relabel=False)
#     return survosify(func=func, **params)


# def gpuregion(func):
#     params = dict(uses_gpu=True, dtype='uint32', fillvalue=0, relabel=True)
#     return survosify(func=func, **params)


# def cpufeature(func):
#     params = dict(uses_gpu=False, dtype='float32', fillvalue=0, relabel=False)
#     return survosify(func=func, **params)


# def cpuregion(func):
#     params = dict(uses_gpu=False, dtype='uint32', fillvalue=0, relabel=True)
#     return survosify(func=func, **params)


# def cpulabel(func):
#     params = dict(uses_gpu=False, dtype='uint8', fillvalue=0, relabel=False)
#     return survosify(func=func, **params)


# def map_pipeline(source, pipeline=None, **kwargs):
#     """
#     Distributes a pipeline of functions across multiple blocks.

#     Parameters
#     ----------
#     source: numpy array-like
#         Array of a given shape and dtype that will be the source of the pipeline.

#     pipeline: list of (func, func_args) pairs
#         A list of functions to be called in a `(func, func_args)` format, where
#         `func_args` is a dictionary containing the parameters of `func`.
#     **kwargs:
#         All other arguments determining how to map the functions across blocks.
#         See `survos2.improc.utils.map_blocks` for more details.

#     Results
#     -------
#     result: numpy array-like
#         Result if `kwargs['out'] is None`, otherwise the result will be stored in
#         `kwargs['out']`.

#     """
#     rfunc = pipeline[-1][0]
#     uses_gpu = any(getattr(f[0], '__uses_gpu__', False) for f in pipeline)
#     wparams = dict(
#         uses_gpu=uses_gpu,
#         dtype=getattr(rfunc, '__out_dtype__', 'float32'),
#         fillvalue=getattr(rfunc, '__out_fillvalue__', 0),
#         relabel=getattr(rfunc, '__requires_relabel__', False)
#     )

#     def wrapper(source, pipeline=None):
#         r = source
#         for i, (func, fparams) in enumerate(pipeline):
#             if uses_gpu and getattr(func, '__uses_gpu__', False):
#                 fparams['keep_gpu'] = (i < len(pipeline) - 1)
#             r = func(r, **fparams)
#         return r

#     fnames = [f[0].__name__ for f in pipeline]
#     wrapper.__name__ = '=>'.join(fnames)

#     func = survosify(wrapper, **wparams)
#     return map_blocks(func, source, pipeline=pipeline, **kwargs)
