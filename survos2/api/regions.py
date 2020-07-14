

import hug
import logging
import os.path as op

import numpy as np
import dask.array as da

from survos2.utils import encode_numpy
from survos2.io import dataset_from_uri
from survos2.improc import map_blocks
from survos2.api.utils import save_metadata, dataset_repr
from survos2.api.types import DataURI, Float, SmartBoolean, \
    FloatOrVector, IntList, String, Int, FloatList, DataURIList
from survos2.api import workspace as ws


__region_fill__ = 0
__region_dtype__ = 'uint32'
__region_group__ = 'regions'
__region_names__ = [None, 'supervoxels', 'megavoxels']


@hug.get()
def get_slice(src:DataURI, slice_idx:Int):
    ds = dataset_from_uri(src, mode='r')
    data = ds[slice_idx]
    return encode_numpy(data)

@hug.get()
@save_metadata
def supervoxels(src:DataURIList, dst:DataURI, shape:IntList=[10,10,10],
                compactness:Float=30, spacing:FloatList=[1,1,1]):
    """
    API wrapper for `survos2.improc.regions.slic3d`.
    """
    from ..improc.regions.slic import slic3d
    map_blocks(slic3d, *src, out=dst, sp_shape=shape, spacing=spacing,
               compactness=compactness, stack=True)


@hug.get()
@save_metadata
def connected_components(src:DataURI, dst:DataURI, remap:SmartBoolean):
    """
    API wrapper for `survos2.improc.regions.ccl3d`.
    """
    from ..improc.regions.ccl import ccl3d
    map_blocks(ccl3d, src, out=dst, remap=remap)


@hug.get()
@save_metadata
def merge_regions(src:DataURI, labels:DataURI, dst:DataURI, min_size:Float):
    """
    API wrapper for `survos2.improc.regions.merge_small`.
    """
    from ..improc.regions.ccl import merge_small
    map_blocks(merge_small, src, labels, out=dst, min_size=min_size)


@hug.get()
def create(workspace:String, order:Int=1):
    region_type = __region_names__[order]
    ds = ws.auto_create_dataset(workspace, region_type, __region_group__,
                                __region_dtype__, fill=__region_fill__)
    ds.set_attr('kind', region_type)
    return dataset_repr(ds)


@hug.get()
@hug.local()
def existing(workspace:String, full:SmartBoolean=False, order:Int=1):
    filter = __region_names__[order]
    datasets = ws.existing_datasets(workspace, group=__region_group__, filter=filter)
    if full:
        return {'{}/{}'.format(__region_group__, k): dataset_repr(v)
                for k, v in datasets.items()}
    return {k: dataset_repr(v) for k, v in datasets.items()}


@hug.get()
def remove(workspace:String, region_id:String):
    ws.delete_dataset(workspace, region_id, group=__region_group__)


@hug.get()
def rename(workspace:String, region_id:String, new_name:String):
    ws.rename_dataset(workspace, region_id, __region_group__, new_name)