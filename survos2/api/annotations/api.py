

import hug
import parse
import os.path as op

import numpy as np

from survos2.api.utils import APIException, dataset_repr
from survos2.api.types import String, SmartBoolean, Float, Int, IntList, DataURI
from survos2.api import workspace as ws

from survos2.config import Config
from survos2.improc import map_blocks
from survos2.io import dataset_from_uri

__level_fill__ = 0
__level_dtype__ = 'uint8'
__group_pattern__ = 'annotations'


CHUNK_SIZE = Config['computing.chunk_size_sparse']


def to_label(idx=0, name='Label', color='#000000', visible=True, **kwargs):
    return dict(idx=idx, name=name, color=color, visible=visible)


@hug.get()
def add_level(workspace:String):
    ds = ws.auto_create_dataset(workspace, 'level', __group_pattern__,
                                __level_dtype__, fill=__level_fill__,
                                chunks=CHUNK_SIZE)
    print(ds, type(ds))
    ds.set_attr('kind', 'level')
    ds.set_attr('modified', [0] * ds.total_chunks)
    return dataset_repr(ds)


@hug.local()
def get_level(workspace:String, level:String, full:SmartBoolean=False):
    if full == False:
        return ws.get_dataset(workspace, level, group=__group_pattern__)
    return ws.get_dataset(workspace, level)


@hug.get()
@hug.local()
def get_levels(workspace:String, full:SmartBoolean=False):
    datasets = ws.existing_datasets(workspace, group=__group_pattern__)
    datasets = [dataset_repr(v) for k, v in datasets.items()]
    if full:
        for ds in datasets:
            ds['id'] = '{}/{}'.format(__group_pattern__, ds['id'])
    return datasets


@hug.get()
def rename_level(workspace:String, level:String, name:String, full:SmartBoolean=False):
    ds = get_level(workspace, level, full)
    ds.set_metadata('name', name)


@hug.get()
def delete_level(workspace:String, level:String, full:SmartBoolean=False):
    if full:
        ws.delete_dataset(workspace, level)
    else:
        ws.delete_dataset(workspace, level, group=__group_pattern__)


@hug.get()
def add_label(workspace:String, level:String, full:SmartBoolean=False):
    from survos2.improc.utils import map_blocks
    from survos2.api.annotations.annotate import erase_label

    ds = get_level(workspace, level, full)
    labels = ds.get_metadata('labels', {})
    idx = max(0, (max(l for l in labels) if labels else 0)) + 1

    if idx >= 16:
        existing_idx = set(labels.keys())
        for i in range(1, 16):
            if i not in existing_idx:
                idx = i
                break
        if idx >= 16:
            raise ValueError('Only 15 labels can be created')

    new_label = to_label(idx=idx)
    labels[idx] = new_label
    ds.set_metadata('labels', labels)
    # Erase label from dataset
    erase_label(ds, label=idx)
    return new_label


@hug.get()
def get_labels(workspace:String, level:String, full:SmartBoolean=False):
    ds = get_level(workspace, level, full)
    labels = ds.get_metadata('labels', {})
    return {k: to_label(**v) for k, v in labels.items()}


@hug.get()
def update_label(workspace:String, level:String, idx:Int, name:String=None,
                 color:String=None, visible:SmartBoolean=None,
                 full:SmartBoolean=False):
    ds = get_level(workspace, level, full)
    labels = ds.get_metadata('labels', {})
    if idx in labels:
        for k, v in (('name', name), ('color', color), ('visible', visible)):
            if v is not None:
                labels[idx][k] = v
        ds.set_metadata('labels', labels)
        return to_label(**labels[idx])
    raise APIException('Label {}::{} does not exist'.format(level, idx))


@hug.get()
def delete_label(workspace:String, level:String, idx:Int, full:SmartBoolean=False):
    ds = get_level(workspace, level, full)
    labels = ds.get_metadata('labels', {})
    if idx in labels:
        del labels[idx]
        ds.set_metadata('labels', labels)
        return dict(done=True)
    raise APIException('Label {}::{} does not exist'.format(level, idx))


@hug.get()
def annotate_voxels(workspace:String, level:String, slice_idx:Int,
                    yy:IntList, xx:IntList, label:Int, full:SmartBoolean=False):
    from survos2.api.annotations.annotate import annotate_voxels as annotate
    ds = get_level(workspace, level, full)
    annotate(ds, slice_idx=slice_idx, yy=yy, xx=xx, label=label)


@hug.get()
def annotate_regions(workspace:String, level:String, region:DataURI,
                     r:IntList, label:Int, full:SmartBoolean=False):
    from survos2.api.annotations.annotate import annotate_regions as annotate
    ds = get_level(workspace, level, full)
    region = dataset_from_uri(region, mode='r')
    annotate(ds, region, r=r, label=label)


@hug.get()
def annotate_undo(workspace:String, level:String, full:SmartBoolean=False):
    from survos2.api.annotations.annotate import undo_annotation
    ds = get_level(workspace, level, full)
    undo_annotation(ds)
