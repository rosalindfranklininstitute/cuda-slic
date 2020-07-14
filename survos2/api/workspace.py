

import hug
import parse
import os.path as op

from survos2.api.utils import APIException
from survos2.api.types import String, Int, IntOrNone

from survos2.utils import get_logger
from survos2.io import dataset_from_uri
from survos2.config import Config
from survos2.model import Workspace, Dataset


logger = get_logger()

### Workspace

@hug.get()
def create(workspace:String):
    workspace, session = parse_workspace(workspace)
    return Workspace.create(workspace)


@hug.get()
def delete(workspace:String):
    workspace, session = parse_workspace(workspace)
    Workspace.remove(workspace)
    return dict(done=True)


@hug.get()
@hug.local()
def get(workspace:String):
    workspace, session = parse_workspace(workspace)
    if Workspace.exists(workspace):
        return Workspace(workspace)
    raise APIException('Workspace \'%s\' does not exist.' % workspace)

### Data

@hug.get()
def add_data(workspace:String, dataset:String):
    import dask.array as da
    from survos2.improc.utils import optimal_chunksize
    ws = get(workspace)
    with dataset_from_uri(dataset, mode='r') as data:
        chunk_size = optimal_chunksize(data, Config['computing.chunk_size'])
        data = da.from_array(data, chunks=chunk_size)
        data -= da.min(data)
        data /= da.max(data)
        ds = ws.add_data(data)
    logger.info(type(ds))
    return ds


@hug.get()
@hug.local()
def get_data(workspace:String):
    workspace, session = parse_workspace(workspace)
    return get(workspace).get_data()

### Sessions

@hug.get()
def list_sessions(workspace:String):
    return get(workspace).available_sessions()


@hug.get()
def add_session(workspace:String, session:String):
    return get(workspace).add_session(session)


@hug.get()
def delete_session(workspace:String, session:String):
    get(workspace).remove_session(session)
    return dict(done=True)


@hug.get()
def get_session(workspace:String, session:String):
    return get(workspace).get_session(session)

### Datasets

@hug.get()
def list_datasets(workspace:String):
    workspace, session = parse_workspace(workspace)
    return get(workspace).available_datasets(session)


@hug.get()
def add_dataset(workspace:String, dataset:String, dtype:String,
                fillvalue:Int=0, group:String=None,
                chunks:IntOrNone=None):
    workspace, session = parse_workspace(workspace)
    if group:
        dataset = '{}/{}'.format(group, dataset)
    return get(workspace).add_dataset(dataset, dtype, session=session,
                                      fillvalue=fillvalue, chunks=chunks)


@hug.get()
@hug.local()
def delete_dataset(workspace:String, dataset:String, group:String=None):
    workspace, session = parse_workspace(workspace)
    if group:
        dataset = '{}/{}'.format(group, dataset)
    get(workspace).remove_dataset(dataset, session=session)
    return dict(done=True)


@hug.get()
@hug.local()
def get_dataset(workspace:String, dataset:String, group:String=None):
    workspace, session = parse_workspace(workspace)
    if group:
        dataset = '{}/{}'.format(group, dataset)
    return get(workspace).get_dataset(dataset, session=session)

### Get Metadata

@hug.get()
@hug.local()
def metadata(workspace:String, dataset:String=None):
    workspace, session = parse_workspace(workspace)
    if dataset:
        ds = get_dataset(workspace, dataset, session=session)
    else:
        ds = get_data(workspace)
    return ds.get_metadata()

### Local utils for plugins

@hug.local()
def existing_datasets(workspace:String, group:String=None, filter:String=None):
    ws, session = parse_workspace(workspace)
    ws = get(ws)
    all_ds = sorted(ws.available_datasets(session=session, group=group))
    result = {}
    for dsname in all_ds:
        if filter is None or filter in dsname:
            ds = ws.get_dataset(dsname, session=session)
            result[dsname.split(op.sep)[1]] = ds
    return result


@hug.local()
def auto_create_dataset(workspace:String, name:String, group:String,
                        dtype:String, fill:Int=0, chunks:IntOrNone=None):
    all_ds = existing_datasets(workspace, group)
    max_idx = 0
    pattern = '{:03d}_{}'
    for dsname in all_ds:
        idx = parse.parse(pattern, dsname)
        if idx:
            max_idx = max(max_idx, idx[0])
    dataset_id = pattern.format(max_idx + 1, name)
    dataset_name = dataset_id.replace('_', ' ').title()
    dataset_file = '{}/{}'.format(group, dataset_id)
    ds = add_dataset(workspace, dataset_file, dtype, fillvalue=fill, chunks=chunks)
    ds.set_attr('name', dataset_name)
    return ds


@hug.local()
def rename_dataset(workspace:String, feature_id:String, group:String, new_name:String):
    ds = get_dataset(workspace, feature_id, group=group)
    ds.set_attr('name', new_name)


@hug.local()
def parse_workspace(workspace:String):
    if '@' in workspace:
        session, workspace = workspace.split('@')
        return workspace, session
    else:
        return workspace, 'default'

