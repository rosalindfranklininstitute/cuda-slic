

import hug
import os.path as op

from survos2.api import workspace as ws
from survos2.api.utils import get_function_api, save_metadata, dataset_repr
from survos2.api.types import DataURI, String, Int, Float, FloatOrVector, \
    SmartBoolean

from survos2.io import dataset_from_uri
from survos2.utils import get_logger
from survos2.improc import map_blocks


__feature_group__ = 'features'
__feature_dtype__ = 'float32'
__feature_fill__ = 0


logger = get_logger()


@hug.get()
@save_metadata
def total_variation(src:DataURI, dst:DataURI, lamda:Float=10,
                    max_iter:Int=100) -> 'Denoising':
    """
    API wrapper around `survos2.improc.features.tv.tvdenoising3d`.
    """
    from ..improc.features.tv import tvdenoising3d
    map_blocks(tvdenoising3d, src, out=dst, lamda=lamda, max_iter=max_iter,
               normalize=True)


@hug.get()
@save_metadata
def gaussian(src:DataURI, dst:DataURI, sigma:FloatOrVector=1) -> 'Denoising':
    """
    API wrapper around `survos2.improc.features.gauss.gaussian`.
    """
    from ..improc.features.gauss import gaussian
    map_blocks(gaussian, src, out=dst, sigma=sigma, normalize=True)


@hug.get()
@save_metadata
def gaussian_center(src:DataURI, dst:DataURI, sigma:FloatOrVector=1):
    """
    API wrapper around `survos2.improc.features.gauss.gaussian_center`.
    """
    from ..improc.features.gauss import gaussian_center
    map_blocks(gaussian_center, src, out=dst, sigma=sigma, normalize=True)


@hug.get()
@save_metadata
def gaussian_norm(src:DataURI, dst:DataURI, sigma:FloatOrVector=1):
    """
    API wrapper around `survos2.improc.features.gauss.gaussian_norm`.
    """
    from ..improc.features.gauss import gaussian_norm
    map_blocks(gaussian_norm, src, out=dst, sigma=sigma, normalize=True)


@hug.get()
def create(workspace:String, feature_type:String):
    ds = ws.auto_create_dataset(workspace, feature_type, __feature_group__,
                                __feature_dtype__, fill=__feature_fill__)
    ds.set_attr('kind', feature_type)
    return dataset_repr(ds)


@hug.get()
@hug.local()
def existing(workspace:String, full:SmartBoolean=False, filter:SmartBoolean=True):
    datasets = ws.existing_datasets(workspace, group=__feature_group__)
    if full:
        datasets = {'{}/{}'.format(__feature_group__, k): dataset_repr(v)
                    for k, v in datasets.items()}
    else:
        datasets = {k: dataset_repr(v) for k, v in datasets.items()}
    if filter:
        datasets = {k: v for k, v in datasets.items() if v['kind'] != 'unknown'}
    return datasets


@hug.get()
def remove(workspace:String, feature_id:String):
    ws.delete_dataset(workspace, feature_id, group=__feature_group__)


@hug.get()
def rename(workspace:String, feature_id:String, new_name:String):
    ws.rename_dataset(workspace, feature_id, __feature_group__, new_name)


@hug.get()
def group():
    return __feature_group__


@hug.get()
def available():
    h = hug.API(__name__)
    all_features = []
    for name, method in h.http.routes[''].items():
        if name[1:] in ['available', 'create', 'existing', 'remove', 'rename', 'group']:
            continue
        name = name[1:]
        func = method['GET'][None].interface.spec
        desc = get_function_api(func)
        category = desc['returns'] or 'Others'
        desc = dict(name=name, params=desc['params'], category=category)
        all_features.append(desc)
    return all_features
