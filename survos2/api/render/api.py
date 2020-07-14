

import hug

from io import StringIO, BytesIO

import numpy as np

from collections import OrderedDict

from survos2.utils import get_logger, encode_numpy
from survos2.config import Config
from survos2.api.utils import APIException
from survos2.api.types import Int, String, IntList, SmartBoolean, IntListOrNone, \
    FloatList, wrap_output_type
from survos2.api import workspace as ws
from survos2.api.annotations import get_labels

from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries


logger = get_logger()


if Config['api.renderer'] in ['matplotlib', 'mpl']:
    from survos2.api.render.mpl_backend import MplRenderer, MplCmaps, BACKEND_NAME
    _Renderer = MplRenderer
    _Cmaps = MplCmaps
else:
    from survos2.api.render.vispy_backend import VispyRenderer, VispyCmaps, BACKEND_NAME
    _Renderer = VispyRenderer
    _Cmaps = VispyCmaps

logger.info('Selecting {} rendering backend.'.format(BACKEND_NAME))


def get_cmap(cmap):
    for k in _Cmaps:
        if cmap in _Cmaps[k]:
            return _Cmaps[k][cmap]
    return str(cmap)


class LayerParams(hug.types.Type):

    def __call__(self, value):
        if value is None:
            return  None
        try:
            return (str(value[0]), get_cmap(value[1]), int(value[2]), bool(value[3]))
        except Exception:
            raise APIException('Invalid layer params provided.')

LayerParams = wrap_output_type(LayerParams())


def _boundary_transform(im, skeletonize=False):
    boundaries = find_boundaries(im)
    if skeletonize:
        boundaries = skeletonize(boundaries)
    return boundaries.astype(np.uint8)


def _transparent_cmap(color):
    return [(0, 0, 0, 0), color]


def _label_cmap(all_labels, labels):
    colors = [(0,0,0,0)] * 16
    for label in labels:
        colors[label] = all_labels[label]['color']
    return colors


KNOWN_LAYERS = [
    ('data', None, None, None),
    ('feature', None, None, None),
    ('regions', _boundary_transform, _transparent_cmap, (0, 1))
]


@hug.get()
def cmaps():
    return {k: v.keys() for k, v in _Cmaps.items()}


@hug.get()
def render_workspace(request,
                     slice_idx:Int, workspace:String,
                     max_size:IntListOrNone=None, binning:Int=1,
                     clim:FloatList=[0,1], png:SmartBoolean=False,
                     **layers):
    """
    """
    database = request.context['session']
    if 'workspace_renderer' in database:
        renderer = database['workspace_renderer']
    else:
        renderer = _Renderer()
        database['workspace_renderer'] = renderer

    renderer.clear()

    for i, (layer, data_tr, cmap_tr, clim_tr) in enumerate(KNOWN_LAYERS):
        if not layer in layers:
            continue
        logger.info('Rendering layer {}: {}'.format(layer, slice_idx))
        dsname, cmap, alpha, visible = LayerParams(layers.pop(layer))
        if dsname == '__data__':
            data = ws.get_data(workspace)
        else:
            data = ws.get_dataset(workspace, dsname)
        data = data[slice_idx]
        renderer.data_size = data.shape
        if data_tr:
            data = data_tr(data)
        if cmap and cmap_tr:
            cmap = cmap_tr(cmap)
        if clim and clim_tr:
            if type(clim_tr) in [tuple, list]:
                clim = clim_tr
            else:
                clim = clim_tr(data, clim)
        renderer.update_layer(layer, dsname, data, cmap=cmap,
                              clim=clim, visible=visible,
                              alpha=alpha, interp='nearest',
                              order=i+1)

    if 'annotations' in layers:
        n = len(KNOWN_LAYERS)
        levels, _, alpha, visible = layers.pop('annotations')
        for i, level in enumerate(levels):
            level, labels = level
            data = ws.get_dataset(workspace, level)
            data = data[slice_idx] & 15
            all_labels = get_labels(workspace, level, full=True)
            cmap = _label_cmap(all_labels, labels)
            clim = _Renderer.label_clim()
            renderer.update_layer(level, level, data, cmap=cmap,
                                  clim=clim, visible=visible,
                                  alpha=alpha, interp='nearest',
                                  order=i+n+1)

    renderer.render_workspace(max_size=max_size, binning=binning)

    if png:
        image = renderer.png.copy()
    else:
        image = renderer.image.copy()

    return encode_numpy(image)

