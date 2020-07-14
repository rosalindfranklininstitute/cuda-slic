

import os
import numpy as np

from collections import OrderedDict

from matplotlib import cm

from survos2.utils import find_library

if find_library('libOSMesa'):
    from vispy import use
    use(app='osmesa')

from vispy import app, gloo
from vispy.io import _make_png
from vispy.visuals import ImageVisual, transforms
from vispy.visuals.filters import Alpha
from vispy.color.colormap import Colormap, ColorArray, get_colormaps as get_vispy_cmaps

from survos2.utils import get_logger
from survos2.api.render.backend import Renderer, Layer

BACKEND_NAME = 'Vispy ({})'.format(app.use_app().backend_name)

logger = get_logger()

VispyCmaps = OrderedDict()
VispyCmaps['Primary'] = dict(grays='grays')
VispyCmaps['Others'] = {k: k for k in get_vispy_cmaps() if k != 'grays'}

for cmap_name in ['viridis', 'inferno', 'magma', 'plasma']:
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, 64))[:, :3]
    cmap = Colormap(ColorArray(colors))
    VispyCmaps['Primary'][cmap_name] = cmap


class VispyLayer(Layer):

    def __init__(self, renderer, data, cmap='viridis', clim=(0, 1),
                 interp='nearest', alpha=100, order=1):
        super().__init__(renderer, data, cmap=cmap, clim=clim, interp=interp,
                         alpha=alpha, order=order)
        self.visual = ImageVisual(self._data, clim=self._clim, cmap=self._cmap,
                                  interpolation=self._interp)
        self.alphaFilter = Alpha(self._alpha / 100.)
        self.visual.attach(self.alphaFilter)
        self._rescale()

    def set_order(self, order):
        super().set_order(order)
        self.visual.transform.translate = (-1, 1, -.01 * order)

    def _convert_cmap(self, cmap):
        if type(cmap) == list:
            cmap = Colormap(ColorArray(cmap))
        return cmap

    def _rescale(self):
        height, width = self._data.shape[:2]
        vscale = -2. / height
        hscale = 2. / width
        zdepth = -0.01 * self._order
        transform = transforms.STTransform(scale=(hscale, vscale),
                                           translate=(-1, 1, zdepth))
        self.visual.transform = transform

    def update_image(self, data, **kwargs):
        self.visual.set_data(data)
        if 'clim' in kwargs:
            self.visual.clim = kwargs['clim']
        if 'cmap' in kwargs:
            cmap = self._convert_cmap(kwargs['cmap'])
            self.visual.cmap = cmap
        if 'interp' in kwargs:
            self.visual.interpolation = kwargs['interp']
        if 'alpha' in kwargs:
            self.alphaFilter.alpha = kwargs['alpha'] / 100.
        if 'order' in kwargs:
            self.set_order(kwargs['order'])
        super().update_image(data, **kwargs)

    def draw(self):
        if self.is_visible():
            self.visual.draw()


class VispyRenderer(app.Canvas, Renderer):

    def __init__(self, size=(512, 512), save_png=False, compression=0,
                 layer_cls=VispyLayer, **kwargs):
        app.Canvas.__init__(self, size=size, **kwargs)
        Renderer.__init__(self, size=size, save_png=save_png,
                          compression=compression,
                          layer_cls=layer_cls)
        self._rendertex = gloo.Texture2D(shape=self._size + (4,))
        self._fbo = gloo.FrameBuffer(self._rendertex, gloo.RenderBuffer(self._size))

    def resize(self, size):
        self._fbo.resize(size[::-1])
        self._size = size

    @property
    def data_size(self):
        return self._data_size

    @data_size.setter
    def data_size(self, size):
        self._data_size = size
        self.resize(size)

    def _draw_layers(self, layers=None):
        with self._fbo:
            gloo.clear(color=True, depth=True)
            gloo.set_viewport(0, 0, *self._size)
            gloo.set_state(depth_test=False)

            for image in self._get_layers(layers):
                if image and image.is_visible():
                    image.draw()

            self._image = gloo.read_pixels((0, 0, *self._size), True)

        if self._save_png:
            self._png = _make_png(self.screenshot, self._compression)

    @staticmethod
    def label_clim():
        return (0, 15)
