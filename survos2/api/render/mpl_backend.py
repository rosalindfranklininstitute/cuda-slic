

import numpy as np

from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, cm, colors

from survos2.api.render.backend import Renderer, Layer


BACKEND_NAME = 'Matplotlib'


_cmaps = [
    ('Default',
        ['gray']
    ),
    ('Perceptually Uniform Sequential', [
        'viridis', 'plasma', 'inferno', 'magma'
    ]),
    ('Sequential', [
        'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
    ]),
    ('Sequential (2)', [
        'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
        'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
        'hot', 'afmhot', 'gist_heat', 'copper'
    ]),
    ('Diverging', [
        'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
        'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
    ]),
    ('Qualitative', [
        'Pastel1', 'Pastel2', 'Paired', 'Accent',
        'Dark2', 'Set1', 'Set2', 'Set3',
        'tab10', 'tab20', 'tab20b', 'tab20c'
    ]),
    ('Miscellaneous', [
        'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
        'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
        'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'
    ])
]

MplCmaps = OrderedDict()
for entry in _cmaps:
    MplCmaps[entry[0]] = OrderedDict()
    for cmap in entry[1]:
        MplCmaps[entry[0]][cmap] = cmap


class MplLayer(Layer):

    def __init__(self, renderer, data, cmap='viridis', clim=(0, 1),
                 interp='nearest', alpha=100, order=1):
        super().__init__(renderer, data, cmap=cmap, clim=clim,
                         interp=interp, alpha=alpha, order=order)
        self.image = renderer.axes.imshow(self._data, cmap=self._cmap,
                                          alpha=self._alpha / 100.,
                                          vmin=self._clim[0], vmax=self._clim[1],
                                          interpolation=self._interp,
                                          zorder=self._order)
        self._rescale()

    def _convert_cmap(self, cmap):
        if type(cmap) == list:
            cmap = colors.ListedColormap(cmap)
        return cmap

    def set_order(self, order):
        super().set_order(order)
        self.image.set_zorder(order)

    def set_visible(self, flag):
        super().set_visible(flag)
        self.image.set_visible(flag)

    def update_image(self, data, **kwargs):
        self.image.set_data(data)
        if 'clim' in kwargs:
            self.image.set_clim(*kwargs['clim'])
        if 'cmap' in kwargs:
            cmap = self._convert_cmap(kwargs['cmap'])
            self.image.set_cmap(cmap)
        if 'interp' in kwargs:
            self.image.set_interpolation(kwargs['interp'])
        if 'alpha' in kwargs:
            self.image.set_alpha(kwargs['alpha'] / 100.)
        if 'order' in kwargs:
            self.set_order(kwargs['order'])
        super().update_image(data, **kwargs)

    def _rescale(self):
        pass

    def draw(self):
        pass


class MplRenderer(Renderer):

    def __init__(self, size=(512, 512), save_image=False, save_png=False,
                 compression=0, layer_cls=MplLayer, **kwargs):
        super().__init__(size=size, save_image=save_image,
                         save_png=save_png, compression=compression,
                         layer_cls=layer_cls)
        self.fig, self.axes = plt.subplots(ncols=1, dpi=100, nrows=1, frameon=False)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.axes.set_yticks([])
        self.axes.set_xticks([])
        self.resize(size)

    def resize(self, size):
        self._size = size
        dpi = self.fig.get_dpi()
        self.fig.set_size_inches(size[0] / dpi, size[1] / dpi)

    @property
    def data_size(self):
        return self._data_size

    @data_size.setter
    def data_size(self, size):
        self._data_size = size
        self.resize(size)

    def _draw_layers(self, layers=None):
        self.fig.canvas.draw()
        data = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self._image = data
        if self._save_png:
            self._png = _make_png(self.screenshot, self._compression)

    @staticmethod
    def label_clim():
        return (0, 16)
