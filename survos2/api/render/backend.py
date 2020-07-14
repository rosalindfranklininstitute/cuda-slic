

from collections import OrderedDict


class Layer(object):

    def __init__(self, renderer, data, cmap='viridis', clim=(0, 1),
                 interp='nearest', alpha=100, order=1):
        self._renderer = renderer
        self._data = data
        self._cmap = self._convert_cmap(cmap)
        self._clim = clim
        self._interp = interp
        self._alpha = alpha
        self._order = order
        self._visible = True
        self._prev_params = dict()

    def _convert_cmap(self, cmap):
        raise NotImplementedError()

    def _rescale(self):
        raise NotImplementedError()

    @property
    def order(self):
        return self._order

    def set_order(self, order):
        self._order = order

    def update_image(self, data, **kwargs):
        self._data = data
        self._rescale()

    def is_visible(self):
        return self._visible

    def set_visible(self, flag):
        self._visible = flag

    def draw(self):
        raise NotImplementedError()

    def save_params(self, params):
        self._prev_params.update(params)

    def filter_params(self, params):
        params =  {k: v for k, v in params.items()
                   if k not in self._prev_params or params[k] != self._prev_params[k]}
        return params


class Renderer(object):

    def __init__(self, size=(512, 512), save_png=False, compression=0,
                 layer_cls=Layer, **kwargs):
        self._save_png = save_png
        self._compression = compression

        self._size = size
        self._data_size = size
        self._binning = 1

        self._image = None
        self._png = None

        self._layer_cls = layer_cls
        self._layers = OrderedDict()

    def add_layer(self, group, name, data, **kwargs):
        im = self._layer_cls(self, data, **kwargs)
        if group not in self._layers:
            self._layers[group] = OrderedDict()
        self._layers[group][name] = im
        return im

    def update_layer(self, group, name, data, **kwargs):
        visible = kwargs.pop('visible', True)
        if group in self._layers and name in self._layers[group]:
            layer = self._layers[group][name]
            kwargs = layer.filter_params(kwargs)
            layer.update_image(data, **kwargs)
        else:
            layer = self.add_layer(group, name, data, **kwargs)
        layer.save_params(kwargs)
        layer.set_visible(visible)

    def del_layer(self, group, name):
        self._layers[group].pop(name, None)

    def resize(self, size):
        raise NotImplementedError()

    @property
    def data_size(self):
        return self._data_size

    @data_size.setter
    def data_size(self, size):
        self._data_size = size

    @property
    def image(self):
        if self._image is not None:
            return self._image
        raise ValueError('No image available')

    @property
    def png(self):
        if self._save_png and self._png is not None:
            return self._png
        return _make_png(self.image, self._compression)

    def render_workspace(self, layers=None, max_size=None, binning=None):
        if binning is not None:
            if binning < 0:
                if max_size is not None and len(max_size) == 2:
                    mw, mh = max_size
                    w, h = self.data_size
                    bw = 1 if w <= mw else w / float(mw)
                    bh = 1 if h <= mh else h / float(mh)
                    nbinning = max(bw, bh)
                else:
                    nbinning = 1
                binning = abs(binning) * nbinning
        else:
            binning = self._binning

        if binning != 1:
            self.resize(tuple(int(s//binning) for s in self.data_size))
            self._binning = binning
        elif self._size != self.data_size:
            self.resize(self.data_size)
            self._binning = 1

        self._draw_layers(layers)

    def _draw_layers(self):
        raise NotImplementedError()

    def clear(self):
        for layer in self._get_all_layers():
            layer.set_visible(False)

    def _get_layers(self, layers=None):
        if layers is None:
            return self._get_all_layers()
        layers = [self._layers[g].get(n, None) for g, n in layers]
        return self._order_layers(layers)

    def _order_layers(self, layers):
        return sorted(layers, key=lambda x: x.order)

    def _get_all_layers(self):
        layers = [self._layers[g][n] for g in self._layers for n in self._layers[g]]
        return self._order_layers(layers)

    @staticmethod
    def label_clim():
        raise NotImplementedError()