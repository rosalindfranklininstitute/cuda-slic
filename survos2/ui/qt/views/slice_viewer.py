

from collections import OrderedDict

from survos2.utils import decode_numpy
from survos2.ui.qt.views.viewer import Viewer
from survos2.ui.qt.views.base import register_view

from survos2.ui.qt.components import *
from survos2.ui.qt.utils import resource
from survos2.ui.qt.qtcompat import QtCore, QtWidgets, QtGui
from survos2.ui.qt.control import Launcher, DataModel

from survos2.ui.qt.plugins.features import FeatureComboBox
from survos2.ui.qt.plugins.regions import RegionComboBox
from survos2.ui.qt.plugins.annotations import MultiAnnotationComboBox


class CmapComboBox(LazyComboBox):

    def __init__(self, parent=None):
        super().__init__(select=1, parent=parent)

    def fill(self):
        result = Launcher.g.run('render', 'cmaps')
        if result:
            for category in result:
                self.addCategory(category)
                for cmap in result[category]:
                    if cmap == 'Greys':
                        cmap = 'Greys_r'
                    self.addItem(cmap)


class Layer(QCSWidget):

    updated = QtCore.pyqtSignal()

    def __init__(self, name='layer', source=None, cmap=None,
                 parent=None):
        super().__init__(parent=parent)
        self.name = name
        self.source = source or ComboBox()
        self.cmap = cmap or CmapComboBox()
        self.slider = Slider(value=100, label=False, auto_accept=False)
        self.checkbox = CheckBox(checked=True)

        hbox = HBox(self, spacing=5, margin=(5, 0, 5, 0))
        hbox.addWidget(self.source, 1)
        hbox.addWidget(self.cmap, 1)
        hbox.addWidget(self.slider)
        hbox.addWidget(self.checkbox)

        if hasattr(self.source, 'currentIndexChanged'):
            self.source.currentIndexChanged.connect(self._params_updated)
        elif hasattr(self.source, 'valueChanged'):
            self.source.valueChanged.connect(self._params_updated)
        if hasattr(self.cmap, 'currentIndexChanged'):
            self.cmap.currentIndexChanged.connect(self._params_updated)
        elif hasattr(self.cmap, 'colorChanged'):
            self.cmap.colorChanged.connect(self._params_updated)

        self.slider.setMinimumWidth(150)
        self.slider.valueChanged.connect(self._params_updated)
        self.checkbox.toggled.connect(self._params_updated)

    def value(self):
        return (self.source.value(), self.cmap.value(),
                self.slider.value(), self.checkbox.value())

    def _params_updated(self):
        self.updated.emit()

    def accept(self):
        self.slider.accept()

    def select(self, view):
        self.source.select(view)
        self.updated.emit()


class DataLayer(Layer):
    def __init__(self, name):
        super().__init__(name, Label('Raw Data'))

    def value(self):
        return ('__data__', self.cmap.value(),
                self.slider.value(), self.checkbox.value())


class FeatureLayer(Layer):
    def __init__(self, name):
        super().__init__(name, FeatureComboBox(full=True))


class RegionLayer(Layer):
    def __init__(self, name):
        region = RegionComboBox(full=True)
        color = ColorButton('#0D47A1')
        super().__init__(name, region, color)


class AnnotationsLayer(Layer):
    def __init__(self, name):
        super().__init__(name, MultiAnnotationComboBox(full=True), Spacing(0))

    def select(self, view):
        if self.source.select_prefix(view):
            self.updated.emit()


class LayerManager(QtWidgets.QMenu):

    __all_layers__ = {}

    paramsUpdated = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        vbox = VBox(self, spacing=5, margin=5)
        self.vmin = RealSlider(value=0, vmin=-1, vmax=1, auto_accept=False)
        self.vmax = RealSlider(value=1, vmin=0, vmax=1, auto_accept=False)
        vbox.addWidget(self.vmin)
        vbox.addWidget(self.vmax)
        self._layers = OrderedDict()
        for key, title, cls in self.__all_layers__:
            layer = cls(key)
            layer.updated.connect(self.on_layer_updated)
            vbox.addWidget(SubHeader(title))
            vbox.addWidget(layer)
            self._layers[key] = layer
        self.vmin.valueChanged.connect(self.on_layer_updated)
        self.vmax.valueChanged.connect(self.on_layer_updated)

    def refresh(self):
        for layer in self._layers.values():
            layer.update()
        return self

    def on_layer_updated(self):
        self.paramsUpdated.emit()

    def show_layer(self, layer, view):
        if layer in self._layers:
            self._layers[layer].select(view)

    def value(self):
        params = {k: v.value() for k, v in self._layers.items()}
        params['clim'] = (self.vmin.value(), self.vmax.value())
        return params

    def accept(self):
        self.vmin.accept()
        self.vmax.accept()
        for layer in self._layers.values():
            layer.accept()


class WorkspaceLayerManager(LayerManager):

    __all_layers__ = (
        ('data', 'Data', DataLayer),
        ('feature', 'Feature', FeatureLayer),
        ('regions', 'Regions', RegionLayer),
        ('annotations', 'Annotations', AnnotationsLayer),
        ('prediction', 'Predictions', Layer)
    )


@register_view(name='slice_viewer')
class SliceViewer(QCSWidget):

    slice_updated = QtCore.pyqtSignal(int)

    def __init__(self, layer_manager=None, parent=None):
        super().__init__(parent=parent)
        self.slider = Slider(auto_accept=False, center=True)
        self.viewer = Viewer()
        self.menu_panel = QtWidgets.QToolBar()
        self.tool_container = VBox(margin=0, spacing=0)
        self.current_tool = None

        vbox = VBox(self, margin=15, spacing=10)
        vbox.addWidget(self.slider)
        vbox.addWidget(self.viewer, 1)

        vbox2 = VBox(margin=0, spacing=0)
        vbox2.addLayout(self.tool_container)
        vbox2.addWidget(self.menu_panel)
        vbox.addLayout(vbox2)

        if layer_manager is None:
            layer_manager = WorkspaceLayerManager

        self.tools = []
        self.viz_params = None
        self.layer_manager = self.add_tool('Layer', 'fa.adjust', layer_manager)
        self.layer_manager.paramsUpdated.connect(self.params_updated)
        self.slider.valueChanged.connect(self.show_slice)
        self.slider.sliderReleased.connect(self._updated)

        self.viewer.mpl_connect('button_press_event', self.show_layer_manager)

        self.update()

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.Show:
            p = self.menu_panel
            point = QtCore.QPoint(0, -source.height())
            source.move(p.mapToGlobal(point))
        return source.event(event)

    def clear_tools(self):
        for tool in self.tools[1:]:
            if tool.menu():
                tool.menu().removeEventFilter(self)
            tool.setParent(None)
        self.tools = [self.tools[0]]

    def add_tool(self, name, icon, menu=None, tool=None):
        btn = ToolIconButton(icon, name, size=(36, 36), color='white',
                             checkable=tool is not None and menu is None)
        self.tools.append(btn)
        self.menu_panel.addWidget(btn)

        if tool:
            tool.setProperty('menu', True)
            tool.setVisible(False)
            tool.set_viewer(self)
            self.tool_container.addWidget(tool)
            btn.toggled.connect(lambda flag: self.toggle_tool(tool, flag))

        if menu:
            menu = menu(btn)
            btn.setMenu(menu)
            btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
            btn.toggled.connect(btn.showMenu)
            menu.installEventFilter(self)
            return menu

    def toggle_tool(self, tool, flag):
        tool.setEnabled(flag)
        if flag:
            self.current_tool = tool
            for tool in layout_widgets(self.tool_container):
                tool.setVisible(False)
            self.current_tool.setVisible(True)
            self.current_tool.slice_updated(self.slider.value())
        else:
            tool.setVisible(False)
            self.current_tool = None

    def update_viz_params(self):
        params = self.layer_manager.value()
        self.viz_params = {k: v for k, v in params.items() if v[0] is not None}

    def show_slice(self, idx):
        params = dict(slice_idx=idx, workspace=True, timeit=True)
        params.update(self.viz_params)
        result = Launcher.g.run('render', 'render_workspace', **params)
        if result:
            image = decode_numpy(result)
            self.viewer.update_image(image)
        self.slider.accept()

    def update(self):
        self.update_viz_params()
        self.show_slice(self.slider.value())

    def _updated(self):
        idx = self.slider.value()
        self.slice_updated.emit(idx)

    def params_updated(self):
        self.update()
        self.layer_manager.accept()

    def setup(self):
        max_depth = DataModel.g.current_workspace_shape[0]
        self.slider.setMaximum(max_depth - 1)

    def show_layer_manager(self, event):
        if event.button != 3:
            return
        self.layer_manager.show()

    def triggerKeybinding(self, key, modifiers):
        if key == QtCore.Qt.Key_H:
            self.viewer.center()
        elif self.current_tool and hasattr(self.current_tool, 'triggerKeybinding'):
            self.current_tool.triggerKeybinding(key, modifiers)

    def install_extension(self, ext):
        self.viewer.install_extension(ext)

    def show_layer(self, plugin, view):
        self.layer_manager.show_layer(plugin, view)
