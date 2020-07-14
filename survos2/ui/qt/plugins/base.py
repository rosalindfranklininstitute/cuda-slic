

from collections import OrderedDict

from ..qtcompat import QtWidgets, QtCore
from ..components import QCSWidget


__available_plugins__ = OrderedDict()


class Plugin(QCSWidget):

    change_view = QtCore.pyqtSignal(str, dict)

    __icon__ = 'square'
    __pname__ = 'plugin'
    __title__ = None
    __views__ = []

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._loaded_views = dict()

    def show_view(self, name, **kwargs):
        if name not in self.__views__:
            raise ValueError('View `{}` was not preloaded.'.format(name))
        self.change_view.emit(name, kwargs)

    def register_view(self, view, widget):
        if not view in self.__views__:
            return
        self._loaded_views[view] = widget

    def on_created(self):
        pass

    def __getitem__(self, name):
        return self._loaded_views[name]


def register_plugin(cls):
    name = cls.__pname__
    icon = cls.__icon__
    title = cls.__title__
    views = cls.__views__

    if name in __available_plugins__:
        raise ValueError('Plugin {} already registered.'.format(name))

    if title is None:
        title = name.capitalize()

    desc = dict(cls=cls, name=name, icon=icon, title=title, views=views)
    __available_plugins__[name] = desc
    return cls


def get_plugin(name):
    if name not in __available_plugins__:
        raise ValueError('Plugin {} not registered'.format(name))
    return __available_plugins__[name]


def list_plugins():
    return list(__available_plugins__.keys())


class Tool(QCSWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._viewer = None
        self._current_idx = 0

    @property
    def viewer(self):
        return self._viewer

    @property
    def current_idx(self):
        return self._current_idx

    def setEnabled(self, flag):
        super().setEnabled(flag)
        if flag:
            self.connect()
        else:
            self.disconnect()

    def connect(self):
        if self.viewer:
            self.viewer.slice_updated.connect(self.slice_updated)

    def disconnect(self):
        if self.viewer:
            self.viewer.slice_updated.disconnect(self.slice_updated)

    def set_viewer(self, viewer):
        self.disconnect()
        self._viewer = viewer
        self.connect()

    def slice_updated(self, idx):
        self._current_idx = idx


class ViewerExtension(QtCore.QObject):

    def __init__(self, modifiers=None, enabled=True):
        super().__init__()
        self.fig = None
        self.axes = None

        self._connections = []
        self._enabled = enabled
        self._modifiers = modifiers or QtCore.Qt.NoModifier

    def isEnabled(self):
        return self._enabled

    def setEnabled(self, flag):
        self._enabled = bool(flag)

    def active(self):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        return self.isEnabled() and modifiers == self._modifiers

    def install(self, fig, axes):
        self.disconnect()
        self.fig = fig
        self.axes = axes

    def disable(self):
        self.disconnect()
        self.fig = None
        self.axes = None

    def connect(self, event, callback):
        if not self.fig:
            return
        func = lambda evt: self.active() and callback(evt)
        self._connections.append(self.fig.mpl_connect(event, func))

    def disconnect(self):
        if not self.fig:
            return
        for conn in self._connections:
            self.fig.mpl_disconnect(conn)
        self._connections.clear()

    def redraw(self):
        if self.fig:
            self.fig.redraw()