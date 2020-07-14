

import os.path as op

from .utils import resource
from .qtcompat import QtWidgets, QtCore, QtGui
from .plugins import list_plugins, get_plugin
from .views import list_views, get_view
from .components import *
from .modal import ModalManager
from .control import Launcher

from survos2.utils import get_logger
from survos2.config import Config


logger = get_logger()


class IconContainer(QCSWidget):

    plugin_selected = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.vbox = VBox(self)
        self.btn_group = QtWidgets.QButtonGroup()
        self.btn_group.setExclusive(True)
        self.plugins = {}

    def load_plugin(self, name, icon, title=''):
        if name in self.plugins:
            return
        btn = ToolIconButton(icon, title, color='white', size=(24, 24), checkable=True)
        self.plugins[name] = btn
        self.vbox.addWidget(btn)
        self.btn_group.addButton(btn)
        btn.toggled.connect(lambda flag: flag and self.select_plugin(name))

    def unload_plugin(self, name):
        if name in self.plugins:
            btn = self.plugins[name]
            btn.setParent(None)
            btn.clicked.disconnect()
            self.btn_group.removeButton(btn)
            del self.plugins[name]

    def select_plugin(self, name):
        if name not in self.plugins:
            return
        self.plugins[name].setChecked(True)
        self.plugin_selected.emit(name)


class PluginContainer(QCSWidget):

    view_requested = QtCore.pyqtSignal(str, dict)

    __sidebar_width__ = 350

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setMinimumWidth(self.__sidebar_width__)
        self.setMaximumWidth(self.__sidebar_width__)

        self.title = Header('Plugin')
        self.container = ScrollPane(parent=self)

        vbox = VBox(self, margin=(1, 0, 2, 0), spacing=5)
        vbox.addWidget(self.title)
        vbox.addWidget(self.container, 1)

        self.plugins = {}
        self.selected_name = None
        self.selected = None

    def load_plugin(self, name, title, cls):
        if name in self.plugins:
            return
        widget = cls()
        widget.change_view.connect(self.view_requested)
        self.plugins[name] = dict(widget=widget, title=title)
        return widget

    def unload_plugin(self, name):
        self.plugins.pop(name, None)

    def show_plugin(self, name):
        if name in self.plugins and name != self.selected_name:
            if self.selected is not None:
                self.selected['widget'].setParent(None)
            self.selected_name = name
            self.selected = self.plugins[name]
            self.title.setText(self.selected['title'])
            self.container.addWidget(self.selected['widget'], 1)
            if hasattr(self.selected['widget'], 'setup'):
                self.selected['widget'].setup()


class ViewContainer(QCSWidget):

    __empty_view__ = dict(idx=0, title=None)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.header = TabBar()
        self.container = QtWidgets.QStackedWidget()
        self.container.addWidget(QtWidgets.QWidget())

        vbox = VBox(self, margin=(1, 0, 0, 0), spacing=5)
        vbox.addWidget(self.header)
        vbox.addWidget(self.container, 1)

        self.views = {}
        self.current_view = None
        self.header.tabSelected.connect(self.select_view)

    def set_available_views(self, views):
        if not self.current_view in views:
            self.container.setCurrentIndex(0)
            self.current_view = None

        self.header.clear()
        views = [v for v in views if v in self.views]

        for view in views:
            self.header.addTab(view, self.views[view]['title'])

        if self.current_view is None:
            self.select_view(views[0])
        else:
            widget = self.views[self.current_view]['widget']
            if hasattr(widget, 'setup'):
                widget.setup()

    def select_view(self, name):
        if name in self.views:
            self.container.setCurrentIndex(self.views[name]['idx'])
            self.current_view = name
            self.header.setSelected(name)
            widget = self.views[self.current_view]['widget']
            if hasattr(widget, 'setup'):
                widget.setup()

    def load_view(self, name, title, cls):
        if name in self.views:
            return
        idx = len(self.views) + 1
        widget = cls()
        self.container.addWidget(widget)
        self.views[name] = dict(title=title, idx=idx, widget=widget)
        return widget

    def unload_view(self, name):
        pass

    def propagate_keybinding(self, evt):
        if self.current_view is not None:
            widget = self.views[self.current_view]['widget']
            if hasattr(widget, 'triggerKeybinding'):
                widget.triggerKeybinding(evt.key(), evt.modifiers())

        if not evt.isAccepted():
            evt.accept()


import time
from multiprocessing import Process


def update_ui():
    logger.info('Updating UI')
    QtCore.QCoreApplication.processEvents()
    time.sleep(0.1)


class MainWindow(QtWidgets.QMainWindow):

    resized = QtCore.pyqtSignal()

    __title__ = "SuRVoS: Super-Region Volume Segmentation workbench"

    def __init__(self, plugins=None, views=None, maximize=False, title=__title__):
        super().__init__()

        self.p = Process(target=update_ui)
        self.p.start()

        material_font = resource('iconfont', 'MaterialIcons-Regular.ttf')
        QtGui.QFontDatabase.addApplicationFont(material_font)

        qcs_path = resource('qcs', 'survos.qcs')
        if op.isfile(qcs_path):
            with open(qcs_path, 'r') as f:
                self.setStyleSheet(f.read())

        self.setWindowTitle(title)

        self._loaded_views = {}
        self._loaded_plugins = {}
        self._setup_layout()

        self.setMinimumSize(1024, 768)
        if maximize:
            self.showMaximized()
        else:
            self.show()

        ModalManager.instance(self).show_loading('Populating workspace')
        self._load_views(views)
        name = self._load_plugins(plugins)
        self.select_plugin(name)

        if Launcher.g.connected:
            ModalManager.g.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()

    def _setup_layout(self):
        container = QtWidgets.QWidget()
        self.setCentralWidget(container)

        self.iconContainer = IconContainer()
        self.pluginContainer = PluginContainer()
        self.viewContainer = ViewContainer()

        hbox = HBox(container)
        hbox.addWidget(self.iconContainer)
        hbox.addWidget(self.pluginContainer)
        hbox.addWidget(self.viewContainer, 1)

        self.iconContainer.plugin_selected.connect(self.show_plugin)
        self.pluginContainer.view_requested.connect(self.show_view)
        self.plugin2views = dict()

    def _load_plugins(self, plugins=None):
        all_plugins = list_plugins() if plugins is None else plugins
        for pname in all_plugins:
            plugin = self.load_plugin(pname)
            for view in self.plugin2views[pname]:
                if view in self._loaded_views:
                    plugin.register_view(view, self._loaded_views[view])
            plugin.on_created()
        return all_plugins[0]

    def _load_views(self, views=None):
        all_views = list_views() if views is None else views
        for vname in all_views:
            view = self.load_view(vname)
            self._loaded_views[vname] = view

    def load_plugin(self, name):
        if name in self.plugin2views:
            return
        plugin = get_plugin(name)
        name = plugin['name']
        title = plugin['title']
        plugin_cls = plugin['cls']
        self.plugin2views[name] = plugin['views']
        self.iconContainer.load_plugin(name, plugin['icon'], title)
        return self.pluginContainer.load_plugin(name, title, plugin_cls)

    def unload_plugin(self, name):
        self.iconContainer.unload_plugin(name)
        self.pluginContainer.unload_plugin(name)

    def load_view(self, name):
        view = get_view(name)
        name, cls, title = view['name'], view['cls'], view['title']
        return self.viewContainer.load_view(name, title, cls)

    def unload_view(self, name):
        self.viewContainer.unload_view(name)

    def select_plugin(self, name):
        self.iconContainer.select_plugin(name)

    def show_view(self, name, **kwargs):
        self.viewContainer.select_view(name, **kwargs)

    def show_plugin(self, name):
        if name in self.plugin2views:
            self.viewContainer.set_available_views(self.plugin2views[name])
            self.pluginContainer.show_plugin(name)

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.setFocus()
            e.accept()
        elif e.key() == ord(Config['qtui.menuKey']):
            current = self.pluginContainer.minimumWidth()
            stop = self.pluginContainer.__sidebar_width__ - current
            start = self.pluginContainer.__sidebar_width__ - stop

            group = QtCore.QParallelAnimationGroup()

            for anim in [b'maximumWidth', b'minimumWidth']:
                anim = QtCore.QPropertyAnimation(self.pluginContainer, anim)
                anim.setDuration(100)
                anim.setStartValue(start)
                anim.setEndValue(stop)
                group.addAnimation(anim)

            group.start()
            self.__anim = group
            e.accept()
        else:
            logger.debug("Propagating event: {}".format(e))
            self.viewContainer.propagate_keybinding(e)

    def closeEvent(self, event):
        """if Launcher.g.terminated or Launcher.g.connected:
            return event.accept()
        Launcher.g.terminated = True
        ModalManager.g.accept()
        QtWidgets.QApplication.processEvents()
        ModalManager.g.terminate()
        event.ignore()"""
        pass
