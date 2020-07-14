

import numpy as np
from skimage.draw import line
from skimage.morphology import disk
from scipy.ndimage import binary_dilation
from matplotlib.colors import ListedColormap

from survos2.utils import decode_numpy

from survos2.ui.qt.plugins.regions import RegionComboBox
from survos2.ui.qt.plugins.base import register_plugin, Plugin, Tool, ViewerExtension

from survos2.ui.qt.components import *
from survos2.ui.qt.qtcompat import QtWidgets, QtCore, QtGui
from survos2.ui.qt.control import Launcher, DataModel



_AnnotationNotifier = PluginNotifier()


def _fill_annotations(combo, full=False):
    params = dict(workspace=True, full=full)
    levels = Launcher.g.run('annotations', 'get_levels', **params)
    if levels:
        for level in levels:
            combo.addCategory(level['name'])
            if not 'labels' in level:
                continue
            for label in level['labels'].values():
                icon = FAIcon('fa.square', color=label['color'])
                lidx = '{}:{}'.format(level['id'], label['idx'])
                data = dict(level=level['id'], idx=label['idx'], color=label['color'])
                combo.addItem(lidx, label['name'], icon=icon, data=data)


class AnnotationComboBox(LazyComboBox):

    def __init__(self, full=False, parent=None):
        self.full = full
        super().__init__(parent=parent, header=(None, 'Select Label'))
        _AnnotationNotifier.listen(self.update)

    def fill(self):
        _fill_annotations(self, full=self.full)


class MultiAnnotationComboBox(LazyMultiComboBox):

    def __init__(self, full=False, parent=None):
        self.full = full
        super().__init__(parent=parent, text='Select Label', groupby=('level', 'idx'))
        _AnnotationNotifier.listen(self.update)

    def fill(self):
        _fill_annotations(self, full=self.full)


class Annotator(ViewerExtension):

    annotated = QtCore.pyqtSignal(tuple)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = (1, 0, 0, 1)
        self.color_overlay = (1, 0, 0, 0.5)

        self.annotating = False

        self.line = None
        self.line_width = 1

        self.region = None
        self.region_mask = None
        self.cmap = None

        self.current = dict()
        self.all_regions = set()

    def install(self, fig, axes):
        super().install(fig, axes)
        self.connect('button_press_event', self.draw_press)
        self.connect('button_release_event', self.draw_release)
        self.connect('motion_notify_event', self.draw_motion)
        self.data_size = DataModel.g.current_workspace_shape[1:]
        self.line_mask = np.zeros(self.data_size, np.bool)
        cmap = ListedColormap([(0,0,0,0)] * 2)
        self.line_mask_ax = self.axes.imshow(self.line_mask, cmap=cmap, vmin=0, vmax=1)
        if self.region is not None:
            self.region_mask = self.axes.imshow(np.empty(self.data_size, np.uint8))
            self.initialize_region()

    def disable(self):
        super().disable()
        if self.line_mask_ax:
            self.line_mask_ax.remove()
            self.line_mask_ax = None
            self.line_mask = None
        if self.region_mask:
            self.region_mask.remove()
            self.region_mask = None

    def set_region(self, region):
        self.region = region
        if self.axes is None:
            return
        if self.region is None and self.region_mask is not None:
            self.region_mask.remove()
            self.region_mask = None
        if region is None:
            return
        if self.region_mask is None:
            self.region_mask = self.axes.imshow(np.empty(self.data_size, np.uint8))
        if region is not None:
            self.initialize_region()
        self.redraw()

    def initialize_region(self):
        n = self.region.max()
        self.region_mask.set_data(self.region)
        self.region_mask.set_clim(0, n)
        self.region_mask.set_alpha(1)
        self.initialize_region_cmap()

    def initialize_region_cmap(self):
        n = self.region.max() + 1
        cmap = [(0, 0, 0, 0)] * n
        self.cmap = ListedColormap(cmap)
        self.region_mask.set_cmap(self.cmap)

    def set_color(self, color):
        self.color = QtGui.QColor(color).getRgbF()
        self.color_overlay = self.color[:3] + (0.5,)
        self.line_mask_ax.set_cmap(ListedColormap([(0,0,0,0), self.color]))

    def set_linewidth(self, line_width):
        self.line_width = line_width

    def draw_press(self, evt):
        if not evt.inaxes == self.axes or evt.button != 1:
            return
        if self.line is not None:
            self.draw_release(evt)

        self.annotating = True

        y = int(evt.ydata)
        x = int(evt.xdata)
        self.current = dict(y=[y], x=[x])
        self.line_mask[y, x] = True

        if self.region is not None and self.cmap is not None:
            if hasattr(self.cmap, '_lut'):
                sv = self.region[y, x]
                self.all_regions |= set([sv])
                self.cmap._lut[sv] = self.color_overlay
                self.region_mask.set_cmap(self.cmap)

        self.fig.redraw()

    def draw_motion(self, evt):
        if not evt.inaxes == self.axes or not self.annotating:
            return
        y = int(evt.ydata)
        x = int(evt.xdata)
        py = self.current['y'][-1]
        px = self.current['x'][-1]
        yy, xx = line(py, px, y, x)
        self.current['y'].append(y)
        self.current['x'].append(x)

        if self.line_width > 1:
            yy, xx = self.dilate_annotations(yy, xx)

        self.line_mask[yy, xx] = True
        self.line_mask_ax.set_data(self.line_mask)

        if self.region is not None and self.cmap is not None:
            svs = set(self.region[yy, xx])
            self.all_regions |= svs
            modified = False
            if hasattr(self.cmap, '_lut'):
                for sv in svs:
                    if self.cmap._lut[sv][-1] == 0:
                        self.cmap._lut[sv] = self.color_overlay
                        modified = True
            if modified:
                self.region_mask.set_cmap(self.cmap)
        self.fig.redraw()

    def draw_release(self, evt):
        if not self.annotating:
            return
        self.annotating = False
        if self.region is None:
            annotations = np.where(self.line_mask)
        else:
            annotations = tuple(self.all_regions)
        # Clear Line
        self.current.clear()
        self.line_mask[:] = False
        self.line_mask_ax.set_data(self.line_mask)
        # Clear regions
        if self.region is not None:
            self.initialize_region_cmap()
            self.all_regions = set()
        # Render clean
        self.fig.redraw()
        # Send result
        self.annotated.emit(annotations)

    def dilate_annotations(self, yy, xx):
        data = np.zeros_like(self.line_mask)
        data[yy, xx] = True

        r = np.ceil(self.line_width / 2)
        ymin = int(max(0, yy.min() - r))
        ymax = int(min(data.shape[0], yy.max()+1+r))
        xmin = int(max(0, xx.min() - r))
        xmax = int(min(data.shape[1], xx.max()+1+r))
        mask = data[ymin:ymax, xmin:xmax]

        mask = binary_dilation(mask, disk(self.line_width / 2).astype(np.bool))
        yy, xx = np.where(mask)
        yy += ymin
        xx += xmin
        return yy, xx


class AnnotationTool(Tool):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        hbox = HBox(self, margin=7, spacing=5)
        self.label = AnnotationComboBox()
        self.region = RegionComboBox(header=(None, 'Voxels'), full=True)
        self.width = Slider(vmin=1, vmax=30)
        hbox.addWidget(self.label)
        hbox.addWidget(self.region)
        hbox.addWidget(self.width)
        hbox.addWidget(None, 1)
        self.label.currentIndexChanged.connect(self.selection_changed)
        self.region.currentIndexChanged.connect(self.selection_changed)
        self.width.valueChanged.connect(self.selection_changed)

        self.selection = dict(label=None, color='#00000000', region=None, width=1)
        self.annotator = Annotator(enabled=False)
        self.annotator.annotated.connect(self.on_annotated)

    def setEnabled(self, flag):
        super().setEnabled(flag)
        if flag:
            self.viewer.install_extension(self.annotator)
            self.annotator.set_color(self.selection['color'])
            self.annotator.set_linewidth(self.selection['width'])
        else:
            self.annotator.disable()

    def selection_changed(self):
        label, region, line_width = self.value()
        color = '#00000000' if label is None else label['color']
        level = 'annotations/' + label['level'] if label else None
        self.selection = dict(label=label, region=region, color=color,
                              width=line_width)
        self.slice_updated(self.current_idx)
        self.annotator.setEnabled(label is not None)
        if label:
            self.annotator.set_color(color)
            self.annotator.set_linewidth(line_width)
            self.viewer.show_layer('annotations', level)

    def value(self):
        return (self.label.value(), self.region.value(), self.width.value())

    def slice_updated(self, idx):
        super().slice_updated(idx)
        region = self.selection['region']
        if region:
            region = DataModel.g.dataset_uri(region)
            params = dict(workpace=True, src=region, slice_idx=idx)
            result = Launcher.g.run('regions', 'get_slice', **params)
            if result:
                region = decode_numpy(result)
                self.annotator.set_region(region)
        else:
            self.annotator.set_region(None)

    def on_annotated(self, points):
        level = self.selection['label']['level']
        label = self.selection['label']['idx']
        params = dict(workspace=True, level=level, label=label)
        if self.selection['region'] is None:
            yy, xx = points
            idx = self.current_idx
            params.update(slice_idx=idx, yy=yy.tolist(), xx=xx.tolist())
            result = Launcher.g.run('annotations', 'annotate_voxels', **params)
        else:
            region = DataModel.g.dataset_uri(self.selection['region'])
            params.update(region=region, r=list(map(int, points)), modal=True)
            result = Launcher.g.run('annotations', 'annotate_regions', **params)
        if result:
            self.viewer.update()

    def triggerKeybinding(self, key, modifiers):
        if self.selection is None or self.selection['label'] is None:
            return
        result = None
        level = self.selection['label']['level']
        params = dict(workspace=True, level=level)
        if modifiers == QtCore.Qt.ControlModifier and key == QtCore.Qt.Key_Z:
            result = Launcher.g.run('annotations', 'annotate_undo', **params)
            if result:
                self.viewer.update()


@register_plugin
class AnnotationPlugin(Plugin):

    __icon__ = 'fa.pencil'
    __pname__ = 'annotations'
    __views__ = ['slice_viewer']

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.btn_addlevel = IconButton('fa.plus', 'Add Level', accent=True)
        self.vbox = VBox(self, spacing=10)
        self.vbox.addWidget(self.btn_addlevel)
        self.levels = {}
        self.btn_addlevel.clicked.connect(self.add_level)
        self.annotation_tool = AnnotationTool()

    def on_created(self):
        self['slice_viewer'].add_tool('Annotations', self.__icon__,
                                      tool=self.annotation_tool)

    def add_level(self):
        level = Launcher.g.run('annotations', 'add_level', workspace=True)
        if level:
            self._add_level_widget(level)
            _AnnotationNotifier.notify()

    def _add_level_widget(self, level):
        widget = AnnotationLevel(level)
        widget.removed.connect(self.remove_level)
        self.vbox.addWidget(widget)
        self.levels[level['id']] = widget

    def uncheck_labels(self):
        for button in self.btn_group.buttons():
            selected = self.btn_group.checkedButton()
            button.setChecked(button is selected)

    def remove_level(self, level):
        if level in self.levels:
            self.levels.pop(level).setParent(None)

    def setup(self):
        result = Launcher.g.run('annotations', 'get_levels', workspace=True)
        if not result:
            return
        # Remove levels that no longer exist in the server
        rlevels = [r['id'] for r in result]
        for level in list(self.levels.keys()):
            if level not in rlevels:
                self.remove_level(level)
        # Populate with new levels if any
        for level in result:
            if level['id'] not in self.levels:
                self._add_level_widget(level)


class AnnotationLevel(Card):

    removed = QtCore.pyqtSignal(str)

    def __init__(self, level, parent=None):
        super().__init__(level['name'], editable=True, collapsible=True,
                         removable=True, addbtn=True, parent=parent)
        self.level_id = level['id']
        self.le_title = LineEdit(level['name'])
        self.le_title.setProperty('header', True)
        self.labels = {}
        self._populate_labels()

    def card_title_edited(self, title):
        params = dict(level=self.level_id, name=title, workspace=True)
        return Launcher.g.run('annotations', 'rename_level', **params)

    def card_add_item(self):
        params = dict(level=self.level_id, workspace=True)
        result = Launcher.g.run('annotations', 'add_label', **params)
        if result:
            self._add_label_widget(result)
            _AnnotationNotifier.notify()

    def card_deleted(self):
        params = dict(level=self.level_id, workspace=True)
        result = Launcher.g.run('annotations', 'delete_level', **params)
        if result:
            self.removed.emit(self.level_id)
            _AnnotationNotifier.notify()

    def remove_label(self, idx):
        if idx in self.labels:
            self.labels.pop(idx).setParent(None)
            self.update_height()

    def _add_label_widget(self, label):
        widget = AnnotationLabel(label, self.level_id)
        widget.removed.connect(self.remove_label)
        self.add_row(widget)
        self.labels[label['idx']] = widget
        self.expand()

    def _populate_labels(self):
        params = dict(level=self.level_id, workspace=True)
        result = Launcher.g.run('annotations', 'get_labels', **params)
        if result:
            for k, label in result.items():
                if k not in self.labels:
                    self._add_label_widget(label)


class AnnotationLabel(QCSWidget):

    __height__ = 30

    removed = QtCore.pyqtSignal(int)

    def __init__(self, label, level_dataset, parent=None):
        super().__init__(parent=parent)
        self.level_dataset = level_dataset
        self.label_idx = label['idx']
        self.label_color = label['color']
        self.label_name = label['name']
        self.label_visible = label['visible']

        self.btn_del = DelIconButton(secondary=True)
        self.txt_label_name = LineEdit(label['name'])
        self.btn_label_color = ColorButton(label['color'])
        self.btn_select = Spacing(35)
        self.setMinimumHeight(self.__height__)
        self.setFixedHeight(self.__height__)

        self.txt_label_name.editingFinished.connect(self.update_label)
        self.btn_label_color.colorChanged.connect(self.update_label)
        self.btn_del.clicked.connect(self.delete)

        hbox = HBox(self)
        hbox.addWidget(HWidgets(self.btn_del, self.btn_label_color,
                                self.txt_label_name, self.btn_select, stretch=2))

    def update_label(self):
        label = dict(idx=self.label_idx, name=self.txt_label_name.text(),
                     color=self.btn_label_color.color, visible=self.label_visible)
        params = dict(level=self.level_dataset, workspace=True)
        result = Launcher.g.run('annotations', 'update_label', **params, **label)
        if result:
            self.label_name = result['name']
            self.label_color = result['color']
            self.label_visible = result['visible']
            _AnnotationNotifier.notify()
        else:
            self.txt_label_name.setText(self.label_name)
            self.btn_label_color.setColor(self.label_color)

    def delete(self):
        params = dict(level=self.level_dataset, workspace=True, idx=self.label_idx)
        result = Launcher.g.run('annotations', 'delete_label', **params)
        if result:
            _AnnotationNotifier.notify()
            self.removed.emit(self.label_idx)
