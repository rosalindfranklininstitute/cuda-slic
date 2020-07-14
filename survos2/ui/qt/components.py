

import re
import os.path as op

import numpy as np
from collections import defaultdict

import qtawesome as qta

from survos2.ui.qt.utils import resource
from survos2.ui.qt.qtcompat import QtWidgets, QtCore, QtGui


__all__ = [
    'SWidget', 'HBox', 'VBox', 'IconButton', 'Header', 'HWidgets',
    'LineEdit', 'QCSWidget', 'TabBar', 'SubHeader', 'ToolIconButton',
    'ScrollPane', 'ColorButton', 'Spacing', 'Card', 'DelIconButton',
    'AddIconButton', 'ViewIconButton', 'CheckBox', 'LineEdit3D',
    'ComboBox', 'LazyComboBox', 'PushButton', 'Slider', 'Label',
    'RealSlider', 'FAIcon', 'MultiComboBox', 'LazyMultiComboBox',
    'PluginNotifier', 'layout_widgets', 'layout_items', 'clear_layout'
]

__dirname__ = op.dirname(op.realpath(__file__))


def layout_items(layout):
   return (layout.itemAt(i) for i in range(layout.count()))


def layout_widgets(layout):
   return (layout.itemAt(i).widget() for i in range(layout.count()))


def clear_layout(layout):
    for widget in layout_widgets(layout):
        widget.setParent(None)


class PluginNotifier(QtCore.QObject):
    updated = QtCore.pyqtSignal()
    def listen(self, *args, **kwargs):
        self.updated.connect(*args, **kwargs)
    def notify(self):
        self.updated.emit()


class SWidget(QtWidgets.QWidget):

    def __init__(self, class_name, parent=None):
        super().__init__(parent=parent)
        obj_name = QCSWidget.convert_name(class_name)
        self.setObjectName(obj_name)
        self.setAttribute(QtCore.Qt.WA_StyledBackground)

        file_path = resource('qcs', obj_name + '.qcs')
        if op.isfile(file_path):
            with open(file_path, 'r') as f:
                self.setStyleSheet(f.read())

    @staticmethod
    def convert_name(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class QCSWidget(SWidget):

    resized = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(self.__class__.__name__, parent=parent)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()

    def __call__(self, widget, stretch=0, connect=None):
        if self.layout() is not None:
            self.layout().addWidget(widget, stretch=stretch)
            if connect and hasattr(widget, connect[0]):
                getattr(widget, connect[0]).connect(connect[1])

    def value(self):
        return None


class HBox(QtWidgets.QHBoxLayout):

    def __init__(self, parent=None, margin=0, spacing=0,
                 align=QtCore.Qt.AlignLeft):
        super().__init__(parent)
        m = [margin] * 4 if type(margin) not in [list, tuple] else margin
        self.setSpacing(spacing)
        self.setAlignment(align)
        self.setContentsMargins(m[0], m[1], m[2], m[3])
        self.m = m

    @property
    def hmargin(self):
        return self.m[0] + self.m[2]

    @property
    def vmargin(self):
        return self.m[1] + self.m[3]

    def addWidget(self, widget, stretch=0):
        if isinstance(widget, str):
            widget = QtWidgets.QLabel(widget)
        elif widget is None:
            widget = QtWidgets.QWidget()
        super().addWidget(widget, stretch)


def FAIcon(*args, **kwargs):
    return qta.icon(*args, **kwargs)


class VBox(QtWidgets.QVBoxLayout):

    def __init__(self, parent=None, margin=0, spacing=0,
                 align=QtCore.Qt.AlignTop):
        super().__init__(parent)
        m = [margin] * 4 if type(margin) not in [list, tuple] else margin
        self.setSpacing(spacing)
        self.setAlignment(align)
        self.setContentsMargins(m[0], m[1], m[2], m[3])
        self.m = m

    @property
    def hmargin(self):
        return self.m[0] + self.m[2]

    @property
    def vmargin(self):
        return self.m[1] + self.m[3]

    def addWidget(self, widget, stretch=0):
        if isinstance(widget, str):
            widget = QtWidgets.QLabel(widget)
        elif widget is None:
            widget = QtWidgets.QWidget()
        super().addWidget(widget, stretch)


class ToolIconButton(QtWidgets.QToolButton):

    def __init__(self, icon, text='', size=None, checkable=False, parent=None, **kwargs):
        super().__init__(parent)
        self.setStyleSheet("""
            QToolButton {
                margin-left: 0px;
            }
            QToolButton::menu-indicator {
                width: 0px;
                border: none;
                image: none;
            }
        """)
        self.setText(text)
        self.setIcon(qta.icon(icon, **kwargs))
        if size is not None:
            self.setIconSize(QtCore.QSize(*size))
        self.setCheckable(checkable)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon);


class Header(QtWidgets.QLabel):

    def __init__(self, text):
        super().__init__(text)
        self.setStyleSheet("""
            QLabel {
                width: 100%;
                min-height: 30px;
                color: white;
                qproperty-alignment: AlignCenter;
                background-color: #0D47A1;
            }
            """)

class SubHeader(QtWidgets.QLabel):

    def __init__(self, text):
        super().__init__(text)
        self.setStyleSheet("""
            QLabel {
                min-height: 25px;
                color: white;
                qproperty-alignment: AlignCenter;
                background-color: #00838F;
            }
            """)


class HWidgets(QtWidgets.QWidget):

    def __init__(self, *widgets, **kwargs):
        super().__init__(kwargs.pop('parent', None))
        self.setObjectName('hwidgets')
        margin = kwargs.pop('margin', 5)
        self.setStyleSheet('margin-left: {}px; margin-right: {}px;'
                           .format(margin, margin))

        stretch = kwargs.pop('stretch', [])
        if type(stretch) not in [list, tuple]:
            stretch = [stretch]
        hbox = HBox(self)
        for i, widget in enumerate(widgets):
            if widget is None:
                widget = QtWidgets.QWidget()
                stretch.append(i)
            elif isinstance(widget, str):
                widget = QtWidgets.QLabel(widget)
            hbox.addWidget(widget, int(i in stretch))
        self.widgets = widgets

    def setStyle(self, style):
        for widget in self.widgets:
            widget.setStyleSheet(style)
        self.setStyleSheet(style)


class LineEdit(QtWidgets.QLineEdit):

    def __init__(self, text=None, default='', parse=str, fontsize=None, **kwargs):
        super().__init__(text, **kwargs)
        self.default = default
        self.parse = parse

        if text is not None:
            self.setText(text)
            self.setPlaceholderText(text)
        else:
            self.setPlaceholderText(str(default))

        if fontsize is not None:
            self.setStyleSheet('font-size: {}px;'.format(fontsize))
        self.returnPressed.connect(self.clearFocus)
        self.editingFinished.connect(self.parseInput)

    def value(self):
        val = self.parse(self.default)
        try:
            val = self.parse(self.text())
        except Exception:
            pass
        return val

    def parseInput(self):
        self.setText(str(self.value()))

    def setDefault(self, val):
        self.default = val
        self.setPlaceholderText(str(val))

    def setValue(self, value):
        self.setText(str(value))


class TabBar(QCSWidget):

    tabSelected = QtCore.pyqtSignal(str)

    def __init__(self, spacing=10, parent=None):
        super().__init__(parent=parent)
        self.hbox = HBox(self, spacing=spacing, align=QtCore.Qt.AlignHCenter)
        self.tabs = dict()
        self.tab_group = QtWidgets.QButtonGroup()
        self.tab_group.setExclusive(True)

    def addTab(self, name, title):
        if name in self.tabs:
            return
        btn = QtWidgets.QPushButton(title)
        btn.setCheckable(True)
        self.hbox.addWidget(btn)
        self.tabs[name] = btn
        self.tab_group.addButton(btn)
        btn.toggled.connect(lambda t: t and self.tabSelected.emit(name))

    def setSelected(self, name):
        if name in self.tabs:
            self.tabs[name].setChecked(True)

    def clear(self):
        for tab in self.tabs.values():
            tab.setParent(None)
            self.tab_group.removeButton(tab)
        self.tabs.clear()


class PushButton(QtWidgets.QPushButton):

    def __init__(self, *args, accent=False, flat=False, **kwargs):
        super().__init__(*args, **kwargs)
        if accent:
            self.setProperty('accent', True)
        if flat:
            self.setFlat(True)


class IconButton(QtWidgets.QPushButton):

    def __init__(self, icon, title='', checkable=False,
                 accent=False, flat=False, error=False,
                 parent=None, toggleable=False):
        super().__init__(title, parent=parent)
        self.icon = icon
        self.altIcon = icon
        self.flat = flat
        self.accent = accent
        self.error = error
        self.toggeable = toggleable
        if accent:
            self.setProperty('accent', True)
        self.setFlat(flat)
        self.setIconColor('white')
        self.setCheckable(checkable)
        self.leaveEvent()
        if toggleable and flat:
            self.setStyleSheet('background-color: transparent;')

    def setIconColor(self, color):
        icon = qta.icon(self.icon, color=color)
        self.setIcon(icon)

    def enterEvent(self, evt=None):
        if self.flat and not self.isChecked():
            color = '#4527A0' if self.accent else '#009688'
            self.setIconColor(color)

    def leaveEvent(self, evt=None):
        if self.flat and (not self.isChecked() or self.toggeable):
            if self.error:
                color = 'red'
            else:
                color = '#009688' if self.accent else '#0D47A1'
            self.setIconColor(color)
        elif self.flat:
            self.setIconColor('white')

    def setChecked(self, flag):
        self.leaveEvent()
        super().setChecked(flag)


class ScrollPane(QtWidgets.QScrollArea):

    def __init__(self, margin=5, parent=None):
        super().__init__(parent=parent)
        container = QtWidgets.QWidget(parent=parent)
        self.setWidget(container)
        self.setWidgetResizable(True)

        self.layout = VBox(container, margin=margin)
        self.layout.setAlignment(QtCore.Qt.AlignTop)

    def addWidget(self, widget, stretch=0):
        self.layout.addWidget(widget, stretch)


class ColorButton(QtWidgets.QPushButton):

    colorChanged = QtCore.pyqtSignal(str)

    def __init__(self, color='#000000', clickable=True, **kwargs):
        super().__init__(**kwargs)
        self.setColor(color)
        if clickable:
            self.clicked.connect(self.on_click)

    def setColor(self, color):
        color = str(QtGui.QColor(color).name())
        if color is None:
            self.setStyleSheet("""
                QPushButton, QPushButton:hover {
                    background-color:
                        qlineargradient(
                            x1:0, y1:0, x2:1, y2:1,
                            stop: 0 white, stop: 0.15 white,
                            stop: 0.2 red,
                            stop: 0.25 white, stop: 0.45 white,
                            stop: 0.5 red,
                            stop: 0.55 white, stop: 0.75 white,
                            stop: 0.8 red, stop: 0.85 white
                        );
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton { background-color: %s; }
                QPushButton:hover {
                    background-color:
                        qlineargradient(
                            x1:0, y1:0, x2:0.5, y2:1,
                            stop: 0 white, stop: 1 %s
                        );
                    }
            """ % (color, color))

        self.color = color

    def on_click(self):
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.color), self.parent())
        if not c.isValid():
            return
        self.setColor(str(c.name()))
        self.colorChanged.emit(self.color)

    def value(self):
        return self.color


class Card(QCSWidget):

    def __init__(self, title=None, removable=False, editable=False,
                 collapsible=False, addbtn=False, parent=None):
        super().__init__(parent=parent)
        self.setProperty('card', True)
        self.spacing = 5
        self.vbox = VBox(self, margin=5, spacing=self.spacing)
        self.total_height = 0
        self.prev_title = title

        if removable:
            btn_del = DelIconButton()
            btn_del.clicked.connect(self.card_deleted)
        else:
            btn_del = Spacing(35)

        if title:
            if editable:
                self.txt_title = LineEdit(title, parse=str)
                self.txt_title.editingFinished.connect(self._title_edited)
            else:
                self.txt_title = Label(title)
        else:
            self.txt_title = None

        header = [btn_del, self.txt_title]

        if addbtn:
            btn_add = AddIconButton()
            btn_add.clicked.connect(self.card_add_item)
            header.append(btn_add)

        if collapsible:
            self.btn_show = ViewIconButton()
            self.btn_show.toggled.connect(self.card_toggled)
            header.append(self.btn_show)
        else:
            self.btn_show = None

        if not addbtn and not collapsible:
            header.append(Spacing(35))

        self.add_row(HWidgets(*header, stretch=1), header=True)
        self._visible = True

    def add_row(self, widget, max_height=30, header=False):
        widget.setProperty('header', header)
        widget.setMaximumHeight(max_height)
        self.vbox.addWidget(widget)
        self.total_height += max_height + self.spacing
        self.setMinimumHeight(self.total_height)

    def update_height(self, max_height=30):
        self.total_height -= (max_height + self.spacing)
        self.setMinimumHeight(self.total_height)

    def _title_edited(self):
        title = self.txt_title.text()
        if title == self.prev_title:
            return
        if self.card_title_edited(title):
            self.prev_title = title
        else:
            self.txt_title.setText(self.prev_title)

    def card_deleted(self):
        pass

    def card_title_edited(self, title):
        return True

    def card_add_item(self):
        pass

    def showContent(self, flag):
        for i in range(1, self.vbox.count()):
            self.vbox.itemAt(i).widget().setVisible(flag)
        self.setMinimumHeight(self.total_height if flag else 30)
        if self.btn_show:
            self.btn_show.blockSignals(True)
            self.btn_show.setChecked(flag)
            self.btn_show.blockSignals(False)
        self._visible = flag

    def collapse(self):
        self.showContent(False)

    def expand(self):
        self.showContent(True)

    def card_toggled(self):
        self.showContent(not self._visible)


class Spacing(QCSWidget):

    def __init__(self, spacing, parent=None):
        super().__init__(parent=parent)
        self.setMinimumWidth(spacing)
        self.setMaximumWidth(spacing)

class DelIconButton(IconButton):

    def __init__(self, secondary=False, **kwargs):
        kwargs.setdefault('flat', True)
        kwargs.setdefault('error', True)
        icon = 'fa.times-circle-o' if secondary else 'fa.times'
        super().__init__(icon, **kwargs)


class AddIconButton(IconButton):

    def __init__(self, **kwargs):
        kwargs.setdefault('flat', True)
        super().__init__('fa.plus', **kwargs)


class ViewIconButton(IconButton):

    def __init__(self, **kwargs):
        kwargs.setdefault('flat', True)
        super().__init__('fa.chevron-down', toggleable=True, **kwargs)
        self.setCheckable(True)
        self.toggled.connect(self.update_icon)
        self.setChecked(True)
        self.update_icon()

    def update_icon(self):
        self.icon = 'fa.chevron-down' if self.isChecked() else 'fa.chevron-up'
        super().leaveEvent()

    def setChecked(self, flag):
        super().setChecked(flag)
        self.update_icon()


class CheckBox(QCSWidget):

    def __init__(self, text=None, checked=False, align=QtCore.Qt.AlignRight,
                 parent=None):
        super().__init__(parent=parent)
        self.chk = QtWidgets.QCheckBox()
        self.chk.setFixedWidth(20)
        hbox = HBox(self, margin=0, spacing=0)

        if text is None:
            hbox.addWidget(self.chk, 1)
            self.txt = None
        else:
            self.txt = QtWidgets.QLabel(text)
            if align == QtCore.Qt.AlignRight:
                hbox.addWidget(self.chk)
                hbox.addWidget(self.txt, 1)
                text_align = QtCore.Qt.AlignLeft
            else:
                hbox.addWidget(self.txt, 1)
                hbox.addWidget(self.chk)
                text_align = QtCore.Qt.AlignRight
            self.txt.setAlignment(QtCore.Qt.AlignVCenter | text_align)
            self.txt.mousePressEvent = self.label_clicked
        self.chk.setChecked(checked)


    def __getattr__(self, attr):
        return self.chk.__getattribute__(attr)

    def setText(self, text):
        if self.txt:
            self.txt.setText(text)

    def label_clicked(self, evt):
        self.chk.toggle()

    def value(self):
        return self.isChecked()


class LineEdit3D(QCSWidget):

    def __init__(self, *args, parent=None, **kwargs):
        super().__init__(parent=parent)
        default = kwargs.get('default', 0)
        if type(default) not in [list, tuple]:
            default = [default] * 3
        self.line_edits = []
        self.hbox = HBox(self, spacing=0)
        self.hbox.addWidget(QtWidgets.QWidget(), 1)
        for i, c in enumerate(['z', 'y', 'x']):
            kwargs.setdefault('default', default[i])
            le = LineEdit(*args, **kwargs)
            le.setAlignment(QtCore.Qt.AlignCenter)
            label = QtWidgets.QLabel(c + ':')
            self.hbox.addWidget(label)
            self.hbox.addWidget(le)
            self.line_edits.append(le)

    def value(self):
        return tuple(le.value() for le in self.line_edits)

    def setValue(self, value):
        if type(value) not in [list, tuple]:
            value = [value] * 3
        for i, v in enumerate(value):
            self.line_edits[i].setValue(v)


class AbstractLazyWrapper(QtCore.QObject):

    def __init__(self, lazy=False, parent=None):
        super().__init__(parent)
        if lazy:
            self.installEventFilter(self)
            self.update()

    def eventFilter(self, target, evt):
        if target == self and evt.type() == QtCore.QEvent.MouseButtonPress:
            self.update()
        return False

    def update(self):
        raise NotImplementedError()


class ComboBox(QtWidgets.QComboBox, AbstractLazyWrapper):

    def __init__(self, select=None, header=None, lazy=False, parent=None):
        self._items = []
        self._header = header
        QtWidgets.QComboBox.__init__(self, parent)
        AbstractLazyWrapper.__init__(self, lazy)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.currentIndexChanged.connect(self.clearFocus)

        if select is not None:
            self.blockSignals(True)
            self.setCurrentIndex(select)
            self.blockSignals(False)

    def addCategory(self, item):
        n = self.count()
        self.addItem(item)
        self.model().item(n).setEnabled(False)

    def _add_header(self):
        if self._header is None:
            return
        if type(self._header) in [tuple, list]:
            self.addItem(self._header[0], self._header[1])
        else:
            self.addItem(self._header, str(self._header))

    def update(self):
        self.blockSignals(True)
        prev = self.keys()
        self.clear()
        self._add_header()
        self.fill()
        prev_exists = False
        for i, (key, value, data) in enumerate(self.items()):
            if key in prev:
                self.select(key)
                prev_exists = True
        self.blockSignals(False)
        if not prev_exists:
            self.currentIndexChanged.emit(0)

    def wheelEvent(self, evt):
        evt.accept()

    def addItem(self, key, value=None, icon=None, data=None):
        self._items.append((key, value, data))
        if icon:
            super().addItem(icon, value if value else key)
        else:
            super().addItem(value if value else key)

    def items(self):
        return (item for item in self._items)

    def removeItem(self, idx):
        self._items.pop(idx)
        super().removeItem(idx)

    def key(self):
        return self.value(key=True)

    def keys(self):
        return [self.key()]

    def value(self, key=False):
        if self.currentIndex() < 0:
            return None
        item = self._items[self.currentIndex()]
        return item[0] if key or item[2] is None else item[2]

    def values(self, key=False):
        return [self.value()]

    def select(self, key):
        for i, (k, v, d) in enumerate(self._items):
            if not k:
                continue
            kname = k.split(op.sep)
            if k == key or (len(kname) > 1 and kname[1] == key):
                return self.setCurrentIndex(i)

    def clear(self):
        super().clear()
        self._items.clear()


class LazyComboBox(ComboBox):

    def __init__(self, *args, **kwargs):
        kwargs['lazy'] = True
        super().__init__(*args, **kwargs)


class MultiComboBox(QtWidgets.QPushButton, AbstractLazyWrapper):

    valueChanged = QtCore.pyqtSignal()

    def __init__(self, header=None, lazy=False, select=None,
                 text='Select', groupby=None, parent=None):
        self._items = []
        self._actions = []
        self._header = header
        self._groupby = groupby
        self._text = text
        QtWidgets.QPushButton.__init__(self, parent=parent)
        self._toolmenu = QtWidgets.QMenu(self)
        self._toolmenu.installEventFilter(self)
        AbstractLazyWrapper.__init__(self, lazy)
        self.setMenu(self._toolmenu)
        #self.setPopupMode(QtWidgets.QToolButton.InstantPopup)

        self.setProperty('combo', True)
        self._toolmenu.setProperty('combo', True)
        self._toolmenu.aboutToHide.connect(self._update_text)

        self._update_text()

        # TODO: improve this
        file_path = resource('qcs', 'survos.qcs')
        if op.isfile(file_path):
            with open(file_path, 'r') as f:
                style = f.read()
                self.setStyleSheet(style)
                self._toolmenu.setStyleSheet(style)

    def _add_header(self):
        if self._header is not None:
            self.addItem(*self._header)

    def _update_text(self):
        names = self.names()
        if len(names) == 0:
            self.setText(self._text)
        else:
            self.setText('; '.join(names))

    def eventFilter(self, target, evt):
        super().eventFilter(target, evt)
        if evt.type() == QtCore.QEvent.MouseButtonRelease:
            action = self._toolmenu.activeAction()
            if action:
                action.setChecked(not action.isChecked())
                self.valueChanged.emit()
            return True
        return False

    def addCategory(self, name):
        item = self.addItem(name)
        item.setEnabled(False)

    def addItem(self, key, value=None, icon=None, data=None, checkable=True):
        self._items.append((key, value, data))
        text = value if value else key
        if icon:
            action = self._toolmenu.addAction(icon, text)
        else:
            action = self._toolmenu.addAction(text)
        action.setCheckable(checkable)
        self._actions.append(action)
        return action

    def items(self):
        return (item for item in self._items)

    def removeItem(self, idx):
        self._items.pop(idx)
        self._toolmenu.removeAction(self._actions.pop(idx))

    def update(self):
        prev = list(self.keys())
        self.clear()
        self._add_header()
        self.fill()
        for i, (key, value, data) in enumerate(self.items()):
            if key in prev:
                prev.remove(key)
                self.setItemChecked(i, True)
        self._update_text()
        if len(prev) > 0:
            self.valueChanged.emit()

    def clear(self):
        self._items.clear()
        self._actions.clear()
        self._toolmenu.clear()

    def select(self, key):
        for i, (k, v, d) in enumerate(self._items):
            if not k:
                continue
            kname = k.split(op.sep)
            if k == key or (len(kname) > 1 and kname[1] == key):
                if not self.itemChecked(i):
                    self.setItemChecked(i, True)
                    self._update_text()
                    self.valueChanged.emit()
                    return True
                break
        return False

    def select_prefix(self, key):
        found = False
        for i, k in enumerate(self._items):
            if k and k[0].startswith(key):
                if not self.itemChecked(i):
                    found = True
                    self.setItemChecked(i, True)
        if found:
            self.valueChanged.emit()
        return found

    def key(self):
        return list(self.keys())

    def value(self):
        return list(self.values())

    def itemChecked(self, idx):
        return self._actions[idx].isChecked()

    def setItemChecked(self, idx, flag=True):
        self._actions[idx].setChecked(flag)

    def names(self):
        return list(item[1] for i, item in enumerate(self.items())
                    if self.itemChecked(i))

    def keys(self):
        return self.values(keys=True)

    def values(self, keys=False):
        values = (item[0] if item[2] is None or keys else item[2]
                  for i, item in enumerate(self.items())
                  if self.itemChecked(i))
        if keys:
            return values
        if self._groupby:
            result = defaultdict(list)
            for val in values:
                if len(self._groupby) == 2:
                    group, target = self._groupby
                    if group in val:
                        result[val[group]].append(val[target])
                else:
                    if self._groupby in val:
                        result[val[self._groupby]].append(val)
            return [(k, v) for k,v in result.items()]
        return values


class LazyMultiComboBox(MultiComboBox):

    def __init__(self, **kwargs):
        kwargs['lazy'] = True
        super().__init__(**kwargs)


class Slider(QCSWidget):

    valueChanged = QtCore.pyqtSignal(int)

    def __init__(self, value=None, vmax=100, vmin=0, step=1, tracking=True,
                 label=True, auto_accept=True, center=False, parent=None):
        super().__init__(parent=parent)
        if value is None:
            value = vmin
        self.setMinimumWidth(200)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(vmin)
        self.slider.setMaximum(vmax)
        self.slider.setValue(value)
        self.slider.setTickInterval(step)
        self.slider.setSingleStep(step)
        self.slider.setTracking(tracking)
        self.step = step

        hbox = HBox(self, spacing=5)
        if label:
            self.label = Label(str(value))
            self.label.setMinimumWidth(50)
            if center:
                hbox.addSpacing(50)
            hbox.addWidget(self.slider, 1)
            hbox.addWidget(self.label)
            self.valueChanged.connect(self.update_label)
        else:
            hbox.addWidget(self.slider, 1)

        self.slider.valueChanged.connect(self.value_changed)
        self.slider.wheelEvent = self.wheelEvent
        self.auto_accept = auto_accept
        self.locked_idx = None
        self.pending = None
        self.blockSignals = self.slider.blockSignals

    def value_changed(self, idx):
        if self.auto_accept:
            self.valueChanged.emit(idx)
        elif self.locked_idx is None:
            self.locked_idx = idx
            self.valueChanged.emit(idx)
        else:
            self.slider.blockSignals(True)
            self.slider.setValue(self.locked_idx)
            self.slider.blockSignals(False)
            self.pending = idx

    def accept(self):
        if self.pending is not None:
            val = self.pending
            self.pending = None
            self.slider.blockSignals(True)
            self.slider.setValue(val)
            self.slider.blockSignals(False)
            self.valueChanged.emit(val)
        self.locked_idx = None

    def update_label(self, idx):
        self.label.setText(str(idx))

    def wheelEvent(self, e):
        if e.angleDelta().y() > 0 and self.value() < self.maximum():
            self.setValue(self.value()+self.step)
        elif e.angleDelta().y() < 0 and self.value() > self.minimum():
            self.setValue(self.value()-self.step)

    def value(self):
        return self.pending or self.slider.value()

    def setValue(self, value):
        return self.slider.setValue(value)

    def __getattr__(self, key):
        return self.slider.__getattribute__(key)


class RealSlider(Slider):

    def __init__(self, value=0, vmax=100, vmin=0, n=1000, **kwargs):
        super().__init__(value=0, vmin=0, vmax=n, **kwargs)
        self._n = n
        self._vmin = vmin
        self._vmax = vmax
        self._update_linspace()
        self.blockSignals(True)
        self.setValue(value)
        self.update_label(self._mapvalue(value))
        self.blockSignals(False)

    def _mapvalue(self, val):
        return (np.abs(self._values - val)).argmin()

    def value(self):
        return self._values[self.slider.value()]

    def update_label(self, idx):
        idx = '{0:.3f}'.format(self._values[idx])
        super().update_label(idx)

    def _update_linspace(self):
        self._values = np.linspace(self._vmin, self._vmax,
                                   self._n + 1, endpoint=True)

    def setValue(self, val):
        idx = self._mapvalue(val)
        super().setValue(idx)

    def setMaximum(self, vmax):
        self._vmax = vmax
        self._update_linspace()

    def setMinimum(self, vmin):
        self._vmin = vmin
        self._update_linspace()

    def maximum(self):
        return self._vmax

    def minimum(self):
        return self._vmin


class Label(QtWidgets.QLabel):

    def __init__(self, *args):
        super().__init__(*args)
        self.setAlignment(QtCore.Qt.AlignCenter)

    def value(self):
        return self.text()
