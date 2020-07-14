

import numpy as np
from skimage import io

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from survos2.ui.qt.components import *
from survos2.ui.qt.utils import resource
from survos2.ui.qt.control import DataModel
from survos2.ui.qt.qtcompat import QtCore, QtWidgets, QtGui
from survos2.ui.qt.plugins.base import ViewerExtension

from survos2.ui.qt.views.base import register_view


class ZoomPan(ViewerExtension):

    def __init__(self, zoom_factor=2, max_zoom_level=4):
        super().__init__(modifiers=QtCore.Qt.ControlModifier)
        self.data_size = [1, 1]
        self.dragging = False
        self.cur_xlim = None
        self.cur_ylim = None
        self.xpress = None
        self.ypress = None

        self.current_zoom = 1
        self.zoom_factor = zoom_factor
        self.max_zoom = 1. / (zoom_factor**max_zoom_level)

    def install(self, fig, axes):
        super().install(fig, axes)
        self.connect('scroll_event', self.do_zoom)
        self.connect('button_press_event', self.pan_press)
        self.connect('button_release_event', self.pan_release)
        self.connect('motion_notify_event', self.pan_motion)

    def do_zoom(self, event):
        # https://stackoverflow.com/questions/11551049/matplotlib-plot-zooming-with-scroll-wheel
        ax = self.axes
        if event.inaxes != ax or not self.active():
            return
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location

        if event.button == 'up':
            scale_factor = 1 / self.zoom_factor
        elif event.button == 'down':
            scale_factor = self.zoom_factor
        else:
            return

        new_zoom = self.current_zoom * scale_factor
        if new_zoom < self.max_zoom:
            return
        if new_zoom > 1:
            return self.center()

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
        ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])

        self.current_zoom = new_zoom
        self.redraw()

    def pan_press(self, event):
        if event.button != 1:
            return
        ax = self.axes
        if event.inaxes != ax or not self.active():
            return
        self.cur_xlim = ax.get_xlim()
        self.cur_ylim = ax.get_ylim()
        self.dragging = True
        self.xpress, self.ypress = event.xdata, event.ydata

    def pan_release(self, event):
        self.dragging = False
        self.redraw()

    def pan_motion(self, event):
        ax = self.axes
        if not self.dragging or event.inaxes != ax:
            return
        if not self.active():
            return self.pan_release(event)
        dx = event.xdata - self.xpress
        dy = event.ydata - self.ypress
        self.cur_xlim -= dx
        self.cur_ylim -= dy
        ax.set_xlim(self.cur_xlim)
        ax.set_ylim(self.cur_ylim)
        self.redraw()

    def center(self):
        self.axes.set_xlim(0, self.data_size[1])
        self.axes.set_ylim(self.data_size[0], 0)
        self.current_zoom = 1
        self.redraw()


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.9)

        self.compute_initial_figure()

        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.updateGeometry()

    def compute_initial_figure(self):
        raise NotImplementedError()

    def redraw(self):
        self.draw_idle()

    def update_figure(self):
        raise NotImplementedError()


class SliceMplCanvas(MplCanvas):

    def __init__(self, axes_color='#0277BD', parent=None):
        self.axes_color = axes_color
        super().__init__(parent=parent)

    def compute_initial_figure(self):
        self.image = None
        self.axes.xaxis.label.set_color(self.axes_color)
        self.axes.yaxis.label.set_color(self.axes_color)
        self.axes.tick_params(colors=self.axes_color)

    def replot_figure(self, image):
        self.axes.clear()
        self.image = self.axes.imshow(image)
        self.redraw()

    def update_figure(self, image):
        if self.image is None:
            return self.replot_figure(image)
        self.image.set_data(image)
        self.redraw()


@register_view(name='viewer')
class Viewer(QCSWidget):

    def __init__(self, camera=True, parent=None):
        super().__init__(parent=parent)
        self.canvas = SliceMplCanvas(parent=self)
        vbox = VBox(self)
        vbox.addWidget(self.canvas, 1)

        if camera:
            self.camera = ZoomPan()
            self.camera.data_size = DataModel.g.current_workspace_shape[1:]
            self.install_extension(self.camera)

    def mpl_connect(self, event, callback):
        return self.canvas.mpl_connect(event, callback)

    def update_image(self, image):
        self.canvas.update_figure(image)

    def update(self):
        pass

    def center(self):
        self.camera.center()

    def set_region(self, region):
        self.canvas.set_region(region)

    def install_extension(self, ext):
        ext.install(self.canvas, self.canvas.axes)
