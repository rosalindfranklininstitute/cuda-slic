

from survos2.ui.qt.components import *
from survos2.ui.qt.qtcompat import QtWidgets, QtCore, QtGui
from survos2.ui.qt.control.singleton import Singleton
from survos2.ui.qt.utils import resource


class _Modal(QCSWidget):

    def __init__(self, message, parent=None):
        super().__init__(parent=parent)
        vbox = VBox(self, margin=30, spacing=20, align=QtCore.Qt.AlignCenter)
        self.label = Label(message)
        self.label.setWordWrap(True)
        vbox.addWidget(self.label)

    def setMessage(self, msg):
        self.label.setText(msg)


class LoadingModal(QCSWidget):

    def __init__(self, message, parent=None):
        super().__init__(parent=parent)
        vbox = VBox(self, margin=30, align=QtCore.Qt.AlignCenter)
        self.loading = Label()
        movie = QtGui.QMovie(resource("loading.gif"))
        self.loading.setMovie(movie)
        self.label = Label(message)
        self.label.setWordWrap(True)
        vbox.addWidget(self.loading)
        vbox.addWidget(self.label)
        movie.start()

    def setMessage(self, msg):
        self.label.setText(msg)


class AcceptModal(_Modal):

    accepted = QtCore.pyqtSignal()

    def __init__(self, message, btn_text='Close', parent=None):
        super().__init__(message, parent=parent)
        self.btn = PushButton(btn_text)
        if parent:
            self.btn.clicked.connect(parent.hide)
        self.btn.clicked.connect(self._trigger)
        self(HWidgets(None, self.btn, None))

    def _trigger(self):
        self.accepted.emit()


class ErrorModal(AcceptModal):

    def __init__(self, message, **kwargs):
        super().__init__(message, **kwargs)
        self.setProperty('error', True)


class WarnModal(AcceptModal):

    def __init__(self, message, **kwargs):
        super().__init__(message, **kwargs)
        self.setProperty('warn', True)


class YesNoModal(_Modal):

    accepted = QtCore.pyqtSignal(bool)

    def __init__(self, message, msg_yes='Accept', msg_no='Cancel', parent=None):
        super().__init__(message, parent=parent)
        self.yes = PushButton(msg_yes)
        self.no = PushButton(msg_no)
        self.yes.clicked.connect(self._yes)
        self.no.clicked.connect(self._no)
        self(HWidgets(None, self.no, None, self.yes, None))

    def _yes(self):
        self.accepted.emit(True)

    def _no(self):
        self.accepted.emit(False)


@Singleton
class ModalManager(QtWidgets.QDialog, SWidget):

    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent=parent)
        SWidget.__init__(self, self.__class__.__name__, parent=parent)
        if parent:
            parent.resized.connect(self.parent_resized)
            self.parent_resized()
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.container = VBox(self, align=QtCore.Qt.AlignCenter)
        self.current_widget = None
        self.showing_exit = False
        self.response = False
        self.hide()

    def parent_resized(self):
        self.move(QtCore.QPoint(0, 0))
        self.resize(self.parent().size())

    def show(self, widget=None, save_state=True, block=False):
        clear_layout(self.container)
        if widget:
            widget.setProperty('container', True)
            self.container.addWidget(widget)
            if save_state:
                self.current_widget = widget
        if block:
            super().exec_()
        else:
            super().show()
        QtWidgets.QApplication.processEvents()

    def show_loading(self, caption):
        self.show(LoadingModal(caption))

    def show_error(self, errmsg):
        widget = ErrorModal(errmsg, parent=self)
        self.show(widget)

    def connection_lost(self):
        self.prev_modal = self.current_widget
        errmsg = 'Connection with the server lost. Try reconnecting.'
        widget = WarnModal(errmsg, btn_text='Reconnect')
        widget.accepted.connect(self._try_reconnect)
        self.show(widget, block=True)

    def hide(self):
        self.prev_modal = self.current_widget = None
        super().hide()

    def terminate(self):
        label = Label('Connection to . Restart the client.')
        label.setProperty('error', True)
        self.show(label)

    def _try_reconnect(self):
        from survos2.ui.qt.control import Launcher

        for i in range(5):
            self.current_widget.setMessage('Attempting to reconnect.. ({}/5)'.format(i+1))
            Launcher.g.reconnect()

        if not Launcher.g.connected:
            errmsg = 'Unable to reconnect to the server. Check your connection ' \
                     'and that Server is running and try again.'
            self.current_widget.setMessage(errmsg)
        else:
            self.accept()
            if self.prev_modal:
                self.show(self.prev_modal)


