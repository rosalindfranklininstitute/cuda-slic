

import time
import multiprocessing

from requests.exceptions import ConnectTimeout, ConnectionError

from survos2.ui.qt.control.model import DataModel
from survos2.ui.qt.control.singleton import Singleton

from survos2.ui.qt.qtcompat import QtCore, QtWidgets
from survos2.ui.qt.modal import ModalManager

from survos2.utils import format_yaml, get_logger, Timer
from survos2.survos import remote_client, parse_response

logger = get_logger()


@Singleton
class Launcher(QtCore.QObject):

    def __init__(self):
        super().__init__()
        self.connected = True
        self.terminated = False

    def set_remote(self, uri):
        self.client = remote_client(uri)
        self.modal = False
        self.prending = []

    def run(self, plugin, command, modal=False, **kwargs):
        if self.terminated:
            return False

        self.modal = modal
        self.title = '{}::{}'.format(plugin.capitalize(),
                                     command.capitalize())
        self.setup(self.title)
        workspace = kwargs.pop('workspace', None)
        if workspace == True:
            if DataModel.g.current_workspace:
                kwargs['workspace'] = DataModel.g.current_workspace
            else:
                return self.process_error('Workspace required but not loaded.')
        elif workspace is not None:
            kwargs['workspace'] = workspace

        func = self._run_background if modal else self._run_command

        success = False
        while not success:
            try:
                if kwargs.pop('timeit', False):
                    with Timer(self.title):
                        result, error = func(plugin, command, **kwargs)
                else:
                    result, error = func(plugin, command, **kwargs)
            except (ConnectTimeout, ConnectionError):
                self.connected = False
                logger.info('ConnectionError - delayed')
                ModalManager.g.connection_lost()
                if self.terminated:
                    return False
            else:
                success = True

        if error:
            return self.process_error(result)

        self.cleanup()
        return result

    def reconnect(self):
        try:
            params = dict(workspace=DataModel.g.current_workspace)
            self._run_command('workspace', 'list_datasets', **params)
        except (ConnectTimeout, ConnectionError):
            pass
        else:
            self.connected = True

    def _run_background(self, plugin, command, **kwargs):
        queue = multiprocessing.Queue()
        kwargs.update(out=queue)
        p = multiprocessing.Process(target=self._run_command,
                                    args=[plugin, command],
                                    kwargs=kwargs)
        p.daemon = True
        p.start()
        while p.is_alive():
            QtWidgets.QApplication.processEvents()
            p.join(0.1)
        return queue.get()

    def _run_command(self, plugin, command, out=None, **kwargs):
        response = self.client.get('{}/{}'.format(plugin, command), **kwargs)
        result = parse_response(plugin, command, response, log=False)
        if out is not None:
            out.put(result)
        else:
            return result

    def setup(self, caption):
        logger.info('### {} ###'.format(caption))
        if self.modal:
            ModalManager.g.show_loading(caption)

    def cleanup(self):
        if self.modal:
            ModalManager.g.hide()
        QtWidgets.QApplication.processEvents()

    def process_error(self, error):
        if not isinstance(error, str):
            error = format_yaml(error, explicit_start=False, explicit_end=False, flow=False)
        try:
            traceback.print_last()
        except Exception as e:
            pass
        logger.error('{} :: {}'.format(self.title, error))
        ModalManager.g.show_error(error)
        QtWidgets.QApplication.processEvents()
        return False

    def terminate(self):
        pass
