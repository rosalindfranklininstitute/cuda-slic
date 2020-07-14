

import re

import hug
from hug.use import HTTP, Local

import logging
from importlib import import_module

from .config import Config
from .api.utils import APIException, handle_exceptions, handle_api_exceptions


__plugins = {}
__api_started = False


def _parse_uri(uri):
    if type(uri) == str:
        pattern = r'@?(?P<host>\w+):(?P<port>[0-9]+)'
        result = re.search(pattern, uri)
        if result is not None:
            return result['host'], result['port']
    elif len(uri) == 2:
        return uri[0], uri[1]
    raise ValueError('Not a valid [host:port] URI.')


def init_api(return_plugins=False):
    global __api_started
    if __api_started:
        return

    # Init app
    api = hug.API(__name__)
    # Set exception handlers
    hug.exception(api=api)(handle_exceptions)
    hug.exception(APIException, api=api)(handle_api_exceptions)
    # Load plugins
    __plugins.clear()

    if Config['api.plugins']:
        for plugin_name in Config['api.plugins']:
            plugin_module = 'survos2.api.{}'.format(plugin_name)
            plugin_path = '/' + plugin_name
            plugin = import_module(plugin_module)
            api.extend(plugin, plugin_path)
            __plugins[plugin_name] = dict(name=plugin_name,
                                          modname=plugin_module,
                                          path=plugin_path,
                                          module=plugin)
    __api_started = True
    if return_plugins:
        return api, __plugins
    return api


def remote_client(uri):
    host, port = _parse_uri(uri)
    endpoint = 'http://{}:{}/'.format(host, port)
    return HTTP(endpoint)


def run_command(plugin, command, uri=None, **kwargs):
    if uri is None:
        init_api()
        client = Local(__plugins[plugin]['module'])
        try:
            response = client.get(command, **kwargs)
        except APIException as e:
            if not e.critical:
                e.critical = Config['logging.level'].lower() == 'debug'
            handle_api_exceptions(e)
            return str(e), True
    else:
        client = remote_client(uri)
        response = client.get('{}/{}'.format(plugin, command), **kwargs)

    return parse_response(plugin, command, response)


def parse_response(plugin, command, response, log=True):
    if response.data == 'Not Found' or '404' in response.data:
        errmsg = 'API {}/{} does not exist.'.format(plugin, command)
        if log:
            logging.critical(errmsg)
        response = errmsg
        return response, True
    elif response.data['error']:
        errmsg = response.data['error_message']
        if log:
            if response.data['critical']:
                logging.critical(errmsg)
            else:
                logging.error(errmsg)
        return errmsg, True
    elif 'errors' in response.data['data']:
        if log:
            logging.critical(errmsg)
        return response.data['data']['errors'], True

    return response.data['data'], False