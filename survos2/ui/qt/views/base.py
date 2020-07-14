

from collections import OrderedDict
from functools import partial


__available_views__ = OrderedDict()


def register_view(cls=None, name=None, title=None):
    if cls is None:
        return partial(register_view, name=name, title=title)

    if name in __available_views__:
        raise ValueError('View {} already registered.'.format(name))

    if title is None:
        title = name.capitalize()

    __available_views__[name] = dict(cls=cls, name=name, title=title)
    return cls


def get_view(name):
    if name not in __available_views__:
        raise ValueError('View {} not registered'.format(name))
    return __available_views__[name]


def list_views():
    return list(__available_views__.keys())