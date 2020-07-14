


class Singleton(object):
    """
    Source: http://stackoverflow.com/a/7346105/764322
    """
    def __init__(self, decorated):
        self._decorated = decorated

    @property
    def g(self):
        return self.instance()

    def instance(self, *args, **kwargs):
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated(*args, **kwargs)
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)