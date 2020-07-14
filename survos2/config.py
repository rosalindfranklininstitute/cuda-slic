

import yaml
import os
import os.path as op


class _Config(type):

    __data__ = { # Defaults
        'title': 'SuRVoS',
        'api': {
            'host': 'localhost',
            'port': 8000,
            'plugins': [],
            'renderer': 'vispy'
        },
        'computing': {
            'chunks': True,
            'chunk_size': 100,
            'chunk_padding': 5,
            'chunk_size_sparse': 10,
            'scale': False,
            'stretch': False
        },
        'model': {
            'chroot': False,
            'dbtype': 'yaml'
        },
        'logging': {
            'file': '',
            'level': 'error',
            'std': True,
            'std_format': '%(levelname)8s | %(message)s',
            'file_format': '%(asctime)s - | %(levelname)8s | %(message)s'
        },
        'qtui': {
            'maximized': False,
            'menuKey': '\\'
        }
    }

    def __getitem__(self, key):
        return self.get(key)

    def get(self, key):
        keys = key.split('.')
        data = self.__data__
        for i, key in enumerate(keys):
            if key in data:
                data = data[key]
            else:
                raise KeyError('Config does not contain key `{}`'
                               .format('.'.join(keys[:i+1])))
        return data

    def __contains__(self, key):
        try:
            self.get(key)
        except KeyError:
            return False
        return True


class Config(object, metaclass=_Config):

    @staticmethod
    def update(data):
        for k, v in data.items():
            if k == 'environments':
                continue
            if type(v) == dict:
                _Config.__data__[k].update(v)
            else:
                _Config.__data__[k] = v

    def __repr__(self):
        return ''


__default_config_files__ = [
    op.join(op.dirname(__file__), '..', 'settings.yaml'),
    op.join(op.expanduser('~'), '.survosrc')
]

for __config_file in __default_config_files__:
    configs = []
    if op.isfile(__config_file):
        with open(__config_file, 'r') as __f:
            configs.append(yaml.safe_load(__f))
    # Load all the default config
    for config in configs:
        Config.update(config)
    # Overwrite with the enviromental config
    # e.g. activate test environment with SURVOS_ENV=test
    for config in configs:
        envs = config.get('environments', [])
        if envs and 'SURVOS_ENV' in os.environ and os.environ['SURVOS_ENV'] in envs:
            Config.update(envs[os.environ['SURVOS_ENV']])
    # Overwrite with `all` special environment
    for config in configs:
        envs = config.get('environments', [])
        if envs and 'all' in envs:
            Config.update(envs['all'])


# Overwrite config with enviromental variables SURVOS_$section_$setting
# e.g.: replace default renderer with SURVOS_API_RENDERER=mpl
for k1, v in _Config.__data__.items():
    if type(v) == dict:
        for k2 in v:
            env_name = 'SURVOS_{}_{}'.format(k1.upper(), k2.upper())
            if env_name in os.environ:
                try:
                    dtype = type(Config[k1][k2])
                    Config[k1][k2] = dtype(os.environ[env_name])
                except ValueError:
                    raise ValueError('Error updating config {}.{} to {}.'
                                     .format(k1, k2, os.environ[env_name]))
