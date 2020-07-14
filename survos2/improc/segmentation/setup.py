

import os
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
from Cython.Build import build_ext


def configuration(parent_package='', top_path=None):
    config = Configuration('segmentation', parent_package, top_path,
                           cmdclass={'build_ext': build_ext})

    config.add_extension('_mappings', sources=['_mappings.pyx'],
                         include_dirs=[get_numpy_include_dirs()])

    qpbo_dir = 'qpbo_src'
    files = ["QPBO.cpp", "QPBO_extra.cpp", "QPBO_maxflow.cpp",
             "QPBO_postprocessing.cpp"]
    files = [os.path.join(qpbo_dir, f) for f in files]
    files = ['_qpbo.pyx'] + files
    config.add_extension('_qpbo', sources=files, language='c++',
                         libraries=["stdc++"], library_dirs=[qpbo_dir],
                         include_dirs=[qpbo_dir, get_numpy_include_dirs()])

    return config


if __name__ == '__main__':
    config = configuration(top_path='').todict()
    setup(**config)