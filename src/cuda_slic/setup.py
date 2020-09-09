import os

from Cython.Build import build_ext
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs


def configuration(parent_package="", top_path=None):
    config = Configuration(
        "regions", parent_package, top_path, cmdclass={"build_ext": build_ext}
    )

    config.add_extension(
        "_rag", sources="_rag.pyx", include_dirs=[get_numpy_include_dirs()]
    )

    config.add_extension(
        "_ccl", sources="_ccl.pyx", include_dirs=[get_numpy_include_dirs()]
    )

    return config


if __name__ == "__main__":
    config = configuration(top_path="").todict()
    setup(**config)
