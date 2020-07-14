

import pytest
import logging

from survos2.utils import get_logger


def pytest_addoption(parser):
    parser.addoption("--loglevel", default=logging.ERROR)


@pytest.fixture(scope="session", autouse=True)
def logger(request):
    loglevel = request.config.getoption('--loglevel')
    try:
        loglevel = int(loglevel)
    except ValueError:
        loglevel = loglevel.upper()
    return get_logger(level=loglevel)