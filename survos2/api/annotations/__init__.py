

import hug

from . import api

hug.API(__name__).extend(api)

# Public API
from .api import get_labels, get_level, get_levels
