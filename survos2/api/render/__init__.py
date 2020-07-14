

import hug

from . import api

hug.API(__name__).extend(api)
