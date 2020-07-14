

import os.path as op


def resource(*args):
    rdir = op.dirname(__file__)
    return op.normpath(op.join(rdir, 'resources', *args))