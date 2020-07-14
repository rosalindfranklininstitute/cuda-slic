

import numpy as np

from ._mappings import _rmeans, _rstats, _rlabels


def normalize(X, norm='l1'):
    if norm in ['l1', 'hellinger']:
        X /= np.abs(X).sum(axis=1)[:, None]
    elif norm == 'l2':
        X /= np.sqrt((X**2).sum(axis=1))[:, None]
    elif norm == 'linf':
        X /= np.abs(X).max(axis=1)[:, None]
    elif norm == 'unit':
        X -= X.min(0)
        X /= X.max(0)
    if norm == 'hellinger':
        np.sqrt(X, X) # inplace
    return X


def rmeans(X, R, nr=None, norm=None):
    nr = nr or R.max()+1
    features =  _rmeans(X, R.ravel(), nr)
    return normalize(features, norm=norm)


def rstats(X, R, nr=None, mode='append', sigmaset=False, covmode='full', norm=None):
    nr = nr or R.max()+1

    if mode not in ['append', 'add', None]:
        raise Exception('Only `append` or `add` methods are accepted')

    means, covars = _rstats(X, R.flatten(), nr)

    if sigmaset:
        # Add small constant to covars to make them positive-definite
        covars += np.eye(covars.shape[-1])[None, ...] * 1e-5
        covars = np.linalg.cholesky(covars) * np.sqrt(means.shape[1])

        if covmode == 'full':
            y1, x1 = np.tril_indices(means.shape[1], k=-1)
            y2, x2 = np.triu_indices(means.shape[1], k=1)
            covars[:, y2, x2] = covars[:, y1, x1]

    if mode == 'add':
        covars += means[:, :, None]

    if sigmaset and covmode == 'tril':
        y, x = np.tril_indices(means.shape[1])
        covars = covars[:, y, x]
    else:
        covars.shape = (covars.shape[0], -1)

    if mode == 'append':
        X = np.c_[means, covars]
    else:
        X = covars

    return normalize(X, norm=norm)


def rlabels(y, R, nr=None, ny=None, norm=None, min_ratio=0):
    nr = nr or R.max() + 1
    ny = ny or y.max() + 1
    features =  _rlabels(y.ravel(), R.ravel(), ny, nr, min_ratio)
    return normalize(features, norm=norm)
