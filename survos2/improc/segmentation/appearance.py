

import numpy as np

from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import RandomForestClassifier

from .mappings import rmeans, rstats, rlabels
from . import _qpbo as qpbo


def train(X_train, y_train, project=False, rnd=42, **kwargs):
    if project is not False:
        if project == 'rproj':
            proj = SparseRandomProjection(n_components=X_train.shape[1], random_state=rnd)
        elif project == 'std':
            proj = StandardScaler()
        elif project == 'pca':
            proj = PCA(n_components='mle', whiten=True, random_state=rnd)
        elif project == 'rpca':
            proj = RandomizedPCA(whiten=True, random_state=rnd)
        elif project == 'rbf':
            proj = RBFSampler(n_components=max(X_train.shape[1], 50), random_state=rnd)
        else:
            raise Error('Projection {} not available'.format(project))

        X_train = proj.fit_transform(X_train)

    kwargs.setdefault('random_state', rnd)
    clf = RandomForestClassifier(**kwargs)
    clf.fit(X_train, y_train)

    if project is not False:
        return clf, proj

    return clf


def predict(X, clf, proj=None, label=True, probs=False, log=False):
    if proj is not None:
        X = proj.transform(X)
    result = {}
    if probs:
        result['probs'] = clf.predict_proba(X)
    if log:
        result['log_probs'] = clf.predict_log_proba(X)
    if label:
        result['class'] = clf.predict(X)
    return result


def rmap(X, y, R, mode='mean', min_ratio=0.1):
    if mode == 'mean':
        Xs = rmeans(X, R)
    elif mode == 'covar':
        Xs = rstats(X, R, mode='add', norm=None)
    elif mode == 'Sigma Set':
        Xs = rstats(X, R, mode='add', sigmaset=True, norm=None)
    else:
        raise ValueError('Uknown mapping mode \'%s\'.' % mode)

    ys = rlabels(y, R, min_ratio=0.1)
    return Xs, ys


def invrmap(Ys, R):
    return Ys[R].astype(np.uint8, copy=False)


def refine(X, U, E, W=None, mode='appearance', lamda=1, gamma=None):
    """
    Refine predictions with an MRF prior.

    Parameters:
    -----------
    X : (N, D) numpy array
        Features (voxel, or regional)
    U : (N, L) numpy array
        Unary potentials, aka labelling cost for each sample. It has
        to be shape `n_samples x n_labels` (`N x L`).
    E : (E, 2) integer array
        Edges of the graph connecting nodes in `X`.
    W : (E,) float array
        Similarity weight between nodes of each edge.

    Returns:
    --------
    y : (N,) uint8 numpy array
        New labels for nodes in `X`
    """

    L = U.shape[1]

    if mode == 'appearance':
        D = np.sqrt(np.sum((X[E[:, 0]] - X[E[:, 1]])**2, axis=1))
        if gamma is False:
            P = 1 - D / D.max()
        else:
            if gamma is None:
                gamma = 1. / (2 * np.mean(D)**2)
            P = np.exp(-gamma * D)
    else:
        P = np.ones(X.shape[0], dtype=np.float32)

    if W is not None:
        P *= W

    C = np.ones((L, L), dtype=np.float32) - np.eye(L, dtype=np.float32)
    C *= lamda

    U = U.astype(np.float32, copy=False)
    P = P.astype(np.float32, copy=False)
    E = E.astype(np.int32, copy=False)
    C = C.astype(np.float32, copy=False)

    if L == 2:
        return qpbo.solve_binary(E, U[:, ::-1].copy(), P, C)
    else:
        return qpbo.solve_aexpansion(E, U, P, C)
