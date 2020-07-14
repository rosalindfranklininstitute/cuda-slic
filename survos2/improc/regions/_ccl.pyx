#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False


import numpy as np
cimport numpy as np


cdef float DMAX = 9999999

ctypedef unsigned int uint32
ctypedef unsigned short uint16


cdef uint32 _find_root(uint32[::1] labels, uint32 p):
    cdef uint32 q = p
    while labels[p] != p:
        p = labels[p]
    labels[q] = p
    return p


cdef void _join_trees(uint32[::1] labels, uint32[::1] sizes, uint32 p, uint32 q):
    if p == q:
        return

    if sizes[p] < sizes[q]:
        labels[p] = q
        sizes[q] += sizes[p]
    else:
        labels[q] = p
        sizes[p] += sizes[q]


cdef void _cremap(uint32[::1] src, int[::1] out):
    cdef int N = src.shape[0]
    cdef int p, n
    cdef int curr_label = 0, new_label

    for n in range(N):
        p = _find_root(src, n)
        if out[p] < 0:
            new_label = curr_label
            out[p] = new_label
            curr_label += 1
        else:
            new_label = out[p]
        out[n] = new_label


def _remap(uint32[::1] src):
    cdef int N = src.shape[0]
    cdef int[::1] out = np.full(N, -1, np.int32)
    _cremap(src, out)
    return np.asarray(out, dtype=np.uint32)


def _relabel2d(uint32[::1] src, int width):
    cdef int N = src.shape[0]
    cdef uint32 n, p, q, v

    cdef uint32[::1] sizes = np.ones(N, np.uint32)
    cdef uint32[::1] labels = np.arange(N, np.uint32)
    cdef int[::1] out = np.full(N, -1, np.int32)

    for n in range(1, N):
        v = src[n]
        p = _find_root(labels, n)

        # x > 0
        if src[n - 1] == v:
            q = _find_root(labels, n - 1)
            if p != q:
                _join_trees(labels, sizes, p, q)

        # y > 0
        if n > width and src[n - width] == v:
            q = _find_root(labels, n - width)
            if p != q:
                _join_trees(labels, sizes, p, q)

    _cremap(labels, out)
    return np.asarray(out, dtype=np.uint32)


def _relabel3d(uint32[::1] src, int height, int width):
    cdef int N = src.shape[0]
    cdef int plane = height * width
    cdef uint32 n, p, q, v

    cdef uint32[::1] sizes = np.ones(N, np.uint32)
    cdef uint32[::1] labels = np.arange(N, dtype=np.uint32)
    cdef uint32[::1] tmplabels
    cdef int[::1] out = np.full(N, -1, np.int32)

    for n in range(1, N):
        v = src[n]
        p = _find_root(labels, n)

        # x > 0
        if src[n - 1] == v:
            q = _find_root(labels, n - 1)
            if p != q:
                _join_trees(labels, sizes, p, q)

        # y > 0
        if n > width and src[n - width] == v:
            q = _find_root(labels, n - width)
            if p != q:
                _join_trees(labels, sizes, p, q)

        # z > 0
        if n > plane and src[n - plane] == v:
            q = _find_root(labels, n - plane)
            if p != q:
                _join_trees(labels, sizes, p, q)

    _cremap(labels, out)
    return np.asarray(out, dtype=np.uint32)


cdef float __compute_dist(float[:, ::1] data, int i, int j):
    cdef int C = data.shape[1], w
    cdef float diff = 0, d
    for w in range(C):
        d = data[i, w] - data[j, w]
        diff += (d * d)
    return diff


def _merge_small3d(float[:, :, :, ::1] data,
                   uint32[:, :, ::1] labels,
                   int nsv, int min_size):
    cdef int D = data.shape[0]
    cdef int H = data.shape[1]
    cdef int W = data.shape[2]
    cdef int C = data.shape[3]
    cdef uint32[::1] svlabels = np.arange(nsv, dtype=np.uint32)
    cdef int[::1] svresult = np.full(nsv, -1, dtype=np.int32)
    cdef float[:, ::1] svdesc = np.zeros((nsv, C), dtype=np.float32)
    cdef float[::1] svdist = np.full(nsv, DMAX, np.float32)
    cdef uint16[::1] counts = np.zeros(nsv, np.uint16)
    cdef int i, j, k, w
    cdef uint32 r, r2, p, p2
    cdef float dist

    for k in range(D):
        for i in range(H):
            for j in range(W):
                r = labels[k, i, j]
                counts[r] += 1
                for w in range(C):
                    svdesc[r, w] += data[k, i, j, w]

    for r in range(nsv):
        if counts[r] > 0:
            for w in range(C):
                svdesc[r, w] /= counts[r]

    for k in range(D - 1):
        for i in range(H - 1):
            for j in range(W - 1):
                r = labels[k, i, j]
                p = _find_root(svlabels, r);

                # X
                r2 = labels[k, i, j+1]
                p2 = _find_root(svlabels, r2)
                if r != r2 and counts[r2] < min_size and p != p2:
                    dist = __compute_dist(svdesc, r, r2)
                    if dist < svdist[r2]:
                        svdist[r2] = dist
                        svlabels[r2] = r
                elif r != r2 and counts[r] < min_size and p != p2:
                    dist = __compute_dist(svdesc, r, r2)
                    if dist < svdist[r]:
                        svdist[r] = dist
                        svlabels[r] = r2

                # Y
                r2 = labels[k, i+1, j]
                p2 = _find_root(svlabels, r2)
                if r != r2 and counts[r2] < min_size and p != p2:
                    dist = __compute_dist(svdesc, r, r2)
                    if dist < svdist[r2]:
                        svdist[r2] = dist
                        svlabels[r2] = r
                elif r != r2 and counts[r] < min_size and p != p2:
                    dist = __compute_dist(svdesc, r, r2)
                    if dist < svdist[r]:
                        svdist[r] = dist
                        svlabels[r] = r2

                # Z
                r2 = labels[k+1, i, j]
                p2 = _find_root(svlabels, r2)
                if r != r2 and counts[r2] < min_size and p != p2:
                    dist = __compute_dist(svdesc, r, r2)
                    if dist < svdist[r2]:
                        svdist[r2] = dist
                        svlabels[r2] = r
                elif r != r2 and counts[r] < min_size and p != p2:
                    dist = __compute_dist(svdesc, r, r2)
                    if dist < svdist[r]:
                        svdist[r] = dist
                        svlabels[r] = r2

    _cremap(svlabels, svresult)
    return np.asarray(svresult, dtype=np.uint32)[labels]


