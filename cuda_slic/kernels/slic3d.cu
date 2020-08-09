

#include <cstdio>

#define DLIMIT 99999999

// Cluster Center
//
// float* f;  // vector of size #channels
// float x, y, z;
//

#define __min(a, b) (((a) < (b)) ? (a) : (b))
#define __max(a, b) (((a) >= (b)) ? (a) : (b))

/*
 * P = point
 * S = data shape
 * F = data # features
 */
__device__
float at(const float* data, const int4& P, const int3& S) {
    long s2d = S.y * S.x, s3d = S.z * S.y * S.x;
    return data[P.w * s3d + P.z * s2d + P.y * S.x + P.x];
}

__device__
float gradient(const float* data, int4& P, const int3& S, int nf) {
    float d;
    float3 diff;
    int4 q; q.z = P.z; q.y = P.y; q.x = P.x;

    for ( int k = 0; k < nf; k++ ) {
        q.w = P.w = k;

        q.x = P.x + 1;
        d = at(data, P, S) - at(data, q, S);
        diff.x += d * d;

        q.x = P.x; q.y = P.y + 1;
        d = at(data, P, S) - at(data, q, S);
        diff.y += d * d;

        q.y = P.y; q.z = P.z + 1;
        d = at(data, P, S) - at(data, q, S);
        diff.z += d * d;
    }

    return diff.x + diff.y + diff.z;
}

__global__
void init_clusters(const float* data,
                   float* centers,
                   const int n_clusters,
                   const int n_features,
                   const int3 sp_grid,
                   const int3 sp_shape,
                   const int3 im_shape)
{
    long lidx = threadIdx.x + (blockIdx.x * blockDim.x);

    if ( lidx >= n_clusters ) {
        return;
    }

    int3 idx;
    int plane = sp_grid.y * sp_grid.x;
    idx.z = lidx / plane;
    int aux = lidx % plane;
    idx.y = aux / sp_grid.x;
    idx.x = aux % sp_grid.x;

    int3 jdx;
    int volume_linear_idx = lidx;
    jdx.z =  volume_linear_idx / (sp_grid.x * sp_grid.y);
    int plane_linear_idx = volume_linear_idx - jdx.z * sp_grid.x * sp_grid.y;
    jdx.y = plane_linear_idx / sp_grid.x;
    jdx.x = plane_linear_idx % sp_grid.x;

    int4 p, q, r;
    p.z = r.z = idx.z * sp_shape.z + sp_shape.z / 2;
    p.y = r.y = idx.y * sp_shape.y + sp_shape.y / 2;
    p.x = r.x = idx.x * sp_shape.x + sp_shape.x / 2;



    int shift = n_features + 3;
    centers[lidx * shift + n_features + 0] = r.z;
    centers[lidx * shift + n_features + 1] = r.y;
    centers[lidx * shift + n_features + 2] = r.x;
}


__global__
void expectation(const float* data,
                 const float* centers,
                 unsigned int* labels,
                 const float m, const float S,
                 const int n_clusters,
                 const int n_features,
                 const float3 spacing,
                 const int3 sp_grid,
                 const int3 sp_shape,
                 const int3 im_shape)
{
    int4 idx, p, q;
    long gidx = threadIdx.x + (blockIdx.x * blockDim.x);

    if ( gidx >= im_shape.x * im_shape.y * im_shape.z ) {
        return;
    }

    // linear index to 3D pixel index transformation
    int plane = im_shape.y * im_shape.x;
    int aux = gidx % plane;
    idx.z = gidx / plane;
    idx.y = aux / im_shape.x;
    idx.x = aux % im_shape.x;

    // approx center grid positoin
    p.z = __max(0, __min(idx.z / sp_shape.z, sp_grid.z - 1));
    p.y = __max(0, __min(idx.y / sp_shape.y, sp_grid.y - 1));
    p.x = __max(0, __min(idx.x / sp_shape.x, sp_grid.x - 1));

    float min_d = DLIMIT, d, dist, adiff, pdiff;
    int R = 2, cshift = n_features + 3;
    long cidx, ridx = 0;

    for ( int k = -R; k <= R; k++ ) {
        q.z = p.z + k;
        if ( q.z < 0 || q.z >= sp_grid.z ) {continue;}

        for ( int i = -R; i <= R; i++ ) {
            q.y = p.y + i;
            if ( q.y < 0 || q.y >= sp_grid.y ) {continue;}

            for ( int j = -R; j <= R; j++ ) {
                q.x = p.x + j;
                if ( q.x < 0 || q.x >= sp_grid.x ) {continue;}

                cidx = q.z * sp_grid.y * sp_grid.x + q.y * sp_grid.x + q.x;

                if ( centers[cidx * cshift] == DLIMIT ) {
                    continue;
                }

                // Appearance diff
                adiff = 0;
                for ( int w = 0; w < n_features; w++ ) {
                    idx.w = w;
                    d = at(data, idx, im_shape) - centers[cidx * cshift + w];
                    adiff += d * d;
                }

                // Position diff
                float3 pd;
                pd.z = (idx.z - centers[cidx * cshift + n_features + 0]) * spacing.z;
                pd.y = (idx.y - centers[cidx * cshift + n_features + 1]) * spacing.y;
                pd.x = (idx.x - centers[cidx * cshift + n_features + 2]) * spacing.x;
                pdiff = pd.z * pd.z + pd.y * pd.y + pd.x * pd.x;
                dist = adiff / (m * m * n_features * n_features) + pdiff / (S * S);

                // Wrapup
                if ( dist < min_d ) {
                    min_d = dist;
                    ridx = cidx;
                }
            }
        }
    }

    labels[gidx] = ridx + 1;
}


__global__
void maximization(const float* data,
                  const unsigned int* labels,
                  float* centers,
                  int n_clusters,
                  int n_features,
                  const int3 sp_grid,
                  const int3 sp_shape,
                  const int3 im_shape)
{
    long lidx = threadIdx.x + (blockIdx.x * blockDim.x);

    if ( lidx >= n_clusters ) {
        return;
    }

    long cshift = n_features + 3;
    int3 cidx;
    cidx.z = (int) centers[lidx * cshift + n_features + 0];
    cidx.y = (int) centers[lidx * cshift + n_features + 1];
    cidx.x = (int) centers[lidx * cshift + n_features + 2];

    float ratio = 2.0f;

    int3 from;
    from.z = __max(cidx.z - sp_shape.z * ratio, 0);
    from.y = __max(cidx.y - sp_shape.y * ratio, 0);
    from.x = __max(cidx.x - sp_shape.x * ratio, 0);

    int3 to;
    to.z = __min(cidx.z + sp_shape.z * ratio, im_shape.z);
    to.y = __min(cidx.y + sp_shape.y * ratio, im_shape.y);
    to.x = __min(cidx.x + sp_shape.x * ratio, im_shape.x);

    int4 p;

    float* f = new float[cshift];
    for ( int k = 0; k < cshift; k++ ) {f[k] = 0;}
    long count = 0, offset, s2d = im_shape.x * im_shape.y;

    for ( p.z = from.z; p.z < to.z; p.z++ ) {
        for ( p.y = from.y; p.y < to.y; p.y++ ) {
            for ( p.x = from.x; p.x < to.x; p.x++ ) {
                offset = p.z * s2d + p.y * im_shape.x + p.x;

                if ( labels[offset] == lidx + 1 ) {
                    for ( int w = 0; w < n_features; w++ ) {
                        p.w = w;
                        f[w] += at(data, p, im_shape);
                    }
                    f[n_features + 0] += p.z;
                    f[n_features + 1] += p.y;
                    f[n_features + 2] += p.x;

                    count += 1;
                }
            }
        }
    }

    if ( count > 0 ) {
        for ( int w = 0; w < cshift; w++ ) {
            centers[lidx * cshift + w] = f[w] / count;
        }
    } else {
        centers[lidx * cshift] = DLIMIT;
    }

    delete[] f;
}