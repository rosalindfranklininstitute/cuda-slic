#define DLIMIT 99999999
#define N_FEATURES {{ n_features }}


#define __min(a, b) (((a) < (b)) ? (a) : (b))
#define __max(a, b) (((a) >= (b)) ? (a) : (b))


__device__
float at(const float* data, const int4& P, const int3& S) {
    long s2d = S.y * S.x, s3d = S.z * S.y * S.x;
    return data[P.w * s3d + P.z * s2d + P.y * S.x + P.x];
}

__device__
float slic_distance(const int4& idx, const long center_idx,
                    const float* data, const float* centers,
                    const int3& im_shape);

__global__
void init_clusters(const float* data,
                   float* centers,
                   const int n_clusters,
                   const int3 sp_grid,
                   const int3 sp_shape,
                   const int3 im_shape)
{
    const long linear_idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if ( linear_idx >= n_clusters ) {
        return;
    }

    // calculating the (0,0,0) index of each superpixel block
    // using linear to cartesian index transformation
    int3 idx;
    int plane_size = sp_grid.y * sp_grid.x;
    idx.z = linear_idx / plane_size;
    int plane_idx = linear_idx % plane_size;
    idx.y = plane_idx / sp_grid.x;
    idx.x = plane_idx % sp_grid.x;

    // centering index to get better spacing
    idx.z = idx.z * sp_shape.z + sp_shape.z / 2;
    idx.y = idx.y * sp_shape.y + sp_shape.y / 2;
    idx.x = idx.x * sp_shape.x + sp_shape.x / 2;

    //saving cluster center positions
    // note: the color is not initialized, but is kept at zero.
    const int stride = N_FEATURES + 3;
    centers[linear_idx * stride + N_FEATURES + 0] = idx.z;
    centers[linear_idx * stride + N_FEATURES + 1] = idx.y;
    centers[linear_idx * stride + N_FEATURES + 2] = idx.x;
}


__global__
void expectation(const float* data,
                 const float* centers,
                 unsigned int* labels,
                 const float m, const float S,
                 const int n_clusters,
                 const float3 spacing,
                 const int3 sp_grid,
                 const int3 sp_shape,
                 const int3 im_shape)
{
    const long gidx = threadIdx.x + (blockIdx.x * blockDim.x);

    if ( gidx >= im_shape.x * im_shape.y * im_shape.z ) {
        return;
    }

    // linear index to 3D pixel index transformation
    int4 idx;
    int plane = im_shape.y * im_shape.x;
    int aux = gidx % plane;
    idx.z = gidx / plane;
    idx.y = aux / im_shape.x;
    idx.x = aux % im_shape.x;

    // approx center grid positoin
    int4 p, q;
    p.z = __max(0, __min(idx.z / sp_shape.z, sp_grid.z - 1));
    p.y = __max(0, __min(idx.y / sp_shape.y, sp_grid.y - 1));
    p.x = __max(0, __min(idx.x / sp_shape.x, sp_grid.x - 1));

    float min_d = DLIMIT;
    const int cshift = N_FEATURES + 3;
    long ridx = 0, cidx;

    const int R = 2;
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
                float adiff = 0;
                for ( int w = 0; w < N_FEATURES; w++ ) {
                    idx.w = w;
                    float d = data[gidx + w] - centers[cidx * cshift + w];
                    adiff += d * d;
                }

                // Position diff
                float3 pd;
                pd.z = (idx.z - centers[cidx * cshift + N_FEATURES + 0]) * spacing.z;
                pd.y = (idx.y - centers[cidx * cshift + N_FEATURES + 1]) * spacing.y;
                pd.x = (idx.x - centers[cidx * cshift + N_FEATURES + 2]) * spacing.x;
                float pdiff = pd.z * pd.z + pd.y * pd.y + pd.x * pd.x;
                float dist = adiff / (m * m * N_FEATURES * N_FEATURES) + pdiff / (S * S);

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
                  const int3 sp_grid,
                  const int3 sp_shape,
                  const int3 im_shape)
{
    long lidx = threadIdx.x + (blockIdx.x * blockDim.x);

    if ( lidx >= n_clusters ) {
        return;
    }

    const int cshift = N_FEATURES + 3;
    int3 cidx;
    cidx.z = (int) centers[lidx * cshift + N_FEATURES + 0];
    cidx.y = (int) centers[lidx * cshift + N_FEATURES + 1];
    cidx.x = (int) centers[lidx * cshift + N_FEATURES + 2];

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

    float f[cshift];
    for ( int k = 0; k < cshift; k++ ) {f[k] = 0;}
    long count = 0, offset, s2d = im_shape.x * im_shape.y;

    for ( p.z = from.z; p.z < to.z; p.z++ ) {
        for ( p.y = from.y; p.y < to.y; p.y++ ) {
            for ( p.x = from.x; p.x < to.x; p.x++ ) {
                offset = p.z * s2d + p.y * im_shape.x + p.x;

                if ( labels[offset] == lidx + 1 ) {
                    for ( int w = 0; w < N_FEATURES; w++ ) {
                        p.w = w;
                        f[w] += at(data, p, im_shape);
                    }
                    f[N_FEATURES + 0] += p.z;
                    f[N_FEATURES + 1] += p.y;
                    f[N_FEATURES + 2] += p.x;

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
}