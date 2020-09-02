/*
Indexing:
idx = pixel/voxel index in cartesian coordinates
cidx = center index in cartesian coordinates

linear_idx = pixel/voxel index in flat array
linear_cidx = center index in flat array

Center Stride:
c_stride = number_of_features + image_dimention
center_addr = linear_cidx * c_stride

Image Stride:
z_stride = image_shape.y * image_shape.x
y_stride = image_shape.x
x_stride = 1


Transformations 3D:
linear_idx = idx.z * z_stride + idx.y * y_stride + idx.x
pixel_addr = linear_idx * n_features

idx.z = linear_idx / z_stride
plane_idx = linear_idx % z_stride
idx.y = plane_idx / y_stride
idx.x = plane_idx % y_stride

Transformations 2D:
linear_idx = idx.y * y_stride + idx.x
pixel_addr = linear_idx * n_features

idx.y = linear_idx / y_stride
idx.x = linear_idx % y_stride

*/

#define DLIMIT 99999999
#define N_FEATURES {{ n_features }}
#define N_CLUSTERS {{ n_clusters }}
// #define MM {{ m }}f
// #define SS {{ S }}f

#define __min(a, b) (((a) < (b)) ? (a) : (b))
#define __max(a, b) (((a) >= (b)) ? (a) : (b))

__constant__ __device__ float SS = {{ S }}f;

__constant__ __device__ int3 SP_GRID 
{
    {{ sp_grid[0] }},
    {{ sp_grid[1] }},
    {{ sp_grid[2] }}
};

__constant__ __device__ int3 SP_SHAPE
{ 
    {{ sp_shape[0] }},
    {{ sp_shape[1] }},
    {{ sp_shape[2] }}
};

__constant__ __device__ int3 IM_SHAPE
{ 
    {{ im_shape[0] }},
    {{ im_shape[1] }},
    {{ im_shape[2] }}
};

__constant__ __device__ float3 SPACING
{ 
    {{ spacing[0] }},
    {{ spacing[1] }},
    {{ spacing[2] }}
};


__device__
float slic_distance(const int3 idx,
                    const float* pixel,
                    const long center_addr, const float* centers,
                    const float3 spacing)
{
    // Color diff
    float color_diff = 0;
    for ( int w = 0; w < N_FEATURES; w++ ) {
        float d = pixel[w] - centers[center_addr + w];
        color_diff += d * d;
    }

    // Position diff
    float3 pd;
    pd.z = (idx.z - centers[center_addr + N_FEATURES + 0]) * spacing.z;
    pd.y = (idx.y - centers[center_addr + N_FEATURES + 1]) * spacing.y;
    pd.x = (idx.x - centers[center_addr + N_FEATURES + 2]) * spacing.x;

    float position_diff = pd.z * pd.z + pd.y * pd.y + pd.x * pd.x;
    float dist = color_diff + position_diff / (SS * SS);
    return dist;
}


__global__
void init_clusters(const float* data,
                   float* centers
                )
{
    const long linear_cidx = threadIdx.x + (blockIdx.x * blockDim.x);

    if ( linear_cidx >= N_CLUSTERS ) {
        return;
    }
    const int3 sp_grid = SP_GRID;
    const int3 sp_shape = SP_SHAPE;

    // calculating the (0,0,0) index of each superpixel block
    // using linear to cartesian index transformation
    int3 cidx;
    int plane_size = sp_grid.y * sp_grid.x;
    cidx.z = linear_cidx / plane_size;
    int plane_idx = linear_cidx % plane_size;
    cidx.y = plane_idx / sp_grid.x;
    cidx.x = plane_idx % sp_grid.x;

    // centering index into middle of suprepixel block
    cidx.z = cidx.z * sp_shape.z + sp_shape.z / 2;
    cidx.y = cidx.y * sp_shape.y + sp_shape.y / 2;
    cidx.x = cidx.x * sp_shape.x + sp_shape.x / 2;

    //saving cluster center positions
    // note: the color is not initialized, but is kept at zero.
    const int c_stride = N_FEATURES + 3;
    centers[linear_cidx * c_stride + N_FEATURES + 0] = cidx.z;
    centers[linear_cidx * c_stride + N_FEATURES + 1] = cidx.y;
    centers[linear_cidx * c_stride + N_FEATURES + 2] = cidx.x;
}


__global__
void expectation(const float* data,
                 const float* centers,
                 unsigned int* labels
                )
{
    const long linear_idx = threadIdx.x + (blockIdx.x * blockDim.x);

    const int3 sp_grid = SP_GRID;
    const int3 sp_shape = SP_SHAPE;
    const int3 im_shape = IM_SHAPE;
    const float3 spacing = SPACING;

    if ( linear_idx >= im_shape.x * im_shape.y * im_shape.z ) {
        return;
    }

    const long pixel_addr = linear_idx * N_FEATURES;
    float pixel[N_FEATURES];
    for ( int w = 0; w < N_FEATURES; w++ ) {
        pixel[w] = data[pixel_addr + w];
    }


    // linear to cartesian index transformation per pixel
    int3 idx;
    int plane_size = im_shape.y * im_shape.x;
    idx.z = linear_idx / plane_size;
    int plane_idx = linear_idx % plane_size;
    idx.y = plane_idx / im_shape.x;
    idx.x = plane_idx % im_shape.x;

    int4 cidx, iter_cidx;
    long iter_linear_cidx;
    long closest_linear_cidx = 0;

    // approx center grid positoin
    cidx.z = __max(0, __min(idx.z / sp_shape.z, sp_grid.z - 1));
    cidx.y = __max(0, __min(idx.y / sp_shape.y, sp_grid.y - 1));
    cidx.x = __max(0, __min(idx.x / sp_shape.x, sp_grid.x - 1));

    float minimum_distance = DLIMIT;
    const int c_stride = N_FEATURES + 3;

    const int R = 2;
    for ( int k = -R; k <= R; k++ ) {
        for ( int j = -R; j <= R; j++ ) {
            for ( int i = -R; i <= R; i++ ) {
                iter_cidx.z = cidx.z + k;
                iter_cidx.y = cidx.y + j;
                iter_cidx.x = cidx.x + i;

                if ( iter_cidx.y < 0 || iter_cidx.y >= sp_grid.y || 
                     iter_cidx.z < 0 || iter_cidx.z >= sp_grid.z ||
                     iter_cidx.x < 0 || iter_cidx.x >= sp_grid.x ) {continue;}

                iter_linear_cidx = iter_cidx.z * sp_grid.y * sp_grid.x +
                                   iter_cidx.y * sp_grid.x +
                                   iter_cidx.x;
                long iter_center_addr = iter_linear_cidx * c_stride;

                if ( centers[iter_center_addr] == DLIMIT ) {
                    continue;
                }

                float dist = slic_distance(idx,
                                           pixel,
                                           iter_center_addr, centers,
                                           spacing);

                // Wrapup
                if ( dist < minimum_distance ) {
                    minimum_distance = dist;
                    closest_linear_cidx = iter_linear_cidx;
                }
            }
        }
    }

    labels[linear_idx] = closest_linear_cidx + 1;
}


__global__
void maximization(const float* data,
                  const unsigned int* labels,
                  float* centers
                  )
{
    const long linear_cidx = threadIdx.x + (blockIdx.x * blockDim.x);
    const int c_stride = N_FEATURES + 3;
    const long center_addr = linear_cidx * c_stride;

    const int3 sp_grid = SP_GRID;
    const int3 sp_shape = SP_SHAPE;
    const int3 im_shape = IM_SHAPE;
    const float3 spacing = SPACING;

    if ( linear_cidx >= N_CLUSTERS ) { return; }

    int3 cidx;
    cidx.z = (int) centers[center_addr + N_FEATURES + 0];
    cidx.y = (int) centers[center_addr + N_FEATURES + 1];
    cidx.x = (int) centers[center_addr + N_FEATURES + 2];

    float ratio = 2.0f;

    int3 from;
    from.z = __max(cidx.z - sp_shape.z * ratio, 0);
    from.y = __max(cidx.y - sp_shape.y * ratio, 0);
    from.x = __max(cidx.x - sp_shape.x * ratio, 0);

    int3 to;
    to.z = __min(cidx.z + sp_shape.z * ratio, im_shape.z);
    to.y = __min(cidx.y + sp_shape.y * ratio, im_shape.y);
    to.x = __min(cidx.x + sp_shape.x * ratio, im_shape.x);


    float f[c_stride];
    for ( int k = 0; k < c_stride; k++ ) {f[k] = 0;}

    long z_stride = im_shape.x * im_shape.y;
    long y_stride = im_shape.x;

    long count = 0;
    int3 p;
    for ( p.z = from.z; p.z < to.z; p.z++ ) {
        for ( p.y = from.y; p.y < to.y; p.y++ ) {
            for ( p.x = from.x; p.x < to.x; p.x++ ) {
                long linear_idx = p.z * z_stride + p.y * y_stride + p.x;
                long pixel_addr = linear_idx * N_FEATURES;

                if ( labels[linear_idx] == linear_cidx + 1 ) {
                    for ( int w = 0; w < N_FEATURES; w++ ) {
                        f[w] += data[pixel_addr + w];
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
        for ( int w = 0; w < c_stride; w++ ) {
            centers[center_addr + w] = f[w] / count;
        }
    } else {
        centers[center_addr] = DLIMIT;
    }
}