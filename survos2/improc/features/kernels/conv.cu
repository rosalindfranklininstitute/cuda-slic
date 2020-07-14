

texture<float, 3, cudaReadModeElementType> texSrc, texK;


__global__
void conv3d_tex(float* out, const int3 im_shape, const int3 k_radius)
{
    long x = blockIdx.x * blockDim.x + threadIdx.x;
    long y = blockIdx.y * blockDim.y + threadIdx.y;
    long z = blockIdx.z * blockDim.z + threadIdx.z;

    if ( x >= im_shape.x || y >= im_shape.y || z >= im_shape.z ) {
        return;
    }

    float sum = 0;
    long i, j, k;
    long px, py, pz;

    for ( k = -k_radius.z; k <= k_radius.z; k++ ) {
        pz = z + k;
        for ( i = -k_radius.y; i <= k_radius.y; i++ ) {
            py = y + i;
            for ( j = -k_radius.x; j <= k_radius.x; j++ ) {
                px = x + j;
                sum += tex3D(texSrc,
                             px + 0.5, py + 0.5, pz + 0.5) * \
                       tex3D(texK,
                             j + k_radius.x + 0.5, \
                             i + k_radius.y + 0.5, \
                             k + k_radius.z + 0.5);
            }
        }
    }

    out[z * im_shape.y * im_shape.x + y * im_shape.x + x] = sum;
}


__global__
void conv3d_axis0(const float* data, const float* kernel, float* out,
                const int3 im_shape, int k_radius)
{
    long x = blockIdx.x * blockDim.x + threadIdx.x;
    long y = blockIdx.y * blockDim.y + threadIdx.y;
    long z = blockIdx.z * blockDim.z + threadIdx.z;
    long s2d = im_shape.y * im_shape.x;

    int kshape = 2 * k_radius + 1;

    extern __shared__ float skernel[];
    long lidx = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

    if ( lidx < kshape ) {
        skernel[lidx] = kernel[lidx];
    }

    __syncthreads();

    if ( x >= im_shape.x || y >= im_shape.y || z >= im_shape.z ) {
        return;
    }

    float sum = 0;
    int k;
    long pz;

    for ( k = -k_radius; k <= k_radius; k++ ) {
        pz = z + k;
        if ( pz < 0 ) pz = -pz;
        if ( pz >= im_shape.z ) pz = im_shape.z - (pz - im_shape.z) - 1;
        if ( pz < 0 ) continue;
        sum += data[pz * s2d + y * im_shape.x + x] * skernel[k + k_radius];
    }

    out[z * s2d + y * im_shape.x + x] = sum;
}

__global__
void conv3d_axis1(const float* data, const float* kernel, float* out,
                const int3 im_shape, int k_radius)
{
    long x = blockIdx.x * blockDim.x + threadIdx.x;
    long y = blockIdx.y * blockDim.y + threadIdx.y;
    long z = blockIdx.z * blockDim.z + threadIdx.z;
    long s2d = im_shape.y * im_shape.x;

    int kshape = 2 * k_radius + 1;

    extern __shared__ float skernel[];
    long lidx = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

    if ( lidx < kshape ) {
        skernel[lidx] = kernel[lidx];
    }

    __syncthreads();

    if ( x >= im_shape.x || y >= im_shape.y || z >= im_shape.z ) {
        return;
    }

    float sum = 0;
    int k;
    long py;

    for ( k = -k_radius; k <= k_radius; k++ ) {
        py = y + k;
        if ( py < 0 ) py = -py;
        if ( py >= im_shape.y ) py = im_shape.y - (py - im_shape.y) - 1;
        if ( py < 0 ) continue;
        sum += data[z * s2d + py * im_shape.x + x] * skernel[k + k_radius];
    }

    out[z * s2d + y * im_shape.x + x] = sum;
}


__global__
void conv3d_axis2(const float* data, const float* kernel, float* out,
                const int3 im_shape, int k_radius)
{
    long x = blockIdx.x * blockDim.x + threadIdx.x;
    long y = blockIdx.y * blockDim.y + threadIdx.y;
    long z = blockIdx.z * blockDim.z + threadIdx.z;
    long s2d = im_shape.y * im_shape.x;

    int kshape = 2 * k_radius + 1;

    extern __shared__ float skernel[];
    long lidx = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

    if ( lidx < kshape ) {
        skernel[lidx] = kernel[lidx];
    }

    __syncthreads();

    if ( x >= im_shape.x || y >= im_shape.y || z >= im_shape.z ) {
        return;
    }

    float sum = 0;
    int k;
    long px;

    for ( k = -k_radius; k <= k_radius; k++ ) {
        px = x + k;
        if ( px < 0 ) px = -px;
        if ( px >= im_shape.x ) px = im_shape.x - (px - im_shape.x) - 1;
        if ( px < 0 ) continue;
        sum += data[z * s2d + y * im_shape.x + px] * skernel[k + k_radius];
    }

    out[z * s2d + y * im_shape.x + x] = sum;
}
