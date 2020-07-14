/*
 * Connected Components in the GPU
 * Paper Source: An Optimized Union-Find Algorithm for Connected
 *               Components Labeling Using GPUs
 * Adapted from: https://github.com/victormatheus/CCL-GPU
 * Modified by: Imanol Luengo <imaluengo@gmail.com>
 */

typedef unsigned int uint32;

#define MAX_UINT32 0xFFFFFFFF


__device__
uint32 find(uint32* buf, uint32 x) {
    uint32 p = x;
    while ( x != buf[x] ) {
        x = buf[x];
    }
    buf[p] = x; // fast linking
    return x;
}


__device__
void findAndUnion(uint32* buf, uint32 g1, uint32 g2) {
    bool done;
    uint32 old;

    do {
        g1 = find(buf, g1);
        g2 = find(buf, g2);

        if (g1 < g2) {
            old = atomicMin(&buf[g2], g1);
            done = (old == g2);
            g2 = old;
        } else if (g2 < g1) {
            old = atomicMin(&buf[g1], g2);
            done = (old == g1);
            g1 = old;
        } else {
            done = true;
        }
    } while ( !done );
}


__global__
void uf_local(const uint32* in, uint32* out, int3 shape) {
    int3 p;
    p.z = blockIdx.z * blockDim.z + threadIdx.z;
    p.y = blockIdx.y * blockDim.y + threadIdx.y;
    p.x = blockIdx.x * blockDim.x + threadIdx.x;

    long image_plane = (shape.y * shape.x);
    long block_plane = (blockDim.y * blockDim.x);
    uint32 gidx = p.z * image_plane + p.y * shape.x + p.x;
    uint32 bidx = threadIdx.z * block_plane + \
                  threadIdx.y * blockDim.x + \
                  threadIdx.x;

    long bsize = blockDim.z * blockDim.y * blockDim.x;
    extern __shared__ uint32 s_buffer[];

    bool in_limits = p.z < shape.z && p.y < shape.y && p.x < shape.x;

    s_buffer[bidx] = bidx;
    s_buffer[bsize + bidx] = in_limits? in[p.z * image_plane + p.y * shape.x + p.x] : 0;

    __syncthreads();

    if ( !in_limits ) {return;}

    uint32 v = s_buffer[bsize + bidx];

    if ( threadIdx.x > 0 && s_buffer[bsize + bidx - 1] == v ) {
        findAndUnion(s_buffer, bidx, bidx - 1);
    }

    __syncthreads();

    if ( threadIdx.y > 0 && s_buffer[bsize + bidx - blockDim.x] == v ) {
        findAndUnion(s_buffer, bidx, bidx - blockDim.x);
    }

    __syncthreads();

    if ( threadIdx.z > 0 && s_buffer[bsize + bidx - block_plane] == v ) {
        findAndUnion(s_buffer, bidx, bidx - block_plane);
    }

    __syncthreads();

    uint32 f = find(s_buffer, bidx);
    uint32 aux = f % block_plane;
    uint32 fz = f / block_plane;
    uint32 fy = aux / blockDim.x;
    uint32 fx = aux % blockDim.x;

    out[gidx] = (blockIdx.z * blockDim.z + fz) * image_plane + \
                (blockIdx.y * blockDim.y + fy) * shape.x + \
                (blockIdx.x * blockDim.x + fx);
}


__global__
void uf_global(const uint32* in, uint32* out, int3 shape) {
    int3 p;
    p.z = blockIdx.z * blockDim.z + threadIdx.z;
    p.y = blockIdx.y * blockDim.y + threadIdx.y;
    p.x = blockIdx.x * blockDim.x + threadIdx.x;

    long image_plane = (shape.y * shape.x);
    uint32 gidx = p.z * image_plane + p.y * shape.x + p.x;

    if ( p.z >= shape.z || p.y >= shape.y || p.x >= shape.x ) {
        return;
    }

    uint32 v = in[gidx];

    if ( p.z > 0 && threadIdx.z == 0 && in[gidx - image_plane] == v ) {
        findAndUnion(out, gidx, gidx - image_plane);
    }

    if ( p.y > 0 && threadIdx.y == 0 && in[gidx - shape.x] == v ) {
        findAndUnion(out, gidx, gidx - shape.x);
    }

    if ( p.x > 0 && threadIdx.x == 0 && in[gidx - 1] == v ) {
        findAndUnion(out, gidx, gidx - 1);
    }

}


__global__
void uf_final(uint32* labels, int3 shape) {
    int3 p;
    p.z = blockIdx.z * blockDim.z + threadIdx.z;
    p.y = blockIdx.y * blockDim.y + threadIdx.y;
    p.x = blockIdx.x * blockDim.x + threadIdx.x;

    long gidx = p.z * shape.y * shape.x + p.y * shape.x + p.x;

    if ( p.z < shape.z && p.y < shape.y && p.x < shape.x ) {
        labels[gidx] = find(labels, gidx);
    }
}
