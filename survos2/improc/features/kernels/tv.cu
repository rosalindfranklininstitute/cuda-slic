

__device__
float divergence(const float* pz, const float* py, const float* px,
                 long idx, const int3& p, long size2d, const int3& shape)
{
    float _div = 0.0f;
    long _idx;
    if ( p.z - 1 >= 0 ) {
        _idx = (p.z - 1) * size2d + p.y * shape.x + p.x;
        _div += (pz[idx] - pz[_idx]);
    } else {
        _div += pz[idx];
    }

    if ( p.y - 1 >= 0 ) {
        _idx = p.z * size2d + (p.y - 1) * shape.x + p.x;
        _div += (py[idx] - py[_idx]);
    } else {
        _div += py[idx];
    }

    if ( p.x - 1 >= 0 ) {
        _idx = p.z * size2d + p.y * shape.x + (p.x - 1);
        _div += (px[idx] - px[_idx]);
    } else {
        _div += px[idx];
    }

    return _div;
}


__device__
void gradient(const float* u, float3& grad, long idx,
              const int3& p, long size2d, const int3& shape)
{
    float uidx = u[idx];

    if ( p.z + 1 < shape.z ) {
        grad.z = (u[(p.z+1)*size2d + p.y*shape.x + p.x] - uidx);
    } else {
        grad.z = 0;
    }

    if ( p.y + 1 < shape.y ) {
        grad.y = (u[p.z*size2d + (p.y+1)*shape.x + p.x] - uidx);
    } else {
        grad.y = 0;
    }

    if ( p.x + 1 < shape.x ) {
        grad.x = (u[p.z*size2d + p.y*shape.x + (p.x+1)] - uidx);
    } else {
        grad.x = 0;
    }
}


__global__
void update_u(const float* f, const float* pz, const float* py, const float* px,
              float* u, float tau, float lambda, const int3 shape)
{
    long idx = blockDim.x * blockIdx.x + threadIdx.x;
    long plane = shape.y * shape.x;

    if ( idx >= plane * shape.z )
        return;

    long t = idx % plane;
    int3 pos;
    pos.z = idx / plane;
    pos.y = t / shape.x;
    pos.x = t % shape.x;

    float _div = divergence(pz, py, px, idx, pos, plane, shape);

    float r = u[idx] * (1.0f - tau) + tau * (f[idx] + lambda * _div);

    u[idx] = r;
}



__global__
void update_p(const float* u, float* pz, float* py, float* px,
              float tau, const int3 shape)
{
    long idx = blockDim.x * blockIdx.x + threadIdx.x;
    long plane = shape.y * shape.x;

    if ( idx >= plane * shape.z )
        return;

    long t = idx % plane;
    int3 pos;
    pos.z = idx / plane;
    pos.y = t / shape.x;
    pos.x = t % shape.x;

    float3 grad, q;
    gradient(u, grad, idx, pos, plane, shape);

    q.z = pz[idx] + tau * grad.z;
    q.y = py[idx] + tau * grad.y;
    q.x = px[idx] + tau * grad.x;

    float n = q.z * q.z + q.y * q.y + q.x * q.x;

    float norm = fmaxf(1.0f, sqrtf(fmaxf(0, n)));

    pz[idx] = q.z / norm;
    py[idx] = q.y / norm;
    px[idx] = q.x / norm;
}