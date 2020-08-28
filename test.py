import cupy as cp


kernel = """
extern "C" {
    constexpr __device__ int3 ii{1,2,3};

    __global__
    void init_vec() {
    const int3 i = ii;
    printf("i.x == %d", i.x);
    }

}
"""
mod = cp.RawModule(code=kernel, options=("-std=c++14",))
vv = mod.get_function('vv')
vv(
    (1,),
    (1,),
    (),
)
