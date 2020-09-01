import cupy as cp


kernel = """
extern "C" {

    __global__
    void init_vec(int3* ii)
    {
        int3 i = *ii;
        printf("i.x == %d", i.x);
        printf("i.y == %d", i.y);
        printf("i.z == %d", i.z);
    }

}
"""
mod = cp.RawModule(code=kernel, options=("-std=c++14",))
vv = mod.get_function("init_vec")
ii = cp.asarray([1, 2, 3], dtype=cp.int32)
vv(
    (1,), (1,), (ii),
)
