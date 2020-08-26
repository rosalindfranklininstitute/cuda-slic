import cupy as cp


kernel = """
extern "C" {
    __global__
    void init_vec() {
    int3 i = make_int3(1,2,3);
    printf("i.x == %d", i.x);
    }

}
"""
vv = cp.RawKernel(kernel, "init_vec")
vv(
    (1,),
    (1,),
    (),
)
