import numpy as np


int2 = np.dtype([("x", np.int32), ("y", np.int32)], align=True)
int3 = np.dtype(
    [("x", np.int32), ("y", np.int32), ("z", np.int32)], align=True
)

float2 = np.dtype([("x", np.float32), ("y", np.float32)], align=True)
float3 = np.dtype(
    [("x", np.float32), ("y", np.float32), ("z", np.float32)], align=True
)
