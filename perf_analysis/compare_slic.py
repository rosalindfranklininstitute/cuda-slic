import time

import numpy as np

from skimage import color, data, filters
from skimage.segmentation import slic

from cuda_slic.slic import slic3d


def time_func(func, *args, **kwargs):
    start = time.time()
    ret = func(*args, **kwargs)
    end = time.time()
    return end - start, ret


def test_compare_slics():
    ratios = []
    for _ in range(5):
        times = {"cuda_slic": [], "scikit_slic": []}
        n = 3
        side = 100
        for i in range(n):
            blob = data.binary_blobs(length=side, n_dim=3, seed=2)
            blob = np.float32(blob)
            n_segments = side ** 3 // 5 ** 3
            t1, _ = time_func(
                slic3d, blob, n_segments=n_segments, compactness=0.2
            )
            times["cuda_slic"].append(t1)
            t2, _ = time_func(
                slic,
                blob,
                n_segments=n_segments,
                compactness=0.2,
                start_label=1,
                max_iter=5,
                multichannel=False,
            )
            times["scikit_slic"].append(t2)

        cuda = times["cuda_slic"]
        sk = times["scikit_slic"]
        print(f"cuda_slic({side})", cuda)
        print(f"sk_slic({side})", sk)
        print("mean cuda = {}".format(sum(cuda) / len(cuda)))
        print("mean sk = {}".format(sum(sk) / len(sk)))
        ratio = sum(sk) / sum(cuda)
        ratios.append(ratio)
        # print("cuda times: {}".format(cuda))
        # print("skimage times: {}".format(sk))
        # print("ratio skimgae/cude = {}".format(ratio))
        # assert ratio > 5
    print("ratio skimgae/cude = {}".format(np.median(ratios)))


if __name__ == "__main__":
    test_compare_slics()
