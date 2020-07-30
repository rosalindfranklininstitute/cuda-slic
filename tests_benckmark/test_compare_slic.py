import time
import numpy as np

from survos2.improc.regions.slic import slic3d

from skimage import data, color, filters
from skimage.util import img_as_float32

from skimage.segmentation import slic

def time_func(func, *args, **kwargs):
    start = time.time()
    ret = func(*args, **kwargs)
    end = time.time()
    return end - start, ret

def test_compare_slics():
    ratios = []
    for _ in range(5):
        times = {'cuda_slic':[], "scikit_slic":[]}
        for i in [100]:
            blob = data.binary_blobs(length=i, n_dim=3, seed=2)
            blob = np.float32(blob)
            n_segments = i**3//5**3
            t1, _ = time_func(slic3d, blob, n_segments=n_segments, compactness=0.2)
            times["cuda_slic"].append(t1)
            t2, _ = time_func(slic, blob, n_segments=n_segments,
                              compactness=0.2, start_label=1,
                              max_iter=5, multichannel=False)
            times["scikit_slic"].append(t2)

        cuda = times["cuda_slic"] 
        sk = times["scikit_slic"]
        ratio = sum(sk)/sum(cuda)/len(sk) 
        ratios.append(ratio)
        # print("cuda times: {}".format(cuda))
        # print("skimage times: {}".format(sk))
        # print("ratio skimgae/cude = {}".format(ratio))
        # assert ratio > 5
    print("ratio skimgae/cude = {}".format(np.median(ratios)))
test_compare_slics()





