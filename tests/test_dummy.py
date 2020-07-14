import numpy as np
import h5py as h5

def func(x):
    return x + 1

def test_answer():
    assert func(3) == 4
