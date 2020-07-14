

import os.path as op
import numpy as np

from survos2.model import Workspace, Dataset

tmp_path = '/scratch/olly/test_survos_datamodel'
wspath = 'test_survos_datamodel'
if Workspace.exists(tmp_path):
    Workspace.remove(tmp_path)
workspace = Workspace.create(tmp_path)

print("Generating data")
data = np.random.rand(300, 500, 500).astype(np.float32)
print("Adding data to workspace")
ds = workspace.add_data(data)

print("Creating datasets")
f1 = workspace.add_dataset('features/tv1', 'uint8')
l1 = workspace.add_dataset('annotations/level1', np.float32)
l2 = workspace.add_dataset('annotations/level2', '|i1')

s1 = workspace.add_session('imanol')
f2 = workspace.add_dataset('features/gauss', np.float32, session='imanol')

print("Showing some data")

print(data[50])
# print(f1[50])
# print(l1[50])
# print(l2[50])

print("Populating L1")

l1[50] = 5

# print(l1[50])

print("Generating L2")
data2 = np.random.randint(0, 10, size=data.shape)

print("Populating L2")
l2.load(data2)

# print(l2[50])

print("Generating F2")
data3 = np.random.rand(*data.shape)

print("Populating F2")
f2[...] = data3

# print(f2[50])

l1.set_metadata('labels', dict(label1=dict(name='Label 1', idx=0, color='#00FF00')))

data = Dataset(op.join(workspace.path, Dataset.__dsname__))
# print(data[50])

f1[20:30, :, 300:] = 1

# print(np.unique(f1[25]))
