# cuda-slic: A CUDA implementation of the SLIC Superpixel algorithm

## SLIC
SLIC stands for __simple linear iteraticve clustering__. SLIC uses
tiled k-means clustering to segment an input image into a set
of superpixels/supervoxels.

This library was designed to segment large 3D images coming from different
detectors at the [Diamond Light Source](https://diamond.ac.uk). These Images
can be upto 500 GB so using a serial CPU code is out of the question.

To speed up processing we use GPU acceleratoin to achieve great speed
improvements over alternative implementations. `cuda-slic` borrows its API
from `skimage.segmentation.slic`.

## Install
```bash
pip install cuda-slic
```
`cuda-slic` uses the pycuda which has the following non-python
build/run dependencies:
1. gcc and g++/gcc-c++ on Linux.
2. the cudatoolkit for linking with cuda libraries.
3. the nvcc compiler. Ships with newer cudatoolkit versions.
See [pycuda docs](https://wiki.tiker.net/PyCuda/Installation/) for 
installation instructions.

## Usage
```python
from cuda_slic import slic
from skimage import data

img = data.astronaut() # 2D RGB image
labels = slic(img, n_segments=100, compactness=10)

# To segment 3D gray scale
vol = data.binary_blobs(length=50, ndim=3, seed=2)
labels = slic(vol, n_segments=100, multichannel=False, compactness=0.1)

# 3D muli-channel
vol = data.binary_blobs(length=33, ndim=4, seed=2)
labels = slic(vol, n_segments=100, multichannel=True, compactness=0.1)
```

# Development
##### Prerequisites:
1. gcc and g++/gcc-c++ installed and available on PATH.
2. cudatoolkit installed and CUDA_HOME pointing to its location.
3. `nvcc` compiler. Ships with recent versions of cudatoolkit.

## Dependency Management

We use `conda` as a dependency installer and virtual env manager.
A development environment can be created with
```bash
conda env create -f environment.yml
```
now you can activate the virtual env with `conda activate cuda-slic`, to deactivate use `conda deactivate`.
To add a dependency, add it to the [environment.yml](environment.yml) file, then you can run
```bash
conda env update -f environment.yml
```

## Tests
In the [notebooks](notebooks) folder there are Jupyter notebooks
where the segmentation algorithm can be visually inspected.

Our unit-testing framework of choice is [Py.test](https://docs.pytest.org/en/latest/). The unit-tests can be run with
```bash
conda activate cuda-slic
pytest
```
