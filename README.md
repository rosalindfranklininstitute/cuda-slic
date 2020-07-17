# cuda-slic: A CUDA implementation of the SLIC Superpixel algorithm

## SLIC
SLIC stands for __simple linear iteraticve clustering__.
SLIC uses windowed k-means clustering to segment an input array to a set of super-regions.
The K-means clustering algo is an embarrasingly parallelizable problem, making it ideally suited for GPU Acceleration.

## Dependency Management
We use `conda` as a dependency installer and virtual env manager. A development environment can be created with
```bash
conda env create -f environment.yml
```
now you can activate the virtual env with `conda activate gpu-slic`, to deactivate use `conda deactivate`.
To add a dependency, add it to the [environment.yml](environment.yml) file, then you can run
```bash
conda env update -f environment.yml
```

## Tests
in the [notebooks](notebooks) folder there are Jupyter notebooks where the clustering algos can be visually inspected.

Our unit-testing framework of choice is [Py.test](https://docs.pytest.org/en/latest/). The unit-tests can be run with
```bash
python -m pytest tests-unit
```
