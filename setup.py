from setuptools import find_packages, setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="cuda_slic",
    version="0.0.1a2",
    python_requires=">=3.5",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "numpy",
        "jinja2",
        "scikit-image",
        "pycuda>=2019.1.2",
    ],
    # metadata to display on PyPI
    author="Omar Elamin",
    author_email="omar.elamin@diamond.ac.uk",
    description="CUDA implementation of the SLIC segmentaion algorithm.",
    keywords="segmentation fast cuda slic clustering kmeans",
    url="https://gitlab.stfc.ac.uk/RosalindFranklinInstitute/cuda-slic",  # project home page, if any
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    options={"bdist_wheel": {"universal": "1"}},
    # could also include long_description, download_url, etc.
    long_description=long_description,
    long_description_content_type="text/markdown",
)
