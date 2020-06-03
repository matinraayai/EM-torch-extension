# EM Torch Extension
## What is this?
This package includes Pytorch tensor kernel operations written in 
C++/CUDA that are used in other EM packages in the Rhoana pipeline.
## EM Packages Currently Using EM Torch Extension:
- [EM-preprocess](https://github.com/donglaiw/EM-preprocess)

## How to Install
### Pre-requisites:
1. A C++ compiler ```gcc > 7```.
2. An NVIDIA CUDA compiler ```nvcc > 9.2```.
3. ```pytorch==1.5.0```
4. The ```setup.py``` finds the latest packages automatically:

``python setup.py install``