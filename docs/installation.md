---
title: Installation
---

# Installation

Prior to installing, have a glance through this guide and take note of the details for your platform.
We install and run Caffe on Ubuntu 14.04 and 12.04, OS X 10.10 / 10.9 / 10.8, and AWS.
The official Makefile and `Makefile.config` build are complemented by an automatic CMake build from the community.

- [Prerequisites](#prerequisites)
- [Compilation](#compilation)
- [Hardware](#hardware)
- Platforms: [Ubuntu guide](install_apt.html), [OS X guide](install_osx.html), and [RHEL / CentOS / Fedora guide](install_yum.html)

When updating Caffe, it's best to `make clean` before re-compiling.

## Prerequisites

Caffe has several dependencies.

* [CUDA](https://developer.nvidia.com/cuda-zone) is required for GPU mode.
    * library version 7.0 and the latest driver version are recommended, but 6.* is fine too
    * 5.5, and 5.0 are compatible but considered legacy
* [BLAS](http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) via ATLAS, MKL, or OpenBLAS.
* [Boost](http://www.boost.org/) >= 1.55
* [OpenCV](http://opencv.org/) >= 2.4 including 3.0
* `protobuf`, `glog`, `gflags`
* IO libraries `hdf5`, `leveldb`, `snappy`, `lmdb`

Pycaffe and Matcaffe interfaces have their own natural needs.

* For Python Caffe:  `Python 2.7` or `Python 3.3+`, `numpy (>= 1.7)`, boost-provided `boost.python`
* For MATLAB Caffe: MATLAB with the `mex` compiler.

**cuDNN Caffe**: for fastest operation Caffe is accelerated by drop-in integration of [NVIDIA cuDNN](https://developer.nvidia.com/cudnn). To speed up your Caffe models, install cuDNN then uncomment the `USE_CUDNN := 1` flag in `Makefile.config` when installing Caffe. Acceleration is automatic. For now cuDNN v1 is integrated but see [PR #1731](https://github.com/BVLC/caffe/pull/1731) for v2.

**CPU-only Caffe**: for cold-brewed CPU-only Caffe uncomment the `CPU_ONLY := 1` flag in `Makefile.config` to configure and build Caffe without CUDA. This is helpful for cloud or cluster deployment.

### CUDA and BLAS

Caffe requires the CUDA `nvcc` compiler to compile its GPU code and CUDA driver for GPU operation.
To install CUDA, go to the [NVIDIA CUDA website](https://developer.nvidia.com/cuda-downloads) and follow installation instructions there. Install the library and the latest standalone driver separately; the driver bundled with the library is usually out-of-date. **Warning!** The 331.* CUDA driver series has a critical performance issue: do not use it.

For best performance, Caffe can be accelerated by [NVIDIA cuDNN](https://developer.nvidia.com/cudnn). Register for free at the cuDNN site, install it, then continue with these installation instructions. To compile with cuDNN set the `USE_CUDNN := 1` flag set in your `Makefile.config`.

Caffe requires BLAS as the backend of its matrix and vector com