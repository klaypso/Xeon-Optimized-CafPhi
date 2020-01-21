---
title: Installation: RHEL / Fedora / CentOS
---

# RHEL / Fedora / CentOS Installation

**General dependencies**

    sudo yum install protobuf-devel leveldb-devel snappy-devel opencv-devel boost-devel hdf5-devel

**Remaining dependencies, recent OS**

    sudo yum install gflags-devel glog-devel lmdb-devel

**Remaining dependencies, if not found**

    # glog
    wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
    tar zxvf glog-0.3.3.tar.gz
    cd glog-0.3.3
    ./configure
    make && make install
    # gflags
    wget https://github.com/schuhschuh/gflags/archive/master.zip
    unzip master.zip
    cd gflags-master
    mkdir build && cd build
   