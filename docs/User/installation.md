# 1.2 Installation

## 1.2.1 Necessary Hardware/Software

Graphitti is designed to be easy to use and fast to simulate with, but given its scope and flexibility, there are some tradeoffs. 

First, and perhaps most importantly, for the speedups that we desire, we found that **CUDA** was the most reasonable way to go. Hence,  if you want to use Graphitti for migrating your model to GPUs, you will need the following: 

- **Linux**: Currently, Graphitti only works on Linux. Any distro that supports **GNU-Make** and your chosen NVIDIA graphics card (if going the GPU route) should work. Make sure you have these packages:
- **NVIDIA GPU**: If you want your simulator to run on GPUs, you must use an NVIDIA GPU that is CUDA capable. Check NVIDIA's website for an up-to-date [list] of CUDA-compliant devices. 
- [**CUDA**]: if you intend to use the GPU functionality for high performance. Graphitti has been tested running on CUDA Version 8.0.44. 
- [HDF5]: HDF5 is a data model, library, and file format for storing and managing data. For example, Matlab has built-in functions that can easily manage, view, and analyze data in HDF5 format. To install HDF5, simply follow the website instructions. If you don't wish to use HDF5, you can use the XML format which is also supported.  

To become a Graphitti user or collaborator, you might also need:

- **[Git](http://git-scm.com/)** & **[GitHub](https://github.com/)**: If you wish to use or contribute to the most up-to-date Graphitti that is currently under development, you will need to get it from GitHub and keep it in sync. 
- **Matlab** or **Octave**: If you want to view the output results using our scripts

Of course, Graphitti is totally open source. If you wanted, you could modify Graphitti and make an OpenCL version. 

## 1.2.2 Download Graphitti
In order to get started with Graphitti, you will need to build it from scratch, which means getting its source codes. You can either download Graphitti source codes as a zip file of a stable release (See [the release page] or fork the development version from Graphitti GitHub repository (See [Fork and clone Graphitti](#1221-fork-and-clone-graphitti)).
### 1.2.2.1 Fork and clone Graphitti
If you are a Github user, you can simply fork and clone Graphitti. If you are new to Github, follow our Wiki page on [Contribute to Graphitti open source project]. You can also go over our [Git Crash Course] for some useful tips.
## 1.2.3 Install Graphitti
In order to compile and run Graphitti, you will need to set up a couple things in the [**CMakeLists.txt**] first.

1. Change to Graphitti directory in your terminal

   ```shell
   $ cd Graphitti/build
   ```

2. Open **CMakeLists.txt** and change the following parameters:

	If you are using **CUDA**, 
   	
	- change the CUDA library directory: ```ENABLE_CUDA YES``` 
	- you might also need to add your CUDA home directory into the ```PATH``` environment variable 

3. Graphitti is written in C++11 and CUDA C/C++. Make sure you have all these dependencies in order to compile Graphitti:
   - [make]
   - [g++]
   - [h5c++]: compile script for HDF5 C++ programs
   - [nvcc]: if you are using GPU for high performance, nvcc is the compiler by Nvidia for use with CUDA

---------
[>> Next: 1.3 Quickstart](quickstart.md)

-------------
[<< Go back to User Documentation page](index.md)

---------
[<< Go back to Graphitti home page](http://uwb-biocomputing.github.io/Graphitti/)

[//]: # (Moving URL links to the bottom of the document for ease of updating - LS)
[//]: # (Links to repo items which exist outside of the docs folder need an absolute link.)

   [Contribute to Graphitti open source project]: <https://github.com/UWB-Biocomputing/Graphitti/blob/master/CONTRIBUTING.md>
   [**CMakeLists.txt**]: <https://github.com/UWB-Biocomputing/Graphitti/blob/master/CMakeLists.txt>
   [Git Crash Course]: <https://github.com/UWB-Biocomputing/BrainGrid/wiki/Git-Crash-Course>
   [the release page]: <https://github.com/UWB-Biocomputing/Graphitti/releases>
   [h5c++]: <https://support.hdfgroup.org/archive/support/HDF5/Tutor/compile.html>
   [nvcc]: <http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#axzz4ftSRZe00>
   [make]: <https://www.gnu.org/software/make/>
   [g++]: <https://gcc.gnu.org/>
   [**CUDA**]: <https://developer.nvidia.com/cuda-downloads>
   [HDF5]: <https://support.hdfgroup.org/documentation/hdf5/latest/_getting_started.html>
   [list]: <https://developer.nvidia.com/cuda-gpus>