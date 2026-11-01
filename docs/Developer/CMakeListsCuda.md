
# Using CUDA and CMakeLists  

This section contains information on a few things found to be necessary for the  
simulator CUDA code to build and run correctly when using CMakeLists. This is  
not a complete guide on using CMakeLists with CUDA.    

## Static Libraries  

Each library created in the CMakeLists.txt file that contains CUDA source code  
should be created with the STATIC keyword. More information on the difference  
between STATIC and SHARED type libraries can be found in the CMake documentation,  
and a short summary is available on the [add_library documentation page](https://cmake.org/cmake/help/latest/command/add_library.html). Libraries  
with no type specified will be assigned one automatically based on conditions  
listed in the documentation, but in the case of our simulator they can be observed  
as being SHARED if no type is specified.   

If a library containing CUDA source files is created as SHARED, the linking of  
the corresponding CUDA device code will be done individually just before the  
step of linking the SHARED library it is a part of. Explicitly declaring libraries  
containing CUDA code as STATIC instead allows the linking of CUDA device code to  
be put off and done all together just before the final executable is linked and  
built. This change allows the linker for CUDA code to function correctly and  
prevents many errors that arise if the libraries are left as SHARED.  

Example of creating a STATIC library:  
    ```add_library(Edges STATIC ${Edges_Source})```  
    Where `${Edges_Source}` is a collection of source files
 
## Separable Compilation  

CUDA separable compilation allows different pieces of CUDA code to be compiled  
into separate objects and then linked together later. More information can be  
found about Separate Compilation and Linking of CUDA code in this [NVIDIA developer  
blog post](https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/). This must be enabled in the CMakeLists.txt file individually for each  
library containing CUDA source code by setting the CUDA_SEPARABLE_COMPILATION  
property to ON. Information on this property can be found [here](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_SEPARABLE_COMPILATION.html) in the CMake  
documentation.  

It is set using the set_property command, for example like this:  
    ```set_property(TARGET Edges PROPERTY CUDA_SEPARABLE_COMPILATION ON)```  
    Where `Edges` is a library containing CUDA source code. 