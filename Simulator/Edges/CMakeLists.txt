# Get all header and source files from current directory and all subdirectories
file(GLOB_RECURSE Edges *.cpp *.h *.cu)

# Create a library from the list of files gathered
add_library(Edges STATIC ${Edges})

if (ENABLE_CUDA)
   set_property(TARGET Edges PROPERTY LANGUAGE CUDA)
   set_property(TARGET Edges PROPERTY CUDA_ARCHITECTURES 35-virtual) 
   set_property(TARGET Edges PROPERTY LINKER_LANGUAGE CUDA)
endif()