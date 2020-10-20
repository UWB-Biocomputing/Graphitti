# Get all header and source files from current directory
file(GLOB Edges *.cpp *.h *.cu)

# Create a library from the list of files gathered
add_library(Edges ${Edges})

if (BRAINGRID_CUDA)
   set_property(TARGET Edges PROPERTY CUDA_ARCHITECTURES 35-virtual) 
endif()