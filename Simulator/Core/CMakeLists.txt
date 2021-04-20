add_subdirectory(FunctionNodes)

# Get all header and source files from current directory
file(GLOB Core *.cpp *.h *.cu)

# Remove Driver from the list of files, don't want 'main' in the library
list(REMOVE_ITEM Core "${CMAKE_CURRENT_SOURCE_DIR}/Driver.cpp")

# Create a library from the list of files gathered
add_library(Core ${Core})

if (ENABLE_CUDA)
   set_property(TARGET Core PROPERTY CUDA_ARCHITECTURES 35-virtual) 
endif()