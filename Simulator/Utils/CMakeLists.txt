# Used to locate other CMakeLists.txt files for further copilation of the project.
add_subdirectory(Matrix)
add_subdirectory(RNG)
#add_subdirectory(Inputs)

# Get all header and source files from current directory
file(GLOB Utils ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.h)

# Create a library from the list of files gathered
add_library(Utils STATIC ${Utils})

if (ENABLE_CUDA)
   set_property(TARGET Utils PROPERTY LANGUAGE CUDA)
   set_property(TARGET Utils PROPERTY CUDA_ARCHITECTURES 35-virtual) 
   set_property(TARGET Utils PROPERTY LINKER_LANGUAGE CUDA)
endif()

