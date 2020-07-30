# Used to locate other CMakeLists.txt files for further copilation of the project.
add_subdirectory(Matrix)
add_subdirectory(RNG)
add_subdirectory(Inputs)
#add_subdirectory(TinyXPath) # ToDo: Uncomment once 'connections-and-layouts' branch is merged

# Get all header and source files from current directory
file(GLOB Utils ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.h)

# ToDo: Remove once 'connections-and-layouts' branch is merged
list(REMOVE_ITEM Utils "${CMAKE_CURRENT_SOURCE_DIR}/ParameterManager.cpp")
list(REMOVE_ITEM Utils "${CMAKE_CURRENT_SOURCE_DIR}/ParameterManager.h")

# Create a library from the list of files gathered
add_library(Utils ${Utils})

