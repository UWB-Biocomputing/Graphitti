# Get all header and source files from current directory
file(GLOB Matrix ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.h)

# Remove these files from the library as they won't compile
list(REMOVE_ITEM Matrix "${CMAKE_CURRENT_SOURCE_DIR}/MatrixFactory.h")
list(REMOVE_ITEM Matrix "${CMAKE_CURRENT_SOURCE_DIR}/MatrixFactory.cpp")

# Create a library from the list of files gathered
add_library(Matrix ${Matrix})
