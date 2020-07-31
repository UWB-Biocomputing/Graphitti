# Get all the files in the directory that can be compiled.
file(GLOB TinyXPath *.cpp *.h *.hpp)

# Create a new Library using the files gathered.
add_library(TinyXPath ${TinyXPath})

# Allows access to the files in the library from any location
target_include_directories(TinyXPath PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


