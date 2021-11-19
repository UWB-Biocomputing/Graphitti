# Get all header and source files from current directory and all subdirectories
file(GLOB_RECURSE Connections *.cpp *.h *.cu)

# Create a library from all the header and source files and in the current directory
add_library(Connections STATIC ${Connections})
