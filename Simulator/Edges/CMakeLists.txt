# Get all header and source files from current directory and all subdirectories
file(GLOB_RECURSE Edges *.cpp *.h *.cu)

# Create a library from the list of files gathered
add_library(Edges STATIC ${Edges})
