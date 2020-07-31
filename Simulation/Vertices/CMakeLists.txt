# Get all header and source files from current directory
file(GLOB Vertices *.cpp *.h)

# Create a library from the list of files gathered
add_library(Vertices ${Vertices})

# The Vertices library has a dependency on the Edges library
target_link_libraries(Vertices Edges)