# Get all header and source files from current directory
file(GLOB Connections *.cpp *.h)

# Create a library from all the header and source files and in the current directory
add_library(Connections ${Connections})