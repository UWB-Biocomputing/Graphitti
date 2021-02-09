# Get all header and source files from current directory
file(GLOB Layouts *.cpp *.h)

# Create a library from the list of files gathered
add_library(Layouts ${Layouts})