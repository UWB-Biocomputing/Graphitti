# Get all header and source files from current directory
file(GLOB paramcontainer *.cpp *.h)

# Create a library from the list of files gathered
add_library(paramcontainer ${paramcontainer})