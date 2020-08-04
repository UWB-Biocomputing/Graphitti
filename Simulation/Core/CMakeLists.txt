add_subdirectory(FunctionNodes)

# Get all header and source files from current directory
file(GLOB Core *.cpp *.h)

# Remove BGDriver from the list of files, don't want 'main' in the library
list(REMOVE_ITEM Core "${CMAKE_CURRENT_SOURCE_DIR}/BGDriver-old.cpp")
list(REMOVE_ITEM Core "${CMAKE_CURRENT_SOURCE_DIR}/BGDriver.cpp")

# Create a library from the list of files gathered
add_library(Core ${Core})