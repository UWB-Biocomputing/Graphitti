cmake_minimum_required(VERSION 3.10)
project(SummerOfBrain)

set(CMAKE_CXX_STANDARD 14)

# Get all the files in the directory that can be compiled
file(GLOB TinyXPathFiles *.cpp *.h *.hpp)

# Create a new Library using the files in the directory
add_library(TinyXLib ${TinyXPathFiles})

# Allows access to the files in the library from any location
target_include_directories(TinyXLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


