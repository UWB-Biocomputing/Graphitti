cmake_minimum_required(VERSION 3.10)
project(SummerOfBrain)

set(CMAKE_CXX_STANDARD 14)

file(GLOB Core ${CMAKE_CURRENT_SOURCE_DIR}/Connections *.cpp *.h)
file(GLOB Utils ${CMAKE_CURRENT_SOURCE_DIR}/Core *.cpp *.h)

add_library(TestLib ${Core} ${Utils})

target_link_libraries(TestLib Core)
target_link_libraries(TestLib gtest gtest_main)
