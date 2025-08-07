cmake_minimum_required(VERSION 3.26)
project(MoBrain)

set(CMAKE_CXX_STANDARD 23)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)


add_subdirectory(include)  
add_subdirectory(src)