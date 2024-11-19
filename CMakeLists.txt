cmake_minimum_required(VERSION 3.24)

project(Final_Project LANGUAGES CXX CUDA)

add_executable(Final_Project main.cu)

# Set the output directories
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
#set_target_properties( Final_Project PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build )
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})


# Compile Info
# Set CUDA architectures to native
set(CMAKE_CUDA_ARCHITECTURES native)
target_compile_features(Final_Project PUBLIC cxx_std_11)
target_compile_features(Final_Project PUBLIC cuda_std_11)



#-----------------------------
# Gen data exe
project(GenData )
add_executable(GenData data_generator.cpp)
# Set the output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
