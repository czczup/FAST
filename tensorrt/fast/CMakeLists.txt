cmake_minimum_required(VERSION 2.6)

project(fast)

add_definitions(-std=c++14)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)
# tensorrt
include_directories(/home/cz2/TensorRT-7.0.0.11/include/)
link_directories(/home/cz2/TensorRT-7.0.0.11/lib/)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_PREFIX_PATH "/home/cz2/libtorch")
find_package(Torch REQUIRED)
#include_directories(/home/cz2/libtorch/include)
#link_directories(/home/cz2/libtorch/lib)

cuda_add_library(ccl SHARED ${PROJECT_SOURCE_DIR}/ccl_cuda.cu ${PROJECT_SOURCE_DIR}/ccl.h)
target_link_libraries(ccl nvinfer cudart ${TORCH_LIBRARIES})

add_executable(fast ${PROJECT_SOURCE_DIR}/fast.cpp)
target_link_libraries(fast nvinfer)
target_link_libraries(fast cudart)
target_link_libraries(fast ccl)
target_link_libraries(fast ${OpenCV_LIBS})
target_link_libraries(fast ${TORCH_LIBRARIES})
add_definitions(-O2 -pthread)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
endif()
