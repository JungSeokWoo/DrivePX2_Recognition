project(px2Recog)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_CXX_STANDARD 11)
add_definitions(-std=c++11)
enable_language("CUDA")

find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62)

add_definitions(-DVIBRANTE)
add_definitions(-DVIBRANTE_V5L)

configure_file(px2Src/framework/DataPath.hpp.in ${CMAKE_SOURCE_DIR}/px2Src/framework/DataPath.hpp)

FILE(GLOB_RECURSE PROJECT_HEADER "*.h" "src/*.h" "src/*.hpp")
FILE(GLOB_RECURSE PROJECT_SRC "*.cpp" "src/*.cpp" "src/*.cu")

# cuda
INCLUDE_DIRECTORIES(/usr/local/cuda/include)

# driveworks
INCLUDE_DIRECTORIES(/usr/local/driveworks/include)
INCLUDE_DIRECTORIES(/usr/local/driveworks/include/dw/sensors/camera)
INCLUDE_DIRECTORIES(/usr/local/driveworks/include/dw/image)

INCLUDE_DIRECTORIES(
src
px2Src
px2Src/glfw-3.2.1/include
)


LINK_DIRECTORIES(
/usr/local/lib
/usr/local/cuda/lib64
/usr/local/driveworks/lib
${CMAKE_SOURCE_DIR}/px2Src/glfw-3.2.1/build/src
)

set(PX2_LIBS
nvmedia
glfw
EGL
driveworks
cudart
drm
GLESv2
nvinfer
nvcaffe_parser
pthread
)

aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST} ${PROJECT_HEADER} ${PROJECT_SRC})

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${PX2_LIBS})


