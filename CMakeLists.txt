cmake_minimum_required(VERSION 3.15 FATAL_ERROR) 

project(MyApp VERSION 1.0 LANGUAGES CXX CUDA)

# Setting the standard of cpp
# using C++20/ gnu++11(compatible with the version of cuda and ubuntu)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Debug)

# Dependencies
find_package(PkgConfig REQUIRED) # for vips installation 
find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED CONFIG COMPONENTS thread system) 
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

pkg_check_modules(VIPS REQUIRED vips-cpp) # img processing 
pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(OpenCV REQUIRED opencv4)

include_directories(${GLIB_INCLUDE_DIRS})
link_directories(${GLIB_LIBRARY_DIRS})
add_definitions(${GLIB_CFLAGS_OTHER})

# TensorRT libraries 
find_library(NVINFER_LIB nvinfer)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin)
find_library(NVINFER_VC_PLUGIN_LIB nvinfer_vc_plugin)

include_directories(
    /usr/include
    /usr/include/x86_64-linux-gnu
    /usr/include
)

link_directories(
    /usr/lib/x86_64-linux-gnu
    /usr/include
)

# Adding paths of the libvips libraries 
include_directories(${VIPS_INCLUDE_DIRS})
link_directories(${VIPS_LIBRARIES_DIRS}) 
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(nlohmann_json REQUIRED)

# Path to Build Darknet - for model inference
set(DARKNET_PATH /home/src/darknet/build)
include_directories($DARKNET_PATH)

# Enable Cuda and Cudnn 
option(ENABLE_GPU "Enable GPU support if available" ON)
option(ENABLE_CUDNN "Enable cuDNN if available" ON)

# Forced Recognition of the pathways of CUDA and Cudnn 
if(ENABLE_GPU)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        message(STATUS "✅ CUDA detected: ${CUDAToolkit_VERSION}")
        enable_language(CUDA)
        set(CMAKE_CUDA_ARCHITECTURES 86) # RTX 3060

        set(GPU_LIBS CUDA::cudart)

        if(ENABLE_CUDNN)
            find_library(CUDNN_LIBRARY cudnn
                HINTS /usr/lib/x86_64-linux-gnu /usr/local/cuda-12.4/lib64
            )
            if(CUDNN_LIBRARY)
                message(STATUS "✅ cuDNN detected: ${CUDNN_LIBRARY}")
                list(APPEND GPU_LIBS ${CUDNN_LIBRARY})
            else()
                message(WARNING "cuDNN not found, proceeding without it")
            endif()
        endif()
    else()
        message(WARNING "CUDA not found, building CPU-only version")
    endif()
endif()

# Running the file after make (in the build option)
# Comment 1 away to run a successful build!! 
file(GLOB TEST_SRC src/main_darknet.cpp) 
# file(GLOB TEST_SRC src/main_tensor.cpp)

add_executable(Main ${TEST_SRC})
target_include_directories(Main PRIVATE include)
target_link_libraries(Main ${GLIB_LIBRARIES} ${VIPS_LIBRARIES} ${GPU_LIBS} ${OpenCV_LIBRARIES} ${NVINFER_LIB} 
${NVINFER_PLUGIN_LIB} ${NVINFER_VC_PLUGIN_LIB} cudart nvonnxparser nlohmann_json::nlohmann_json darknet 
Boost::thread Boost::system pthread GTest::gtest GTest::gtest_main)

# Memory Status to keep track of the output. 
message(STATUS "")
message(STATUS "========== Build Summary ==========")
message(STATUS "Project:          ${PROJECT_NAME}")
message(STATUS "C++ Standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV:           ${OpenCV_VERSION}")
message(STATUS "VIPS:             ${VIPS_VERSION}")
message(STATUS "GLib:             ${GLIB_VERSION}")
message(STATUS "CUDA Enabled:     ${ENABLE_GPU}")
message(STATUS "cuDNN Enabled:    ${ENABLE_CUDNN}")
message(STATUS "===================================")
