cmake_minimum_required(VERSION 3.18)
project(main LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LIBTORCH_DIR ${CMAKE_SOURCE_DIR}/libtorch)
set(LIBTORCH_ZIP ${CMAKE_SOURCE_DIR}/libtorch.zip)

if(NOT EXISTS ${LIBTORCH_DIR})
    message(STATUS "Downloading LibTorch...")

    file(DOWNLOAD
        https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.0.zip
        ${LIBTORCH_ZIP}
        SHOW_PROGRESS
    )

    file(ARCHIVE_EXTRACT
        INPUT ${LIBTORCH_ZIP}
        DESTINATION ${CMAKE_SOURCE_DIR}
    )
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${LIBTORCH_DIR})

find_package(Torch REQUIRED)

add_executable(main src/main.cpp)

target_link_libraries(main PRIVATE ${TORCH_LIBRARIES})

target_compile_options(main PRIVATE -Wall -Wextra -Wpedantic)

set_property(TARGET main PROPERTY CXX_STANDARD 17)