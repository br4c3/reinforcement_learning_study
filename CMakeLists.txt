cmake_minimum_required(VERSION 3.18)
project(main LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

add_executable(main
    src/main.cpp
)

target_include_directories(main
    PRIVATE ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(main
    PRIVATE ${TORCH_LIBRARIES}
)

target_compile_options(main PRIVATE
    -Wall -Wextra -Wpedantic
)

set_property(TARGET main PROPERTY CXX_STANDARD 17)