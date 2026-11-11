cmake_minimum_required(VERSION 3.16)

project(tflite_patch C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

find_package(Threads REQUIRED)

add_library(tflite_patch SHARED
    src/hook.c
    src/invoke_hook.c
    src/log_filter.c
    src/models.c
    src/sentinel.c
    src/tensors.c
    src/trace.c
)

target_compile_definitions(tflite_patch PRIVATE _GNU_SOURCE)
target_compile_options(tflite_patch PRIVATE -Wall -Wextra -fPIC)
target_include_directories(tflite_patch PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(tflite_patch PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)

set_target_properties(tflite_patch PROPERTIES
    C_VISIBILITY_PRESET default
    OUTPUT_NAME "tflite_patch"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

