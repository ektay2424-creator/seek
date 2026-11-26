cmake_minimum_required(VERSION 3.16)
project(seek LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(seek_core INTERFACE)
target_include_directories(seek_core INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/onnxruntime-win-x64-1.27.0/include
)

find_package(Threads REQUIRED)

add_executable(seek src/main.cpp)
target_link_libraries(seek PRIVATE seek_core Threads::Threads)

add_executable(seek_bench benchmarks/bench.cpp)
target_link_libraries(seek_bench PRIVATE seek_core Threads::Threads)

enable_testing()
add_executable(seek_tests tests/test_core.cpp)
target_link_libraries(seek_tests PRIVATE seek_core Threads::Threads)
add_test(NAME seek_tests COMMAND seek_tests)

if(MINGW)
  target_link_options(seek       PRIVATE -static)
  target_link_options(seek_bench PRIVATE -static)
  target_link_options(seek_tests PRIVATE -static)
endif()

if(WIN32)
  target_link_libraries(seek_bench PRIVATE psapi)
endif()

if(MINGW)
  target_link_options(seek       PRIVATE -static)
  target_link_options(seek_bench PRIVATE -static)
  target_link_options(seek_tests PRIVATE -static)
endif()