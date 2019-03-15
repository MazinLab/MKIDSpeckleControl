cmake_minimum_required(VERSION 2.8)
project(MKIDFocalPlaneControl)

include_directories(include)
find_package(OpenCV 3.3 REQUIRED)
find_package(Boost REQUIRED)

file(GLOB SOURCES "src/*.cpp")

set (CMAKE_CXX_FLAGS "-DBOOST_LOG_DYN_LINK -g -pg -O0 -std=c++11")
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(mainScript ${SOURCES})
target_link_libraries(mainScript ${OpenCV_LIBS} -lImageStreamIO -lmkidshm -lboost_log -lboost_system -lrt -lpthread)

