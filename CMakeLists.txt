cmake_minimum_required(VERSION 2.8)
project(MKIDSpeckleControl)

find_package(OpenCV 3.4.8 REQUIRED)
find_package(Boost REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(include ${PYTHON_INCLUDE_DIRS})
link_directories(${BOOST_LIBRARY_DIR})

file(GLOB SOURCES "src/*.cpp")

#set (CMAKE_CXX_FLAGS "-DBOOST_LOG_DYN_LINK -g -O0 -std=c++11") #debug
#set (CMAKE_CXX_FLAGS "-DBOOST_LOG_DYN_LINK -pg -Ofast -fopenmp -std=c++11") #profile
set (CMAKE_CXX_FLAGS "-DBOOST_LOG_DYN_LINK -Ofast -fopenmp -std=c++11") #run TODO: make these real options
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB TESTSOURCES "tests/main.cpp" ${SOURCES})
add_executable(test ${TESTSOURCES})
target_link_libraries(test ${OpenCV_LIBS} -lImageStreamIO -lmkidshm -lboost_log -lboost_system -lboost_thread -lrt -lpthread)

file(GLOB SIMSOURCES "sim/*.cpp")
add_executable(linDMToMKID ${SIMSOURCES})
target_link_libraries(linDMToMKID ${OpenCV_LIBS} -lImageStreamIO -lmkidshm -lboost_log -lboost_system -lboost_thread -lrt -lpthread)

# python extension
#set(LIBRARY_OUTPUT_PATH "./python")
file(GLOB SPECKPY "speckpy/SpeckPy.cpp" ${SOURCES})
add_library(_speckpy MODULE ${SPECKPY})
set_target_properties(_speckpy PROPERTIES PREFIX "")
target_link_libraries(_speckpy ${OpenCV_LIBS} -lboost_log -lboost_python -lboost_system -lboost_thread ${PYTHON_LIBRARIES} -lImageStreamIO -lmkidshm  -lrt -lpthread)
install(TARGETS _speckpy LIBRARY DESTINATION speckpy)
