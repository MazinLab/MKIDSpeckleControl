cmake_minimum_required(VERSION 2.8)
project(MKIDFocalPlaneControl)

find_package(OpenCV 3.3 REQUIRED)
find_package(Boost REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(include ${PYTHON_INCLUDE_DIRS})
link_directories(${BOOST_LIBRARY_DIR})

file(GLOB SOURCES "src/*.cpp")

set (CMAKE_CXX_FLAGS "-DBOOST_LOG_DYN_LINK -g -pg -O0 -std=c++11")
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(mainScript ${SOURCES})
target_link_libraries(mainScript ${OpenCV_LIBS} -lImageStreamIO -lmkidshm -lboost_log -lboost_system -lrt -lpthread)

file(GLOB SIMSOURCES "sim/*.cpp")
add_executable(linDMToMKID ${SIMSOURCES})
target_link_libraries(linDMToMKID ${OpenCV_LIBS} -lImageStreamIO -lmkidshm -lboost_log -lboost_system -lrt -lpthread)

# python extensions
set(LIBRARY_OUTPUT_PATH "./python")
file(GLOB SPECKTODMPY "python/PySpeckleToDM.cpp" "src/SpeckleToDM.cpp" "src/DMChannel.cpp")
add_library(speckletodm SHARED ${SPECKTODMPY})
set_target_properties(speckletodm PROPERTIES PREFIX "")
target_link_libraries(speckletodm ${OpenCV_LIBS} -lboost_log -lboost_python -lboost_system ${PYTHON_LIBRARIES} -lImageStreamIO -lmkidshm  -lrt -lpthread)

file(GLOB PROPERTYTREEPY "python/PyPropertyTree.cpp")
add_library(propertytree SHARED ${PROPERTYTREEPY})
set_target_properties(propertytree PROPERTIES PREFIX "")
target_link_libraries(propertytree -lboost_log -lboost_python -lboost_system ${PYTHON_LIBRARIES})

file(GLOB SPECKLENULLERPY "python/PySpeckleNuller.cpp")
add_library(specklenuller SHARED ${SPECKLENULLERPY})
set_target_properties(specklenuller PROPERTIES PREFIX "")
target_link_libraries(specklenuller ${OpenCV_LIBS} -lboost_log -lboost_python -lboost_system ${PYTHON_LIBRARIES} -lImageStreamIO -lmkidshm  -lrt -lpthread)

file(GLOB SPECKPY "python/SpeckPy.cpp" ${SOURCES})
add_library(speckpy SHARED ${SPECKPY})
set_target_properties(speckpy PROPERTIES PREFIX "")
target_link_libraries(speckpy ${OpenCV_LIBS} -lboost_log -lboost_python -lboost_system ${PYTHON_LIBRARIES} -lImageStreamIO -lmkidshm  -lrt -lpthread)
