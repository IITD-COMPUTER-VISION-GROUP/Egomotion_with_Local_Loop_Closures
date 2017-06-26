cmake_minimum_required(VERSION 2.8.12)

project(ELLC)

#########################################################
# FIND C++11 Support
#########################################################
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

#########################################################
# FIND OPENCV
#########################################################
find_package(OpenCV REQUIRED)
#########################################################

set(EIGEN3_INCLUDE_DIR "/usr/local/include/eigen3/")

include_directories(${OpenCV_INCLUDE_DIRS}
					${EIGEN3_INCLUDE_DIR})

add_library(ELLC-helpers src/DepthPropagation.cpp
                       src/DisplayFunc.cpp
                       src/EigenInitialization.cpp
                       src/Frame.cpp
                       src/GlobalOptimize.cpp
                       src/ImageFunc.cpp
                       src/PixelWisePyramid.cpp
                       src/Pyramid.cpp
                       src/UserDefinedFunc.cpp)



add_executable(ELLC src/main.cpp)


target_link_libraries(ELLC ${OpenCV_LIBS}
							boost_system 
							boost_thread
                           ELLC-helpers)
