cmake_minimum_required(VERSION 3.9)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(LQRPlanner)

find_package(Eigen3 REQUIRED)
message(STATUS "Path to Eigen directory: ${Eigen3_DIR}")

find_package(Threads REQUIRED)

add_subdirectory(proto)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library (VehicleModel src/VehicleModel.cc ${Eigen_LIBRARIES})
add_library (LqrPlanner src/LqrPlanner.cc ${Eigen_LIBRARIES})

add_executable (main src/main.cc)
target_link_libraries (main protobuf proto LqrPlanner VehicleModel ${CMAKE_THREAD_LIBS_INIT})
