# Built in the raisin_novnc container against the raisin install (like log/prior_eval):
#   cmake -S . -B build -DRAISIN_ROOT=/home/raisin_master && cmake --build build -j
cmake_minimum_required(VERSION 3.15)
project(offline_slam_host)
set(CMAKE_CXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(RAISIN_ROOT /home/raisin_master CACHE PATH "raisin_master checkout with install/")
# Same -march as the raisin build (raisin_master/CMakeLists.txt RAISIN_MARCH): Eigen objects passed to raisin
# libraries (tf::Transform::sendTransform takes an Isometry3d) must use the same AVX alignment or they crash.
set(RAISIN_MARCH x86-64-v3 CACHE STRING "-march used for the raisin install")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${RAISIN_MARCH}")
list(APPEND CMAKE_PREFIX_PATH ${RAISIN_ROOT}/install ${RAISIN_ROOT}/.cache/vcpkg_installed/x64-linux)
find_package(raisin_network REQUIRED)
find_package(raisin_plugin REQUIRED)
find_package(raisin_robot REQUIRED)
find_package(raisin_tf REQUIRED)
find_package(raisim REQUIRED)
find_package(Eigen3 REQUIRED)
add_executable(offline_slam_host offline_slam_host.cpp)
target_include_directories(offline_slam_host PRIVATE ${RAISIN_ROOT}/install/include)
target_link_libraries(offline_slam_host raisin_network raisin_plugin raisin_robot raisin_tf raisim::raisim Eigen3::Eigen dl)
