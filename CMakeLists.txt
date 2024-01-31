cmake_minimum_required(VERSION 3.10)


project(SMPLModelOptimization VERSION 1.0)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)


find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)


add_executable(SMPLModelOptimization main.cpp)


target_link_libraries(SMPLModelOptimization Eigen3::Eigen)
target_link_libraries(SMPLModelOptimization ${CERES_LIBRARIES})
target_include_directories(SMPLModelOptimization PRIVATE ./)
