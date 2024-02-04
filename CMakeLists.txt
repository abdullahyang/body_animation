cmake_minimum_required(VERSION 3.10)

project(SMPLModelOptimization VERSION 1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)


set(Python3_ROOT_DIR "C:/Users/35449/anaconda3/envs/py36")
set(Python3_EXECUTABLE "C:/Users/35449/anaconda3/envs/py36/python.exe")
set(Python3_LIBRARY "C:/Users/35449/anaconda3/envs/py36/libs/python36.lib")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

add_executable(SMPLModelOptimization main.cpp)
include_directories(./include/)

target_link_libraries(SMPLModelOptimization Eigen3::Eigen)
target_link_libraries(SMPLModelOptimization ${CERES_LIBRARIES})
target_link_libraries(SMPLModelOptimization Python3::Python)

target_include_directories(SMPLModelOptimization PRIVATE ${Python3_INCLUDE_DIRS} ./)

