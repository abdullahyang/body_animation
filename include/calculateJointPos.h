#pragma once
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <direct.h>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>

using namespace Eigen;
using namespace std;
using namespace ceres;

std::string exec(const char* cmd);

Vector3d CalculateJointPosition(const double* poseParameters, int jointIndex);