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
#include <ctime>
#include <Python.h>
#include <math.h>

using json = nlohmann::json;
using namespace Eigen;
using namespace std;
using namespace ceres;
bool eval = 0;
double L2_Regularization_Lambda = 0.1;
Vector2d root_trans_global;

Eigen::Matrix<double, 24, 3> jointPositions;
struct Keypoint {
    float x, y, confidence;
};
vector<Keypoint> keypoints;

struct CameraParams {
    Matrix3f intrinsicMatrix;
    Vector3f translation;
    Matrix3f rotationMatrix;
};
CameraParams camParams;

std::string exec(const char* cmd)
{
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd, "r"), _pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

std::string arrayToString(const double* array, int size) {
    std::string result;
    for (int i = 0; i < size; ++i) {
        if (i > 0)
            result += ",";
        result += std::to_string(array[i]);
    }
    return result;
}

Eigen::Matrix<double, 24, 3> parseJointPosition(std::string str)
{
    std::vector<double> values;
    std::stringstream ss(str);
    char bracket, comma;
    double value;
    // cout << str << endl;
    // Expect an opening bracket
    if (!(ss >> bracket) || bracket != '[') {
        std::cerr << "Expected '[' at the beginning of the input string." << std::endl;
    }

    while (ss >> value) {
        values.push_back(value);
        ss >> comma; // To skip the comma
    }

    // Check for closing bracket
    if (ss.fail() && !ss.eof()) {
        ss.clear(); // Clear the fail state
        ss >> bracket;
        if (bracket != ']') {
            std::cerr << "Expected ']' at the end of the input string." << std::endl;
        }
    }
    Eigen::Matrix<double, 24, 3> matrix;
    int idx = 0;
    for (int i = 0; i < 24; ++i) {
        for (int j = 0; j < 3; ++j) {
            // matrix(i, j) = values[idx++]/2 + 0.5;
            matrix(i, j) = values[idx++] / 2;
        }
    }
    return matrix;
}
/*
Eigen::Matrix<double, 24, 3> CalculateJointPosition(const double* poseParameters, const double* shapeParameters)
{
    std::string poseparams;
    std::string shapeparams;
    poseparams = arrayToString(poseParameters, 72);
    shapeparams = arrayToString(shapeParameters, 10);
    std::string cmd = "conda run -n py36 python ../data/model/get_loc.py " + poseparams + " " + shapeparams;
    std::string output = "";
    try {
        output = exec(cmd.c_str());
        // std::cout << "Output: " << output;
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return parseJointPosition(output);
}
*/

Eigen::Matrix<double, 24, 3> CalculateJointPosition(const double* poseParameters, const double* shapeParameters)
{
    std::string poseparams;
    std::string shapeparams;
    poseparams = arrayToString(poseParameters, 72);
    shapeparams = arrayToString(shapeParameters, 10);
    PyObject* sysModule = PyImport_ImportModule("sys");
    PyObject* sysPath = PyObject_GetAttrString(sysModule, "path");
    PyList_Append(sysPath, PyUnicode_FromString("."));
    PyObject* sysArgv = PyObject_GetAttrString(sysModule, "argv");
    Py_DecRef(sysArgv);
    sysArgv = PyList_New(0);
    PyObject* ioModule = PyImport_ImportModule("io");
    PyObject* stringIO = PyObject_CallMethod(ioModule, "StringIO", NULL);
    PyObject* originalStdout = PyObject_GetAttrString(sysModule, "stdout");
    PyObject_SetAttrString(sysModule, "stdout", stringIO);
    // add params
    PyList_Append(sysArgv, PyUnicode_FromString("get_loc.py"));
    PyList_Append(sysArgv, PyUnicode_FromString(poseparams.c_str()));
    PyList_Append(sysArgv, PyUnicode_FromString(shapeparams.c_str()));
    PyObject_SetAttrString(sysModule, "argv", sysArgv);
    FILE* file = fopen("get_loc.py", "r");
    PyRun_SimpleFile(file, "get_loc.py");
    fclose(file);
    // get output
    PyObject* output = PyObject_CallMethod(stringIO, "getvalue", NULL);
    const char* outputStr = PyUnicode_AsUTF8(output);
    std::string cppOutput(outputStr);
    PyObject_SetAttrString(sysModule, "stdout", originalStdout);

    return parseJointPosition(cppOutput);
}


struct SmplCostFunctor {
    SmplCostFunctor(const std::vector<Keypoint>& keypoints, const CameraParams& camParams)
        : keypoints_(keypoints), camParams_(camParams) {}

    bool operator()(const double* const parameters, double* residuals) const {
        int OpenPoseMapping[14] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
        int SMPLMapping[14] = { 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7 };

        for (int i = 0; i < 17; i++)
        {
            residuals[i] = 0;
        }
        
        for (int i = 0; i < 72; i = i + 3)
        {
            // penalize > 180 degree rotation
            if (sqrt(parameters[i] * parameters[i] + parameters[i + 1] * parameters[i + 1] + parameters[i + 2] * parameters[i + 2]) > 1.57)
            {
                residuals[16] += 500.0;
            }
        }

        //const double* poseParameters = parameters;
        //const double* shapeParameters = parameters + 72;
        jointPositions = CalculateJointPosition(parameters, parameters + 72);
        Vector3d jointPosition3D_root = jointPositions.row(SMPLMapping[7]);
        // std::cout << "Joint Position 3D: " << jointPosition3D.transpose() << std::endl;
        Vector3d projected_root = camParams_.intrinsicMatrix.cast<double>() * (camParams_.rotationMatrix.cast<double>() * jointPosition3D_root + camParams_.translation.cast<double>());
        Vector2d jointPosition2D_root(projected_root(0) / projected_root(2), projected_root(1) / projected_root(2));
        Vector2d root_trans = { keypoints_[OpenPoseMapping[7]].x - jointPosition2D_root.x() , keypoints_[OpenPoseMapping[7]].y - jointPosition2D_root.y() };
        root_trans_global = root_trans;
        for (int i = 0; i < 14; i++)
        {
            Vector3d jointPosition3D = jointPositions.row(SMPLMapping[i]);
            // std::cout << "Joint Position 3D: " << jointPosition3D.transpose() << std::endl;
            Vector3d projected = camParams_.intrinsicMatrix.cast<double>() * (camParams_.rotationMatrix.cast<double>() * jointPosition3D + camParams_.translation.cast<double>());
            Vector2d jointPosition2D(projected(0) / projected(2), projected(1) / projected(2));
            jointPosition2D.x() += root_trans.x();
            jointPosition2D.y() += root_trans.y();
            if (eval)
            {
                std::cout << "i: " << i << std::endl;
                std::cout << "Joint Position 2D: " << jointPosition2D.transpose() << std::endl;
                std::cout << "GT: " << keypoints_[OpenPoseMapping[i]].x << " " << keypoints_[OpenPoseMapping[i]].y << std::endl;
            }
            // residuals[0] += keypoints_[OpenPoseMapping[i]].confidence * sqrt((jointPosition2D.x() - keypoints_[OpenPoseMapping[i]].x) * (jointPosition2D.x() - keypoints_[OpenPoseMapping[i]].x) + (jointPosition2D.y() - keypoints_[OpenPoseMapping[i]].y) * (jointPosition2D.y() - keypoints_[OpenPoseMapping[i]].y));
            residuals[i] += keypoints_[OpenPoseMapping[i]].confidence * sqrt((jointPosition2D.x() - keypoints_[OpenPoseMapping[i]].x) * (jointPosition2D.x() - keypoints_[OpenPoseMapping[i]].x) + (jointPosition2D.y() - keypoints_[OpenPoseMapping[i]].y) * (jointPosition2D.y() - keypoints_[OpenPoseMapping[i]].y));
            
        }
        for (int i = 0; i < 72; i = i + 3)
        {
            // L2 regularization
            // residuals[16] += L2_Regularization_Lambda * (std::pow(parameters[i], 2) + std::pow(parameters[i + 1], 2) + std::pow(parameters[i + 2], 2));
        }

        // left toe
        Vector3d jointPosition3Dt = jointPositions.row(SMPLMapping[10]);
        Vector3d projectedt = camParams_.intrinsicMatrix.cast<double>() * (camParams_.rotationMatrix.cast<double>() * jointPosition3Dt + camParams_.translation.cast<double>());
        Vector2d jointPosition2Dt(projectedt(0) / projectedt(2), projectedt(1) / projectedt(2));
        jointPosition2Dt.x() += root_trans.x();
        jointPosition2Dt.y() += root_trans.y();
        double kpx = (keypoints_[19].x + keypoints_[20].x) / 2;
        double kpy = (keypoints_[19].y + keypoints_[20].y) / 2;
        double kpc = (keypoints_[19].confidence + keypoints_[20].confidence) / 2;
        residuals[14] += kpc * sqrt((jointPosition2Dt.x() - kpx) * (jointPosition2Dt.x() - kpx) + (jointPosition2Dt.y() - kpy) * (jointPosition2Dt.y() - kpy));

        // right toe
        jointPosition3Dt = jointPositions.row(SMPLMapping[11]);
        projectedt = camParams_.intrinsicMatrix.cast<double>() * (camParams_.rotationMatrix.cast<double>() * jointPosition3Dt + camParams_.translation.cast<double>());
        jointPosition2Dt(projectedt(0) / projectedt(2), projectedt(1) / projectedt(2));
        jointPosition2Dt.x() += root_trans.x();
        jointPosition2Dt.y() += root_trans.y();
        kpx = (keypoints_[22].x + keypoints_[23].x) / 2;
        kpy = (keypoints_[22].y + keypoints_[23].y) / 2;
        kpc = (keypoints_[22].confidence + keypoints_[23].confidence) / 2;
        residuals[15] += kpc * sqrt((jointPosition2Dt.x() - kpx) * (jointPosition2Dt.x() - kpx) + (jointPosition2Dt.y() - kpy) * (jointPosition2Dt.y() - kpy));

        //residuals[1] += L2_Regularization_Lambda * (std::pow(parameters[3 * SMPLMapping[10]], 2) + std::pow(parameters[3 * SMPLMapping[10] + 1], 2) + std::pow(parameters[3 * SMPLMapping[10] + 2], 2));
        //residuals[1] += L2_Regularization_Lambda * (std::pow(parameters[3 * SMPLMapping[11]], 2) + std::pow(parameters[3 * SMPLMapping[11] + 1], 2) + std::pow(parameters[3 * SMPLMapping[11] + 2], 2));
        for (int i = 72; i < 82; i++)
        {
            // residuals[1] += parameters[i] * parameters[i];
        }
        // cout << residuals[0] << endl;
        // cout << time(nullptr) << endl;
        int SMPL_Elbows_knees[4] = {4, 5, 18, 19};
        // residuals[2] = 0;
        // TODO: penalize unnatural bendings

        return true;
    }

    std::vector<Keypoint> keypoints_;
    CameraParams camParams_;
};


vector<Keypoint> ParseKeypoints(const string& jsonString) {
    json j = json::parse(jsonString);
    vector<Keypoint> keypoints;

    if (!j["people"].empty()) {
        auto people = j["people"][0];
        auto keypointsArray = people["pose_keypoints_2d"];
        for (size_t i = 0; i < keypointsArray.size(); i += 3) {
            keypoints.push_back({keypointsArray[i], keypointsArray[i + 1], keypointsArray[i + 2]});
        }
    }
    return keypoints;
}


int main() {
    _putenv_s("PYTHONPATH", "C:\\Users\\35449\\anaconda3\\envs\\py36\\Lib;C:\\Users\\35449\\anaconda3\\envs\\py36\\DLLs;C:\\Users\\35449\\anaconda3\\envs\\py36\\Lib\\site-packages;%PATH%");
    _putenv_s("PYTHONHOME", "D:\\TUM\\3DSMC\\project\\body_animation\\data\\model");
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }

    std::string filePath = "../data/EHF/20_2Djnt.json";
    cout << _getcwd(NULL, 0) << endl;
    // Read keypoints
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "can't open: " << filePath << std::endl;
        return -1;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    std::string jsonInput = buffer.str();

    camParams.intrinsicMatrix << 1498.22426237, 0, 790.263706,
                                 0, 1498.22426237, 578.90334,
                                 0, 0, 1;
    camParams.translation << -0.03609917, 0.43416458, 2.37101226;
    Vector3f rotationVector(-2.98747896, 0.01172457, -0.05704687);
    AngleAxisf angleAxis(rotationVector.norm(), rotationVector.normalized());
    camParams.rotationMatrix = angleAxis.toRotationMatrix();

    vector<Keypoint> keypoints = ParseKeypoints(jsonInput);
    Problem problem;
    double Parameters[82] = {0}; // 72 pose params + 10 shape params
    Parameters[72] = 15;
    PyRun_SimpleString("import os");
    PyRun_SimpleString("os.chdir('../data/model/')");
    eval = 0;
    jointPositions = CalculateJointPosition(Parameters, Parameters+72);

    //CostFunction* cost_function = new AutoDiffCostFunction<SmplCostFunctor, 1, 82>(
        //new SmplCostFunctor(keypoints, camParams));
    ceres::NumericDiffOptions numeric_diff_options;
    numeric_diff_options.relative_step_size = 0.01;
    // *** WHY CAN'T CHANGE STEP SIZE HERE? ***
    CostFunction* cost_function = new NumericDiffCostFunction<SmplCostFunctor, RIDDERS, 17, 82>(
        new SmplCostFunctor(keypoints, camParams), TAKE_OWNERSHIP);
    //CostFunction* cost_function = new NumericDiffCostFunction<SmplCostFunctor, CENTRAL, 17, 82>(
        //new SmplCostFunctor(keypoints, camParams), TAKE_OWNERSHIP);
    //CostFunction* cost_function = new AutoDiffCostFunction<SmplCostFunctor, 1, 82>(
        //new SmplCostFunctor(keypoints, camParams));

    problem.AddResidualBlock(cost_function, nullptr, Parameters);
    // setting bounds
    for (int i = 0; i < 72; ++i) { // pose params
        // problem.SetParameterLowerBound(Parameters, i, -1.57);
        // problem.SetParameterUpperBound(Parameters, i, 1.57);
    }
    for (int i = 72; i < 82; ++i) { // shape params
        //problem.SetParameterLowerBound(Parameters, i, -3.0);
        //problem.SetParameterUpperBound(Parameters, i, 3.0);
    }


    // addCostFunctions(&problem, keypoints, camParams, Parameters);
    // TODO: add other cost functions according to the paper

    Solver::Options options;
    options.linear_solver_type = DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 32;
    options.max_solver_time_in_seconds = 1200;
    options.trust_region_strategy_type = ceres::DOGLEG;

    Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << endl;

    eval = 1;


    int OpenPoseMapping[14] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
    int SMPLMapping[14] = { 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 7, 1, 4, 8 };
    jointPositions = CalculateJointPosition(Parameters, Parameters + 72);
    for (int i = 0; i < 14; i++)
    {
        Vector3d jointPosition3D = jointPositions.row(SMPLMapping[i]);
        Vector3d projected = camParams.intrinsicMatrix.cast<double>() * (camParams.rotationMatrix.cast<double>() * jointPosition3D + camParams.translation.cast<double>());
        Vector2d jointPosition2D(projected(0) / projected(2), projected(1) / projected(2));
        jointPosition2D.x() += root_trans_global.x();
        jointPosition2D.y() += root_trans_global.y();
        if (eval)
        {
            std::cout << "i: " << i << std::endl;
            std::cout << "Joint Position 2D: " << jointPosition2D.transpose() << std::endl;
            std::cout << "GT: " << keypoints[OpenPoseMapping[i]].x << " " << keypoints[OpenPoseMapping[i]].y << std::endl;
        }
    }


    cout << "Pose parameter :" << endl;
    for (int i = 0; i < 72; ++i) {
        cout << Parameters[i];
        if (i != 71) cout << ",";
    }
    cout << endl << "Shape parameter :";
    for (int i = 72; i < 82; i++)
    {
        cout << Parameters[i];
        if (i != 81) cout << ",";
    }

    cout << endl;
    if (Py_FinalizeEx() < 0) {
        exit(120);
    }

    return 0;
}
