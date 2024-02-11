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
#include <cstdlib>
#include <ctime>

using json = nlohmann::json;
using namespace Eigen;
using namespace std;
using namespace ceres;
bool eval = 0;
double L2_Regularization_Lambda = 0.1;
Vector2d root_trans_global;

double global_Parameters[73];

Eigen::Matrix<double, 25, 3> jointPositions; // 24 SMPL joints + nose
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


std::string arrayToString(const double* array, int size) {
    std::string result;
    for (int i = 0; i < size; ++i) {
        if (i > 0)
            result += ",";
        result += std::to_string(array[i]);
    }
    return result;
}

Eigen::Matrix<double, 25, 3> parseJointPosition(std::string str)
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
    Eigen::Matrix<double, 25, 3> matrix;
    int idx = 0;
    for (int i = 0; i < 25; ++i) {
        for (int j = 0; j < 3; ++j) {
            // matrix(i, j) = values[idx++]/2 + 0.5;
            matrix(i, j) = values[idx++] / 2;
        }
    }
    return matrix;
}

Eigen::Matrix<double, 25, 3> CalculateJointPosition(const double* poseParameters, const double* shapeParameters)
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
        // working on figuring out wrong head matching, see fit_3D.py in SMPLify
        
        int OpenPoseMapping[14] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
        int SMPLMapping[14] = { 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7 };
        // int OpenPoseMapping[15] = { 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1 };
        // int SMPLMapping[15] = { 24, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 12 };
        for (int i = 0; i < 19; i++)
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
        // test
        double shapeParameters[10] = { 0 };
        shapeParameters[0] = parameters[72];
        // jointPositions = CalculateJointPosition(parameters, parameters + 72);
        jointPositions = CalculateJointPosition(parameters, shapeParameters);
        Vector3d jointPosition3D_root = jointPositions.row(SMPLMapping[7]);
        Vector3d projected_root = camParams_.intrinsicMatrix.cast<double>() * (camParams_.rotationMatrix.cast<double>() * jointPosition3D_root + camParams_.translation.cast<double>());
        Vector2d jointPosition2D_root(projected_root(0) / projected_root(2), projected_root(1) / projected_root(2));
        Vector2d root_trans = { keypoints_[OpenPoseMapping[7]].x - jointPosition2D_root.x() , keypoints_[OpenPoseMapping[7]].y - jointPosition2D_root.y() };
        root_trans_global = root_trans;
        for (int i = 0; i < 14; i++)
        {
            Vector3d jointPosition3D = jointPositions.row(SMPLMapping[i]);
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
            if (OpenPoseMapping[i] == 11 || OpenPoseMapping[i] == 14)
            {
                // more weight on ankles
                // residuals[i] += keypoints_[OpenPoseMapping[i]].confidence * sqrt((jointPosition2D.x() - keypoints_[OpenPoseMapping[i]].x) * (jointPosition2D.x() - keypoints_[OpenPoseMapping[i]].x) + (jointPosition2D.y() - keypoints_[OpenPoseMapping[i]].y) * (jointPosition2D.y() - keypoints_[OpenPoseMapping[i]].y));
            }
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
        residuals[18] += kpc * sqrt((jointPosition2Dt.x() - kpx) * (jointPosition2Dt.x() - kpx) + (jointPosition2Dt.y() - kpy) * (jointPosition2Dt.y() - kpy));
        if (eval)
        {
            std::cout << "left toe" << std::endl;
            std::cout << "Joint Position 2D: " << jointPosition2Dt.transpose() << std::endl;
            std::cout << "GT: " << kpx << " " << kpy << std::endl;
        }


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
        if (eval)
        {
            std::cout << "right toe" << std::endl;
            std::cout << "Joint Position 2D: " << jointPosition2Dt.transpose() << std::endl;
            std::cout << "GT: " << kpx << " " << kpy << std::endl;
        }
        
        // TODO: penalize unnatural bendings
        // 4,5 knee; 18,19 elbow
        int SMPL_Elbows_knees[4] = { 4, 5, 18, 19 };
        for (int i = 0; i < 4; i++)
        {
            int Param_index = 3 * SMPL_Elbows_knees[i];
            double theta = sqrt(parameters[Param_index] * parameters[Param_index] + parameters[Param_index + 1] * parameters[Param_index + 1] + parameters[Param_index + 2] * parameters[Param_index + 2]);
            double alpha = 10;
            // knees
            if (i < 2)
            {
                residuals[17] += alpha * exp(-theta);
            }
            // elbows
            if (i >= 2)
            {
                residuals[17] += alpha * exp(theta);
            }
        }
        return true;
    }

    std::vector<Keypoint> keypoints_;
    CameraParams camParams_;
};

struct HipsCostFunctor {
    HipsCostFunctor(const std::vector<Keypoint>& keypoints, const CameraParams& camParams)
        : keypoints_(keypoints), camParams_(camParams){}

    bool operator()(const double* const parameters, double* residuals) const {
        // working on figuring out wrong head matching, see fit_3D.py in SMPLify

        int OpenPoseMapping[14] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
        int SMPLMapping[14] = { 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7 };
        // int OpenPoseMapping[15] = { 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1 };
        // int SMPLMapping[15] = { 24, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 12 };
        for (int i = 0; i < 2; i++)
        {
            residuals[i] = 0;
        }

        // test
        double shapeParameters[10] = { 0 };
        shapeParameters[0] = parameters[72];
        // jointPositions = CalculateJointPosition(parameters, parameters + 72);
        jointPositions = CalculateJointPosition(parameters, shapeParameters);
        Vector3d jointPosition3D_root = jointPositions.row(SMPLMapping[7]);
        Vector3d projected_root = camParams_.intrinsicMatrix.cast<double>() * (camParams_.rotationMatrix.cast<double>() * jointPosition3D_root + camParams_.translation.cast<double>());
        Vector2d jointPosition2D_root(projected_root(0) / projected_root(2), projected_root(1) / projected_root(2));
        Vector2d root_trans = { keypoints_[OpenPoseMapping[7]].x - jointPosition2D_root.x() , keypoints_[OpenPoseMapping[7]].y - jointPosition2D_root.y() };
        root_trans_global = root_trans;

        // left toe
        Vector3d jointPosition3Dt = jointPositions.row(SMPLMapping[10]);
        Vector3d projectedt = camParams_.intrinsicMatrix.cast<double>() * (camParams_.rotationMatrix.cast<double>() * jointPosition3Dt + camParams_.translation.cast<double>());
        Vector2d jointPosition2Dt(projectedt(0) / projectedt(2), projectedt(1) / projectedt(2));
        jointPosition2Dt.x() += root_trans.x();
        jointPosition2Dt.y() += root_trans.y();
        double kpx = (keypoints_[19].x + keypoints_[20].x) / 2;
        double kpy = (keypoints_[19].y + keypoints_[20].y) / 2;
        double kpc = (keypoints_[19].confidence + keypoints_[20].confidence) / 2;
        residuals[0] += 2 * sqrt((jointPosition2Dt.x() - kpx) * (jointPosition2Dt.x() - kpx) + (jointPosition2Dt.y() - kpy) * (jointPosition2Dt.y() - kpy));
        // residuals[0] += 2 * (jointPosition2Dt.x() - kpx) * (jointPosition2Dt.x() - kpx);
        if (!eval)
        {
            std::cout << "left toe" << std::endl;
            std::cout << "Joint Position 2D: " << jointPosition2Dt.transpose() << std::endl;
            std::cout << "GT: " << kpx << " " << kpy << std::endl;
        }


        // right toe
        jointPosition3Dt = jointPositions.row(SMPLMapping[11]);
        projectedt = camParams_.intrinsicMatrix.cast<double>() * (camParams_.rotationMatrix.cast<double>() * jointPosition3Dt + camParams_.translation.cast<double>());
        jointPosition2Dt(projectedt(0) / projectedt(2), projectedt(1) / projectedt(2));
        jointPosition2Dt.x() += root_trans.x();
        jointPosition2Dt.y() += root_trans.y();
        kpx = (keypoints_[22].x + keypoints_[23].x) / 2;
        kpy = (keypoints_[22].y + keypoints_[23].y) / 2;
        kpc = (keypoints_[22].confidence + keypoints_[23].confidence) / 2;
        residuals[1] += 2 * sqrt((jointPosition2Dt.x() - kpx) * (jointPosition2Dt.x() - kpx) + (jointPosition2Dt.y() - kpy) * (jointPosition2Dt.y() - kpy));
        // residuals[1] += 2 * (jointPosition2Dt.x() - kpx) * (jointPosition2Dt.x() - kpx);
        if (!eval)
        {
            std::cout << "right toe" << std::endl;
            std::cout << "Joint Position 2D: " << jointPosition2Dt.transpose() << std::endl;
            std::cout << "GT: " << kpx << " " << kpy << std::endl;
        }

        
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
    srand(static_cast <unsigned> (time(0)));
    
    // double Parameters[82] = {0}; // 72 pose params + 10 shape params
    double Parameters[73] = {0}; // 72 pose params + first shape param
    for (int i = 0; i < 72; i++)
    {
        // Parameters[i] = -0.1 + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / (0.2)));
        // cout << Parameters[i] << endl;
    }
    // double Parameters[73] = { -0.0776407,0.00484453,0.010599,0.179969,-0.169484,0.101799,0.162619,0.0555054,-0.0891914,-0.190091,0.0404299,-0.035251,0.0265726,-1.42267,0.0212773,0.0674557,1.41078,0.0403866,-0.186953,0.0184656,-0.0161024,0,0,0,0,0,0,-0.18683,0.00231426,0.00341318,0,0,0,0,0,0,0,0,0,0.401941,0.326392,-0.17685,-0.096041,-0.323272,-0.0870878,0,0,0,-0.631725,0.454499,-0.446829,0.483115,-0.609692,0.636811,-0.103335,0.283711,0.0363246,-0.127691,-0.025355,0.349959,0,0,0,0,0,0,0,0,0,0,0,0,15.7047 };
    Parameters[72] = 15;
    double shapeParameters[10] = { 0 };
    shapeParameters[0] = Parameters[72];
    // double poseParameters[72] = { 0 };
    PyRun_SimpleString("import os");
    PyRun_SimpleString("os.chdir('../data/model/')");
    eval = 0;
    // jointPositions = CalculateJointPosition(Parameters, Parameters+72);
    jointPositions = CalculateJointPosition(Parameters, shapeParameters);
    ceres::NumericDiffOptions numeric_diff_options;
    numeric_diff_options.relative_step_size = 0.01;
    CostFunction* cost_function = new NumericDiffCostFunction<SmplCostFunctor, RIDDERS, 19, 73>(
        new SmplCostFunctor(keypoints, camParams), TAKE_OWNERSHIP);
    problem.AddResidualBlock(cost_function, nullptr, Parameters);
    
    
    // try optimizing hips rotation
    //CostFunction* hips_cost_function = new NumericDiffCostFunction<HipsCostFunctor, RIDDERS, 2, 73>(
        //new HipsCostFunctor(keypoints, camParams), TAKE_OWNERSHIP);
    //problem.AddResidualBlock(hips_cost_function, nullptr, Parameters);


    Solver::Options options;
    options.linear_solver_type = DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 32;
    options.max_solver_time_in_seconds = 1000;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.dense_linear_algebra_library_type = ceres::CUDA;
    options.max_num_iterations = 30;
    Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << endl;

    eval = 1;

    int OpenPoseMapping[15] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0 };
    int SMPLMapping[15] = { 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 24 };
    // jointPositions = CalculateJointPosition(Parameters, Parameters + 72);
    jointPositions = CalculateJointPosition(Parameters, shapeParameters);

    cout << "3D joint Positions" << endl << jointPositions << endl;
    cout << "( The No.24 joint 'Nose' is manually added )" << endl;

    for (int i = 0; i < 15; i++)
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

    Vector3d jointPosition3Dt = jointPositions.row(SMPLMapping[10]);
    Vector3d projectedt = camParams.intrinsicMatrix.cast<double>() * (camParams.rotationMatrix.cast<double>() * jointPosition3Dt + camParams.translation.cast<double>());
    Vector2d jointPosition2Dt(projectedt(0) / projectedt(2), projectedt(1) / projectedt(2));
    jointPosition2Dt.x() += root_trans_global.x();
    jointPosition2Dt.y() += root_trans_global.y();
    double kpx = (keypoints[19].x + keypoints[20].x) / 2;
    double kpy = (keypoints[19].y + keypoints[20].y) / 2;
    double kpc = (keypoints[19].confidence + keypoints[20].confidence) / 2;
    if (eval)
    {
        std::cout << "left toe" << std::endl;
        std::cout << "Joint Position 2D: " << jointPosition2Dt.transpose() << std::endl;
        std::cout << "GT: " << kpx << " " << kpy << std::endl;
    }


    // right toe
    jointPosition3Dt = jointPositions.row(SMPLMapping[11]);
    projectedt = camParams.intrinsicMatrix.cast<double>() * (camParams.rotationMatrix.cast<double>() * jointPosition3Dt + camParams.translation.cast<double>());
    jointPosition2Dt(projectedt(0) / projectedt(2), projectedt(1) / projectedt(2));
    jointPosition2Dt.x() += root_trans_global.x();
    jointPosition2Dt.y() += root_trans_global.y();
    kpx = (keypoints[22].x + keypoints[23].x) / 2;
    kpy = (keypoints[22].y + keypoints[23].y) / 2;
    kpc = (keypoints[22].confidence + keypoints[23].confidence) / 2;
    if (eval)
    {
        std::cout << "right toe" << std::endl;
        std::cout << "Joint Position 2D: " << jointPosition2Dt.transpose() << std::endl;
        std::cout << "GT: " << kpx << " " << kpy << std::endl;
    }

    cout << "Pose parameter :" << endl;
    for (int i = 0; i < 72; ++i) {
        cout << Parameters[i];
        if (i != 71) cout << ",";
    }
    cout << endl << "Shape parameter :";
    shapeParameters[0] = Parameters[72];
    for (int i = 0; i < 10; i++)
    {
        cout << shapeParameters[i];
        if (i != 9) cout << ",";
    }
    cout << endl << "Model root translation: " << root_trans_global.x() << "," << root_trans_global.y() << endl;



    cout << "20" << endl;
    if (Py_FinalizeEx() < 0) {
        exit(120);
    }

    return 0;
}
