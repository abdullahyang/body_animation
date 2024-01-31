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

Eigen::Matrix<double, 24, 3> jointPositions;
using json = nlohmann::json;
using namespace Eigen;
using namespace std;
using namespace ceres;

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
            matrix(i, j) = values[idx++]/2 + 0.5;
        }
    }
    return matrix;
}

Eigen::Matrix<double, 24, 3> CalculateJointPosition(const double* poseParameters, const double* shapeParameters)
{
    std::string poseparams;
    std::string shapeparams;
    poseparams = arrayToString(poseParameters, 72);
    shapeparams = arrayToString(shapeParameters, 10);
    std::string cmd = "conda run -n py35 python ../data/model/get_loc.py " + poseparams + " " + shapeparams;
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


struct SmplCostFunctor {
    SmplCostFunctor(const std::vector<Keypoint>& keypoints, const CameraParams& camParams)
        : keypoints_(keypoints), camParams_(camParams) {}

    bool operator()(const double* const parameters, double* residuals) const {
        int OpenPoseMapping[14] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
        int SMPLMapping[14] = { 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 6, 1, 4, 7 };
        residuals[0] = 0;
        const double* poseParameters = parameters;
        const double* shapeParameters = parameters + 72;
        CalculateJointPosition(poseParameters, shapeParameters);
        //CameraParams camParams_ = camParams;
        //vector<Keypoint> keypoints_ = keypoints;
        for (int i = 0; i < 14; i++)
        {
            Vector3d jointPosition3D = jointPositions.row(SMPLMapping[i]);
            //std::cout << "Joint Position 3D: " << jointPosition3D.transpose() << std::endl;
            Vector3d projected = camParams_.intrinsicMatrix.cast<double>() * (camParams_.rotationMatrix.cast<double>() * jointPosition3D + camParams_.translation.cast<double>());
            Vector2d jointPosition2D(projected(0) / projected(2), projected(1) / projected(2));
            //std::cout << "Joint Position 2D: " << jointPosition2D.transpose() << std::endl;
            //std::cout << "GT: " << keypoints_[OpenPoseMapping[i]].x << " " << keypoints_[OpenPoseMapping[i]].y << std::endl;
            residuals[0] += keypoints_[OpenPoseMapping[i]].confidence * sqrt((jointPosition2D.x() - keypoints_[OpenPoseMapping[i]].x) * (jointPosition2D.x() - keypoints_[OpenPoseMapping[i]].x) + (jointPosition2D.y() - keypoints_[OpenPoseMapping[i]].y) * (jointPosition2D.y() - keypoints_[OpenPoseMapping[i]].y));
        }
        cout << residuals[0] << endl;
        cout << time(nullptr) << endl;
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

    std::string filePath = "../data/EHF/01_2Djnt.json";
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
    
    jointPositions = CalculateJointPosition(Parameters, Parameters+72);
     

    // CostFunction* cost_function = new SmplCostFunction(keypoints, camParams);

    //CostFunction* cost_function = new AutoDiffCostFunction<SmplCostFunctor, 1, 82>(
        //new SmplCostFunctor(keypoints, camParams));

    CostFunction* cost_function = new NumericDiffCostFunction<SmplCostFunctor, CENTRAL, 1, 82>(
        new SmplCostFunctor(keypoints, camParams), TAKE_OWNERSHIP);

    problem.AddResidualBlock(cost_function, nullptr, Parameters);


    // addCostFunctions(&problem, keypoints, camParams, Parameters);
    // TODO: add other cost functions according to the paper

    Solver::Options options;
    options.linear_solver_type = DENSE_QR;
    options.minimizer_progress_to_stdout = true;


    Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    // cout << summary.FullReport() << endl;
 
    for (int i = 0; i < 82; ++i) {
        if (i < 72)
        {
            // cout << "Pose parameter " << i << ": " << Parameters[i] << endl;
        }
        else cout << "Shape parameter " << i-72 << ": " << Parameters[i] << endl;
    }

    return 0;
}
