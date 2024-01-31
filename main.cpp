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

using json = nlohmann::json;
using namespace Eigen;
using namespace std;
using namespace ceres;

struct Keypoint {
    float x, y, confidence;
};

struct CameraParams {
    Matrix3f intrinsicMatrix;
    Vector3f translation;
    Matrix3f rotationMatrix;
};

class SmplCostFunction : public SizedCostFunction<2, 72> {
public:
    SmplCostFunction(const Keypoint& keypoint, const CameraParams& camParams, int jointIndex)
        : keypoint_(keypoint), camParams_(camParams), jointIndex_(jointIndex) {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        Vector3d jointPosition3D = CalculateJointPosition(parameters[0], jointIndex_);
        std::cout << "Joint Position 3D: " << jointPosition3D.transpose() << std::endl;
        Vector3d projected = camParams_.intrinsicMatrix.cast<double>() * (camParams_.rotationMatrix.cast<double>() * jointPosition3D + camParams_.translation.cast<double>());

        Vector2d jointPosition2D(projected(0) / projected(2), projected(1) / projected(2));
        residuals[0] = jointPosition2D.x() - keypoint_.x;
        residuals[1] = jointPosition2D.y() - keypoint_.y;
        return true;
    }

private:
    Keypoint keypoint_;
    CameraParams camParams_;
    int jointIndex_;

    Vector3d CalculateJointPosition(const double* poseParameters, int jointIndex) const {
        // TODO: Calculating 3D pos according to SMPL parameters
        Vector3d jointPosition3D;
        return jointPosition3D;
        /* The return will be like :
        
        [[ 0.0000,  0.0000,  0.0000],
        [ 0.1047,  0.0264,  0.0553],
        [-0.0143, -0.0332,  0.1131],
        [-0.0413, -0.0274, -0.1057],
        [-0.2409, -0.0764,  0.1925],
        [ 0.1994,  0.1042, -0.1879],
        [-0.0770, -0.1661, -0.0882],
        [-0.2542, -0.3910, -0.0687],
        [ 0.5843,  0.0894, -0.3292],
        [-0.1209, -0.1988, -0.0581],
        [-0.1335, -0.4436, -0.1084],
        [ 0.6292,  0.0716, -0.1997],
        [-0.3246, -0.3035, -0.0078],
        [-0.2722, -0.2150, -0.0874],
        [-0.1955, -0.3176,  0.0087],
        [-0.3154, -0.2185, -0.0316],
        [-0.3563, -0.2580, -0.0795],
        [-0.2850, -0.2706,  0.0043],
        [-0.3687, -0.5200, -0.0980],
        [-0.4968, -0.3812,  0.1020],
        [-0.5849, -0.4042, -0.0570],
        [-0.7013, -0.5189,  0.0302],
        [-0.5684, -0.3480,  0.0060],
        [-0.7739, -0.4913, -0.0072]]

        */ 

    }
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
    CameraParams camParams;
    std::string filePath = "../data/EHF/01_2Djnt.json";
    cout << getcwd(NULL, 0) << endl;
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
    double poseParameters[72] = {0};

    for (int i = 0; i < keypoints.size(); ++i) {
        CostFunction* cost_function = new SmplCostFunction(keypoints[i], camParams, i);
        problem.AddResidualBlock(cost_function, nullptr, poseParameters);
    }
    // TODO: add other loss functions according to the paper

    Solver::Options options;
    options.linear_solver_type = DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << endl;
 
    for (int i = 0; i < 72; ++i) {
        cout << "Pose parameter " << i << ": " << poseParameters[i] << endl;
    }

    return 0;
}
