import torch
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
# coding=utf-8
import sys



def load_smpl_model(model_path, gender='neutral'):
    """
    Load SMPL model.
    :param model_path: Path to the SMPL model (.pkl file).
    :param gender: Gender of the model ('male', 'female', or 'neutral').
    :return: Initialized SMPL model.
    """
    return SMPL_Layer(
        center_idx=0,
        gender=gender,
        model_root=model_path
    )

def get_joint_locations(smpl_layer, pose_params, shape_params):
    """
    Get 3D joint locations from SMPL model.
    :param smpl_layer: SMPL model layer.
    :param pose_params: Tensor of pose parameters (size: 72).
    :param shape_params: Tensor of shape parameters (size: 10).
    :return: 3D joint locations.
    """
    batch_size = 1
    pose_params = pose_params.unsqueeze(0)
    shape_params = shape_params.unsqueeze(0)

    verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
    return Jtr.squeeze(0), verts.squeeze(0)[332]

def calculate_joint_position(pPoseParams, pShapeParams):
    # torch.set_printoptions(sci_mode=False)
    # model_path = '../data/model'
    import os
    print(os.getcwd())
    model_path = 'D:/TUM/3DSMC/project/body_animation/data/model'
    # smpl_layer = load_smpl_model(model_path)
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender='neutral',
        model_root='.'
    )
    pose_input = str(pPoseParams).split(',')
    
    pose_params = []
    shape_params = []
    for param in pose_input:
        pose_params.append(float(param))
    
    shape_input = pShapeParams.split(',')
    for param in shape_input:
        shape_params.append(float(param))
    pose_params = torch.Tensor(pose_params)
    shape_params = torch.Tensor(shape_params)

    # Get joint locations
    joint_locations = get_joint_locations(smpl_layer, pose_params, shape_params)
    
    numbers = []
    for row in joint_locations:
        for number in row:
            numbers.append(float(number))

    return numbers

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)
    # Example usage
    # model_path = './code/models/' # Update this path
    # model_path = 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    # model_path = '../data/model'
    model_path = './'
    smpl_layer = load_smpl_model(model_path)
    pose_input = str(sys.argv[1]).split(',')

    pose_params = []
    shape_params = []
    for param in pose_input:
        pose_params.append(float(param))

    shape_input = sys.argv[2].split(',')
    for param in shape_input:
        shape_params.append(float(param))
    # Random pose and shape parameters (you will replace these with your own)
    pose_params = torch.Tensor(pose_params)
    shape_params = torch.Tensor(shape_params)

    # Get joint locations
    joint_locations, nose = get_joint_locations(smpl_layer, pose_params, shape_params)
    joint_locations = torch.cat([joint_locations, nose.unsqueeze(0)], dim=0)


    numbers = []
    for row in joint_locations:
        for number in row:
            numbers.append(float(number))
    print(numbers)