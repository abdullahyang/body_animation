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
    return Jtr.squeeze(0)

if __name__ == '__main__':
    for i in range(0,len(sys.argv)):
        print(sys.argv[i])
    # Example usage
    # model_path = './code/models/' # Update this path
    # model_path = 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    model_path = './'
    smpl_layer = load_smpl_model(model_path)

    # Random pose and shape parameters (you will replace these with your own)
    pose_params = torch.randn(72)
    shape_params = torch.randn(10)

    # Get joint locations
    joint_locations = get_joint_locations(smpl_layer, pose_params, shape_params)
    print(joint_locations)