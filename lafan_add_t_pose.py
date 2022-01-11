import numpy as np
import sys
import os
sys.path.append("./motion")
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions

import copy

"""
    Description: bvh前补充一帧z-oriented T Pose
    Author: y00451286
    Date: 2022.01.11
"""

bvh_filename = r'E:\Sissi_Personal\dataset\lafan\ubisoft-laforge-animation-dataset\retarget\bvh_raw\aiming1_subject1.bvh'
anim, names, ftime = BVH.load(bvh_filename)
joint_names_2_idx_dict = {}
for i in range(len(names)):
    joint_names_2_idx_dict[names[i]] = i

# 以ref pose为基准来掰T Pose
ref_anim = copy.deepcopy(anim[:1])
ref_anim.positions[:, :1, :] = 0
ref_anim.rotations = Quaternions(np.array([[[1, 0, 0, 0]]], dtype = np.float32).repeat(ref_anim.shape[1], axis = 1))
ref_positions = Animation.positions_global(ref_anim)
print("ref_positions:", ref_positions)

## Step1. 构造重心坐标系(此处仅使用旋转, 位移不考虑)
## right_upleg_vec, left_arm_vec
right_upleg_vec = ref_positions[0][joint_names_2_idx_dict['RightUpLeg']] - ref_positions[0][joint_names_2_idx_dict['LeftUpLeg']]
left_arm_vec = ref_positions[0][joint_names_2_idx_dict['LeftArm']] - ref_positions[0][joint_names_2_idx_dict['RightArm']]
right_vec = (right_upleg_vec + left_arm_vec) / np.sqrt(((right_upleg_vec + left_arm_vec) ** 2).sum(axis = -1))[np.newaxis, ...]
print("right_vec:", right_vec)

## upvec
upleg_mid_position = (ref_positions[0][joint_names_2_idx_dict['RightUpLeg']] + ref_positions[0][joint_names_2_idx_dict['LeftUpLeg']]) / 2
arm_mid_position = (ref_positions[0][joint_names_2_idx_dict['LeftArm']] + ref_positions[0][joint_names_2_idx_dict['RightArm']]) / 2
up_vec = (arm_mid_position - upleg_mid_position) / np.sqrt(((arm_mid_position - upleg_mid_position) ** 2).sum(axis = -1))[np.newaxis, ...]
# print("up_vec:", up_vec)

forward_vec = np.cross(up_vec, right_vec)
forward_vec = forward_vec / np.sqrt((forward_vec ** 2).sum(axis = -1))[np.newaxis, ...]
print("forward_vec:", forward_vec)

up_vec = np.cross(right_vec, forward_vec)
up_vec = up_vec / np.sqrt((up_vec ** 2).sum(axis = -1))[np.newaxis, ...]
print("up_vec:", up_vec)

root_world_rotmat = np.concatenate((-right_vec[..., np.newaxis].astype(np.float32),
                                    up_vec[..., np.newaxis].astype(np.float32),
                                    forward_vec[..., np.newaxis].astype(np.float32)), axis = 1)
# print("root_world_rotmat:", root_world_rotmat)
root_world_quat = Quaternions.from_transforms(root_world_rotmat[np.newaxis])
print("root_world_quat:", root_world_quat)

# 计算重心坐标系: R(root_local->root_world) * R(root_world) = R(root_local) ==> I
rot_root_local_2_root_world = root_world_quat[0].__neg__()

## Step2. 在重心坐标系下掰骨骼
# 将所有骨段的offset全部转换到重心坐标系下
print("ref_anim.offsets:", ref_anim.offsets)
ref_anim_local_offsets = rot_root_local_2_root_world * ref_anim.offsets
print("ref_anim_local_offsets shape:", ref_anim_local_offsets.shape)

x_axis = np.array([1, 0, 0])
y_axis = np.array([0, 1, 0])
z_axis = np.array([0, 0, 1])
# 定义骨段在重心坐标系下需掰到的朝向
# 对骨段的父joint施加旋转，同父joint的骨段定义一次即可
# parent_name, child_name, parent_2_child_direction
# 肩膀处的节点可能需要手动指定一下!!!
bone_target_direction_list = [["Hips", "LeftUpLeg", x_axis],
                              ["LeftUpLeg", "LeftLeg", -y_axis],
                              ["LeftLeg", "LeftFoot", -y_axis],
                              ["LeftFoot", "LeftToe", z_axis],
                              ["RightUpLeg", "RightLeg", -y_axis],
                              ["RightLeg", "RightFoot", -y_axis],
                              ["RightFoot", "RightToe", z_axis],
                              ["Spine", "Spine1", y_axis],
                              ["Spine1", "Spine2", y_axis],
                              ["Spine2", "Neck", y_axis],
                              ["Neck", "Head", y_axis],
                              ["LeftShoulder", "LeftArm", x_axis],
                              ["LeftArm", "LeftForeArm", x_axis],
                              ["LeftForeArm", "LeftHand", x_axis],
                              ["RightShoulder", "RightArm", -x_axis],
                              ["RightArm", "RightForeArm", -x_axis],
                              ["RightForeArm", "RightHand", -x_axis]]

# 所有joints在root局部坐标系下的旋转
rot_all_joints_in_root_local = Quaternions(np.array([[1, 0, 0, 0]], dtype = np.float32).repeat(ref_anim.shape[1], axis = 0))
for pair_idx, joint_pair in enumerate(bone_target_direction_list):
    parent_idx = joint_names_2_idx_dict[joint_pair[0]]
    child_idx = joint_names_2_idx_dict[joint_pair[1]]
    unit_offset = ref_anim_local_offsets[child_idx] / np.sqrt((ref_anim_local_offsets[child_idx] ** 2).sum(axis = -1))
    print(joint_pair[0], joint_pair[1], ", source:", unit_offset, joint_pair[2])
    # 父joint在root局部坐标系下需要的旋转
    # 旋转轴必须是人平面的垂直轴(需要限定旋转轴, 否则得到的将是最短的旋转量的旋转轴)
    # 在靠近0度/180度的时候容易这种方法容易算错!!!
    rot_all_joints_in_root_local[parent_idx] = Quaternions.between(unit_offset, joint_pair[2])
    print("rot_all_joints_in_root_local[parent_idx]:", rot_all_joints_in_root_local[parent_idx])
#print(rot_all_joints_in_root_local.angle_axis())

for idx in range(ref_anim.shape[1]):
    parent_idx = ref_anim.parents[idx]
    ref_anim.rotations[0][idx] = rot_all_joints_in_root_local[parent_idx].__neg__() * rot_all_joints_in_root_local[idx]

BVH.save("./lafan_test.bvh", ref_anim, names, ftime)










