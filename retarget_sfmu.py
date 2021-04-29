import re
import sys
import numpy as np
import scipy.io as io

sys.path.append('../../motion')

import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots
from InverseKinematics import BasicJacobianIK, JacobianInverseKinematics


rest, names, ftime = BVH.load("0005_2FeetJump001.bvh")
# rest_targets: F*J*3
rest_targets = Animation.positions_global(rest)
# max height of rest pose(T Pose, 0 frame pose)
rest_height = rest_targets[0,:,1].max()

rest.positions = rest.offsets[np.newaxis]
rest.rotations.qs = rest.orients.qs[np.newaxis]
# 别的数据集也用了
BVH.save('./sfmu/rest.bvh', rest, names)

filename = "0007_Cartwheel001.bvh"

anim, _, ftime = BVH.load(filename)
anim_targets = Animation.positions_global(anim)
anim_height = anim_targets[0, :, 1].max()
# 第一个序列骨架高度/当前序列骨架高度 * 所有点的position
targets = (rest_height / anim_height) * anim_targets[1:]

anim = anim[1:]
anim.orients.qs = rest.orients.qs.copy()
anim.offsets = rest.offsets.copy()
# 当前序列root的position变成缩放之后的root的position
anim.positions[:, 0] = targets[:, 0]
anim.positions[:, 1:] = rest.positions[:, 1:].repeat(len(targets), axis=0)

targetmap = {}

for ti in range(targets.shape[1]):
    targetmap[ti] = targets[:, ti]
# 当前序列的运动数据全部重定向到第一个骨架上去
# 目的是要用第一个序列的骨架实现   类似于当前序列的所有关节的position   ->targets已做过所有关节点的比例缩放
ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=2.0, silent=True)
ik()

BVH.save("0005_Cartwheel001.bvh", anim, names, ftime)
