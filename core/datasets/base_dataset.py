import copy
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import visualization.plot_3d_global as plot_3d
from core import Motion, MotionRep
from core.models.utils import default
from torch.utils import data
from tqdm import tqdm
from utils.quaternion import qinv, qrot, quaternion_to_cont6d

from .kinematics import getSkeleton

# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)


class Motion2Positions:

    def __init__(self, data_root: str, motion_rep: MotionRep):

        self.skeleton = getSkeleton(
            "./core/datasets/data/000021_pos.npy",
            motion_rep=motion_rep.value,
        )
        self.data_root = data_root

    def recover_root_rot_pos(self, data: Motion) -> Tuple[torch.Tensor, torch.Tensor]:

        rot_vel = data.root_params[..., 0]
        r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        """Get Y-axis rotation from rotation velocity"""
        r_rot_ang[..., 1:] = rot_vel[..., :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

        r_rot_quat = torch.zeros(data.root_params.shape[:-1] + (4,)).to(data.device)
        r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        r_rot_quat[..., 2] = torch.sin(r_rot_ang)

        r_pos = torch.zeros(data.root_params.shape[:-1] + (3,)).to(data.device)
        r_pos[..., 1:, [0, 2]] = data.root_params[..., :-1, 1:3]
        """Add Y-axis rotation to root position"""
        r_pos = qrot(qinv(r_rot_quat), r_pos)

        r_pos = torch.cumsum(r_pos, dim=-2)

        r_pos[..., 1] = data.root_params[..., 3]
        return r_rot_quat, r_pos

    def recover_from_rot(self, data: Motion) -> torch.Tensor:

        skeleton = getSkeleton(
            os.path.join(self.data_root, "motion_data/000021_pos.npy"),
            motion_rep=data.motion_rep.value,
        )

        data.tensor()

        if data.root_params is None:
            data.root_params = torch.zeros((data.rotations.shape[:-1] + (4,)))

        joints_num = data.nb_joints
        r_rot_quat, r_pos = self.recover_root_rot_pos(data)
        # print("rpos", r_pos)
        r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

        cont6d_params = data.rotations
        cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
        cont6d_params = cont6d_params.view(-1, joints_num, 6)

        positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

        return positions

    def recover_from_ric(self, data: Motion) -> torch.Tensor:

        data.tensor()

        if data.root_params is None:
            data.root_params = torch.zeros((data.positions.shape[:-1] + (4,)))

        joints_num = data.nb_joints
        if joints_num == 22 or joints_num == 52:
            r_rot_quat, r_pos = self.recover_root_rot_pos(data)
            # r_pos[:, [0, 2]] = 0
            positions = data.positions
            positions = positions.view(positions.shape[:-1] + (-1, 3)).to(torch.float32)

            """Add Y-axis rotation to local joints"""
            positions = qrot(
                qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)),
                positions,
            )

            """Add root XZ to joints"""
            positions[..., 0] += r_pos[..., 0:1]
            positions[..., 2] += r_pos[..., 2:3]

            """Concate root and joints"""
            positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

        else:
            positions = data.positions
            positions = positions.view(positions.shape[:-1] + (-1, 3)).float()

        return positions

    def __call__(
        self, motion: Motion, hml_rep=None, from_rotation=False
    ) -> torch.Tensor:
        hml_rep = motion.hml_rep if hml_rep is None else hml_rep

        if "p" not in hml_rep or from_rotation is True:
            xyz = self.recover_from_rot(motion)
        else:
            xyz = self.recover_from_ric(motion)

        return xyz


def split_hands(hand_data: Motion) -> Tuple[Motion, Motion]:

    left_hand = Motion(MotionRep.LEFT_HAND, hml_rep=hand_data.hml_rep)
    right_hand = Motion(MotionRep.RIGHT_HAND, hml_rep=hand_data.hml_rep)
    hml_rep = hand_data.hml_rep
    if "p" in hml_rep:
        left_hand.positions = hand_data.positions[..., :45]
        right_hand.positions = hand_data.positions[..., 45:]
    if "r" in hml_rep:
        left_hand.rotations = hand_data.rotations[..., :90]
        right_hand.rotations = hand_data.rotations[..., 90:]
    if "v" in hml_rep:
        left_hand.velocity = hand_data.velocity[..., :45]
        right_hand.velocity = hand_data.velocity[..., 45:]

    return left_hand, right_hand


class BaseMotionDataset(data.Dataset):
    def __init__(
        self,
        dataset_root="/srv/hays-lab/scratch/sanisetty3/motionx",
        motion_rep=MotionRep.FULL,
        hml_rep="gprvc",
    ) -> None:
        """Initializes the BaseMotionDataset class.

        Args:
            dataset_root (str): The root directory of the dataset.
            train_mode (str, optional): The training motion_rep. Defaults to "full".
            use_rotation (bool, optional): Whether to use rotation. Defaults to True.
        """

        self.motion_rep = motion_rep
        self.hml_rep = hml_rep

        ## Making sure hand does not have global params

        if self.motion_rep in [
            MotionRep.HAND,
            MotionRep.LEFT_HAND,
            MotionRep.RIGHT_HAND,
        ]:
            self.hml_rep = self.hml_rep.replace("g", "").replace("c", "")

        self.data_root = dataset_root

        # self.mean = np.load(os.path.join(self.data_root, "motion_data/Mean_cl.npy"))
        # self.std = np.load(os.path.join(self.data_root, "motion_data/Std_cl.npy"))
        self.mean = np.load("./core/datasets/data/Mean_cl.npy")
        self.std = np.load("./core/datasets/data/Std_cl.npy")
        self.body_mean, self.hand_mean, self.full_mean = self.hmldata_process(self.mean)
        self.body_std, self.hand_std, self.full_std = self.hmldata_process(self.std)

        self.left_hand_mean, self.right_hand_mean = split_hands(self.hand_mean)
        self.left_hand_std, self.right_hand_std = split_hands(self.hand_std)

        self.mot2pos = Motion2Positions(self.data_root, self.motion_rep)

    @property
    def nb_joints(self):
        if self.motion_rep == MotionRep.FULL:
            return 52
        elif self.motion_rep == MotionRep.BODY:
            return 22
        elif self.motion_rep == MotionRep.HAND:
            return 30
        elif self.motion_rep == MotionRep.LEFT_HAND:
            return 15
        elif self.motion_rep == MotionRep.RIGHT_HAND:
            return 15

    @property
    def motion_dim(self):
        dim = 0

        if "g" in self.hml_rep:
            dim += 4
        if "p" in self.hml_rep:
            if self.motion_rep == MotionRep.BODY or self.motion_rep == MotionRep.FULL:
                dim += (self.nb_joints - 1) * 3
            else:
                dim += (self.nb_joints) * 3
        if "r" in self.hml_rep:
            if self.motion_rep == MotionRep.BODY or self.motion_rep == MotionRep.FULL:
                dim += (self.nb_joints - 1) * 6
            else:
                dim += (self.nb_joints) * 6
        if "v" in self.hml_rep:
            dim += self.nb_joints * 3
        if "c" in self.hml_rep:
            dim += 4

        return dim

    def inv_transform(self, data: Motion) -> Motion:
        """Inverse transforms the data.

        Args:
            data (torch.Tensor): The input data.
            train_mode (Optional[str], optional): The training motion_rep. Defaults to None.

        Returns:
            torch.Tensor: The inverse-transformed data.
        """
        motion_rep = data.motion_rep

        inv_data = data
        # copy.deepcopy(data)

        if motion_rep == MotionRep.FULL:
            inv_data.inv_transform(self.mean, self.std)
        elif motion_rep == MotionRep.HAND:
            inv_data.inv_transform(self.hand_mean, self.hand_std)
        elif motion_rep == MotionRep.BODY:
            inv_data.inv_transform(self.body_mean, self.body_std)

        elif motion_rep == MotionRep.LEFT_HAND:
            inv_data.inv_transform(self.left_hand_mean, self.left_hand_std)
        elif motion_rep == MotionRep.RIGHT_HAND:
            inv_data.inv_transform(self.right_hand_mean, self.right_hand_std)

        return inv_data

    def transform(self, data: Motion) -> Motion:
        motion_rep = data.motion_rep
        trn_data = copy.deepcopy(data)

        if motion_rep == MotionRep.FULL:
            trn_data.transform(self.mean, self.std)
        elif motion_rep == MotionRep.HAND:
            trn_data.transform(self.hand_mean, self.hand_std)
        elif motion_rep == MotionRep.BODY:
            trn_data.transform(self.body_mean, self.body_std)

        elif motion_rep == MotionRep.LEFT_HAND:
            trn_data.transform(self.left_hand_mean, self.left_hand_std)
        elif motion_rep == MotionRep.RIGHT_HAND:
            trn_data.transform(self.right_hand_mean, self.right_hand_std)

        return trn_data

    def toMotion(
        self,
        motion: Union[torch.Tensor, np.ndarray],
        motion_rep: MotionRep = None,
        hml_rep: str = None,
    ):

        assert len(motion.shape) == 2, "remove batch dimension"
        if hml_rep is None:
            hml_rep = self.hml_rep
        if motion_rep is None:
            motion_rep = self.motion_rep

        if motion_rep == MotionRep.FULL:
            joint_num = 52
        elif motion_rep == MotionRep.BODY:
            joint_num = 22
        elif motion_rep == MotionRep.HAND:
            joint_num = 30
        elif motion_rep == MotionRep.LEFT_HAND or motion_rep == MotionRep.RIGHT_HAND:
            joint_num = 15

        split_seq = []

        if "g" in hml_rep and (
            motion_rep == MotionRep.BODY or motion_rep == MotionRep.FULL
        ):
            split_seq.append(4)
        if "p" in hml_rep:
            if motion_rep == MotionRep.BODY or motion_rep == MotionRep.FULL:
                split_seq.append((joint_num - 1) * 3)
            else:
                split_seq.append((joint_num) * 3)
        if "r" in hml_rep:
            if motion_rep == MotionRep.BODY or motion_rep == MotionRep.FULL:
                split_seq.append((joint_num - 1) * 6)
            else:
                split_seq.append((joint_num) * 6)
        if "v" in hml_rep:
            split_seq.append(joint_num * 3)
        if "c" in hml_rep and (
            motion_rep == MotionRep.BODY or motion_rep == MotionRep.FULL
        ):
            split_seq.append(4)

        params = np.split(motion, np.cumsum(split_seq), -1)[:-1]
        hml_prm = dict(zip(list(hml_rep), params))

        motion_ = Motion(
            motion_rep=motion_rep,
            hml_rep=hml_rep,
            root_params=hml_prm["g"] if "g" in hml_rep else None,
            positions=hml_prm["p"] if "p" in hml_rep else None,
            rotations=hml_prm["r"] if "r" in hml_rep else None,
            velocity=hml_prm["v"] if "v" in hml_rep else None,
            contact=hml_prm["c"] if "c" in hml_rep else None,
        )

        return motion_

    def hmldata_process(
        self,
        hml_data: np.array,
        joint_num: int = 52,
        body_joints: int = 22,
        hand_joints: int = 30,
        hml_rep: Optional[str] = None,
    ):
        """Processes the HML data.

        Args:
            hml_data (np.array): The input HML data.
            joint_num (int, optional): The number of joints. Defaults to 52.
            body_joints (int, optional): The number of body joints. Defaults to 22.
            hand_joints (int, optional): The number of hand joints. Defaults to 30.

        Returns:
            tuple: The processed data.
        """

        if hml_rep is None:
            hml_rep = self.hml_rep

        split_seq = np.cumsum(
            [4, (joint_num - 1) * 3, (joint_num - 1) * 6, joint_num * 3, 4]
        )

        root_params, local_pos, local_rots, local_vels, foot = np.split(
            hml_data, split_seq, -1
        )[:-1]

        local_pos_body, local_pos_hand = np.split(
            local_pos, np.cumsum([(body_joints - 1) * 3, hand_joints * 3]), -1
        )[:-1]
        local_rots_body, local_rots_hand = np.split(
            local_rots, np.cumsum([(body_joints - 1) * 6, hand_joints * 6]), -1
        )[:-1]
        local_vel_body, local_vel_hand = np.split(
            local_vels, np.cumsum([(body_joints) * 3, hand_joints * 3]), -1
        )[:-1]

        body_motion = Motion(
            motion_rep=MotionRep.BODY,
            hml_rep=hml_rep,
            root_params=root_params if "g" in hml_rep else None,
            positions=local_pos_body if "p" in hml_rep else None,
            rotations=local_rots_body if "r" in hml_rep else None,
            velocity=local_vel_body if "v" in hml_rep else None,
            contact=foot if "c" in hml_rep else None,
        )
        hand_motion = Motion(
            motion_rep=MotionRep.HAND,
            hml_rep=hml_rep.replace("g", "").replace("c", ""),
            positions=local_pos_hand if "p" in hml_rep else None,
            rotations=local_rots_hand if "r" in hml_rep else None,
            velocity=local_vel_hand if "v" in hml_rep else None,
        )

        full_motion = Motion(
            motion_rep=MotionRep.FULL,
            hml_rep=hml_rep,
            root_params=root_params if "g" in hml_rep else None,
            positions=local_pos if "p" in hml_rep else None,
            rotations=local_rots if "r" in hml_rep else None,
            velocity=local_vels if "v" in hml_rep else None,
            contact=foot if "c" in hml_rep else None,
        )

        return body_motion, hand_motion, full_motion

    def process_hand(
        self, body: Motion, hand: Union[Motion, List[Motion]], mode="remove"
    ) -> Motion:

        def joinHands(hands: List[Motion]) -> Motion:
            l_hand, r_hand = hands[0], hands[1]
            hand = l_hand + r_hand
            hand.motion_rep = MotionRep.HAND
            hand.hml_rep = body.hml_rep.replace("g", "").replace("c", "")

            return hand

        if isinstance(hand, list):

            hand = joinHands(hand)

        if "p" not in body.hml_rep or "p" not in hand.hml_rep:
            return hand

        l_wrist_pos_param = body.positions[..., 19 * 3 : 20 * 3].reshape(-1, 1, 3)
        r_wrist_pos_param = body.positions[..., 20 * 3 : 21 * 3].reshape(-1, 1, 3)

        finger_param = hand.positions.reshape(-1, 30, 3)
        if mode == "remove":
            finger_param_left = finger_param[:, :15, :] - l_wrist_pos_param
            finger_param_right = finger_param[:, 15:, :] - r_wrist_pos_param
        elif mode == "add":
            finger_param_left = finger_param[:, :15, :] + l_wrist_pos_param
            finger_param_right = finger_param[:, 15:, :] + r_wrist_pos_param
        hand.positions[..., :45] = finger_param_left.reshape(
            hand.positions.shape[:-1] + (45,)
        )
        hand.positions[..., 45:90] = finger_param_right.reshape(
            hand.positions.shape[:-1] + (45,)
        )

        return hand

    def join_body_hands(
        self, body: Motion, hand: Union[Motion, List[Motion]]
    ) -> Motion:
        if isinstance(hand, list):
            hand = self.process_hand(body, hand, "add")

        full_params = body + hand

        return full_params

    def to_full_joint_representation(self, body: Motion, left: Motion, right: Motion):
        left_inv = self.inv_transform(left)
        right_inv = self.inv_transform(right)
        body_inv = self.inv_transform(body)
        processed_hand = self.process_hand(body_inv, [left_inv, right_inv], "add")
        joined_motion = self.join_body_hands(body_inv, processed_hand)
        joined_motion = self.transform(joined_motion)

        return joined_motion

    def get_processed_motion(
        self,
        motion: Union[np.ndarray, torch.Tensor],
        motion_rep: MotionRep = MotionRep.FULL,
        hml_rep: str = "gprvc",
    ) -> Motion:

        if motion_rep == MotionRep.FULL:

            body_params, hand_params, full_params = self.hmldata_process(
                motion, hml_rep=hml_rep
            )
            full_params = self.transform(full_params)
            return full_params

        elif motion_rep == MotionRep.BODY:

            body_params, hand_params, full_params = self.hmldata_process(
                motion, hml_rep=hml_rep
            )
            body_params = self.transform(body_params)
            return body_params

        elif motion_rep in [
            MotionRep.HAND,
            MotionRep.LEFT_HAND,
            MotionRep.RIGHT_HAND,
        ]:

            body_params, hand_params, full_params = self.hmldata_process(
                motion, hml_rep=hml_rep
            )

            if "p" in hml_rep:
                hand_params = self.process_hand(body_params, hand_params, "remove")
            hand_params = self.transform(hand_params)

            if motion_rep == MotionRep.HAND:
                return hand_params

            left_hand, right_hand = split_hands(hand_params)
            if motion_rep == MotionRep.LEFT_HAND:
                return left_hand
            else:
                return right_hand

    def to_xyz(
        self,
        motion: Motion,
        hml_rep=None,
        from_rotation=False,
        translation_external=None,
    ) -> torch.Tensor:
        if translation_external is not None:
            positions = self.mot2pos(motion, hml_rep, from_rotation)
            positions[..., 0] += translation_external[..., 0:1]
            positions[..., 2] += translation_external[..., 1:2]

            return positions

        return self.mot2pos(motion, hml_rep, from_rotation)

    def render_hml(
        self,
        motion: Union[np.ndarray, torch.Tensor, Motion],
        save_path: str = "motion",
        zero_trans=False,
        zero_orient=False,
        from_rotation=False,
        translation_external=None,
    ):

        if not isinstance(motion, Motion):
            motion = self.toMotion(motion)

        motion = self.inv_transform(motion)
        motion.tensor()
        if zero_trans and motion.root_params is not None:
            motion.root_params[:, [1, 2]] = 0
        if zero_orient and motion.root_params is not None:
            motion.root_params[:, 0] = 0
        xyz = self.to_xyz(
            motion,
            from_rotation=from_rotation,
            translation_external=translation_external,
        ).cpu()

        if len(xyz.shape) > 3:
            xyz = xyz[0]

        # print("xyz", xyz[:, 0])

        plot_3d.render(
            np.array(xyz),
            save_path,
        )

    def findAllFile(self, base):
        file_path = []
        for root, ds, fs in os.walk(base, followlinks=True):
            for f in fs:
                fullname = os.path.join(root, f)
                file_path.append(fullname)
        return file_path

    # @abstractmethod
    # def __len__(self):
    #     raise NotImplementedError("not implemented")

    # @abstractmethod
    # def __getitem__(self, item):
    #     raise NotImplementedError("not implemented")
