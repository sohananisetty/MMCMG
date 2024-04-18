import os
from dataclasses import dataclass

import numpy as np
import torch
import visualization.Animation as Animation
import visualization.BVH_mod as BVH
from torch import nn
from tqdm import tqdm
from visualization.InverseKinematics import BasicInverseKinematics
from visualization.Quaternions import Quaternions
from visualization.remove_fs import *
from visualization.utils.quat import between, fk, ik, ik_rot


class SMPLXParam:

    def __init__(self) -> None:
        self.template = BVH.load(
            "./visualization/data/smplx_template.bvh", need_quater=True
        )
        self.kinematic_chain = [
            [0, 1, 4, 7, 10],
            [0, 2, 5, 8, 11],
            [0, 3, 6, 9, 12, 15],
            [9, 13, 16, 18, 20],
            [9, 14, 17, 19, 21],
            [20, 22, 23, 24],
            [20, 25, 26, 27],
            [20, 28, 29, 30],
            [20, 31, 32, 33],
            [20, 34, 35, 36],
            [21, 37, 38, 39],
            [21, 40, 41, 42],
            [21, 43, 44, 45],
            [21, 46, 47, 48],
            [21, 49, 50, 51],
        ]
        self.end_points = [4, 8, 13, 17, 21]

        self.re_order = [
            0,
            1,
            4,
            7,
            10,
            2,
            5,
            8,
            11,
            3,
            6,
            9,
            12,
            15,
            13,
            16,
            18,
            20,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            14,
            17,
            19,
            21,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
        ]
        self.re_order_inv = [
            0,
            1,
            5,
            9,
            2,
            6,
            10,
            3,
            7,
            11,
            4,
            8,
            12,
            14,
            33,
            13,
            15,
            34,
            16,
            35,
            17,
            36,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
        ]
        self.parents = self.template.parents


class SMPLParam:

    def __init__(self) -> None:
        self.template = BVH.load(
            "./visualization/data/smpl_template.bvh", need_quater=True
        )
        self.kinematic_chain = [
            [0, 1, 4, 7, 10],
            [0, 2, 5, 8, 11],
            [0, 3, 6, 9, 12, 15],
            [9, 13, 16, 18, 20],
            [9, 14, 17, 19, 21],
        ]
        self.end_points = [4, 8, 13, 17, 21]

        self.re_order = [
            0,
            1,
            4,
            7,
            10,
            2,
            5,
            8,
            11,
            3,
            6,
            9,
            12,
            15,
            13,
            16,
            18,
            20,
            14,
            17,
            19,
            21,
        ]
        self.re_order_inv = [
            0,
            1,
            5,
            9,
            2,
            6,
            10,
            3,
            7,
            11,
            4,
            8,
            12,
            14,
            18,
            13,
            15,
            19,
            16,
            20,
            17,
            21,
        ]
        self.parents = self.template.parents


class Joint2BVHConvertor:
    def __init__(self, mode="smplx"):

        if mode == "smplx":
            param = SMPLXParam()
        else:
            param = SMPLParam()
        self.template = param.template

        self.re_order = param.re_order

        self.re_order_inv = param.re_order_inv

        self.end_points = (
            param.end_points
        )  # left_foot right_foot head left_wrist right_wrist

        self.template_offset = self.template.offsets.copy()

        self.parents = param.parents

    def convert(self, positions, filename, iterations=10, foot_ik=True, fps=30):
        """
        Convert the SMPL joint positions to Mocap BVH
        :param positions: (N, 52, 3)
        :param filename: Save path for resulting BVH
        :param iterations: iterations for optimizing rotations, 10 is usually enough
        :param foot_ik: whether to enfore foot inverse kinematics, removing foot slide issue.
        :return:
        """
        positions = positions[:, self.re_order]
        new_anim = self.template.copy()
        new_anim.rotations = Quaternions.id(positions.shape[:-1])
        new_anim.positions = new_anim.positions[0:1].repeat(positions.shape[0], axis=-0)
        new_anim.positions[:, 0] = positions[:, 0]

        if foot_ik:
            positions = remove_fs(
                positions,
                None,
                fid_l=(3, 4),
                fid_r=(7, 8),
                interp_length=5,
                force_on_floor=True,
            )
        ik_solver = BasicInverseKinematics(
            new_anim, positions, iterations=iterations, silent=True
        )
        new_anim = ik_solver()

        # BVH.save(filename, new_anim, names=new_anim.names, frametime=1 / 20, order='zyx', quater=True)
        glb = Animation.positions_global(new_anim)[:, self.re_order_inv]
        if filename is not None:
            BVH.save(
                filename,
                new_anim,
                names=new_anim.names,
                frametime=1 / fps,
                order="zyx",
                quater=True,
            )
        return new_anim, glb
