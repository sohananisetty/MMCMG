import codecs as cs
import itertools
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from core import MotionRep
from core.datasets.base_dataset import BaseMotionDataset
from core.datasets.conditioner import ConditionProvider

dataset_names_default = [
    "animation",
    "humanml",
    "perform",
    "GRAB",
    "idea400",
    "humman",
    "beat",
    "game_motion",
    "music",
    "aist",
    "fitness",
    "moyo",
    "choreomaster",
    "dance",
    "kungfu",
    "EgoBody",
    "HAA500",
]


def load_dataset(
    dataset_args,
    dataset_names=dataset_names_default,
    split: str = "train",
    weight_scale: Optional[List[int]] = None,
):
    if weight_scale is None:
        weight_scale = [1] * len(dataset_names)
    assert len(dataset_names) == len(weight_scale), "mismatch in size"
    dataset_list = []
    weights = []
    for dataset_name in dataset_names:
        dataset_list.append(
            VQSMPLXMotionDataset(
                dataset_name,
                dataset_root=dataset_args.dataset_root,
                split=split,
                motion_min_length_s=dataset_args.motion_min_length_s,
                motion_max_length_s=dataset_args.motion_max_length_s,
                motion_rep=dataset_args.motion_rep,
                hml_rep=dataset_args.hml_rep,
                window_size=dataset_args.window_size,
            )
        )

    concat_dataset = torch.utils.data.ConcatDataset(dataset_list)

    if split != "train" or len(dataset_names) == 1:
        return concat_dataset, None, None

    for i, ds in enumerate(dataset_list):
        weights.append(
            [weight_scale[i] * concat_dataset.__len__() / (ds.__len__())] * ds.__len__()
        )

    weights = list(itertools.chain.from_iterable(weights))

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(weights)
    )

    return concat_dataset, sampler, weights


def default(val, d):
    return val if val is not None else d


class VQSMPLXMotionDataset(BaseMotionDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        motion_rep: str = "full",
        hml_rep: str = "gprvc",
        motion_min_length_s=2,
        motion_max_length_s=10,
        window_size=120,
        fps: int = 30,
        split: str = "train",
        **kwargs,
    ):
        super().__init__(dataset_root, MotionRep(motion_rep), hml_rep)
        self.dataset_name = dataset_name
        self.split = split
        self.fps = fps
        data_root = dataset_root

        self.window_size = window_size
        self.enable_var_len = True if window_size is None else False

        self.min_motion_length = motion_min_length_s * fps
        self.max_motion_length = motion_max_length_s * fps

        self.text_dir = os.path.join(data_root, "texts/semantic_labels")
        self.face_text_dir = os.path.join(data_root, "texts/face_texts")

        self.motion_dir = os.path.join(data_root, "motion_data/new_joint_vecs")

        split_file = os.path.join(data_root, f"motion_data/{split}.txt")

        self.id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                if dataset_name + "/" in line:
                    try:
                        motion = np.load(os.path.join(self.motion_dir, line.strip()))
                        min_length = (
                            default(self.window_size, self.min_motion_length)
                            if self.window_size != -1
                            else motion.shape[0]
                        )

                        if motion.shape[0] >= min_length:
                            self.id_list.append(line.strip())
                    except:
                        continue

        print(f"Total number of motions {dataset_name}: {len(self.id_list)}")

    def __len__(self) -> int:
        return len(self.id_list)

    def mask_augment(self, motion, perc_n=0.0, perc_d=0.0):
        n, d = motion.shape
        num_masked_n = int(n * perc_n)
        num_masked_d = int(d * perc_d)

        n_ind = list(np.random.choice(np.arange(n), num_masked_n))
        d_ind = list(np.random.choice(np.arange(d), num_masked_d))

        motion[n_ind, :] = 0
        motion[:, d_ind] = 0

        return motion

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        motion = np.load(os.path.join(self.motion_dir, self.id_list[item]))

        if self.enable_var_len:
            if self.max_motion_length < 0:
                mot_len = motion.shape[0]
            else:
                mot_len = np.random.randint(
                    self.window_size, min(motion.shape[0], self.max_motion_length)
                )

        else:
            if self.window_size == -1:
                mot_len = (motion).shape[0]

            else:
                mot_len = self.window_size

        idx = random.randint(0, motion.shape[0] - mot_len)

        motion = motion[idx : idx + mot_len]
        "Z Normalization"

        motion = motion[: (motion.shape[0] // 4) * 4]

        processed_motion = self.get_processed_motion(
            motion, motion_rep=self.motion_rep, hml_rep=self.hml_rep
        )

        return {
            "name": self.id_list[item],
            "motion": processed_motion,
        }


def simple_collate(
    samples: List[Tuple[torch.Tensor, str]],
    conditioner: ConditionProvider,
) -> Dict[str, torch.Tensor]:

    inputs = {}

    names = []
    lens = []
    motions = []
    device = conditioner.device

    for sample in samples:
        names.append(sample["name"])
        motions.append(sample["motion"]())
        lens.append(len(sample["motion"]))

    motion, mask = conditioner._get_motion_features(
        motion_list=motions,
        # max_length=max(lens),
    )

    inputs["names"] = np.array(names)
    inputs["lens"] = np.array(lens)
    inputs["motion"] = (torch.Tensor(motion).to(device), torch.Tensor(mask).to(device))

    return inputs
