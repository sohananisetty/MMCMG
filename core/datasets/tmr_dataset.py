import codecs as cs
import itertools
import json
import math
import os
import random
from glob import glob
from os.path import join as pjoin
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from core import MotionRep
from core.datasets.base_dataset import BaseMotionDataset
from core.datasets.conditioner import ConditionProvider
from torch.utils import data
from tqdm import tqdm

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
            TMRDataset(
                dataset_name,
                dataset_root=dataset_args.dataset_root,
                split=split,
                motion_min_length_s=dataset_args.motion_min_length_s,
                motion_max_length_s=dataset_args.motion_max_length_s,
                motion_rep=dataset_args.motion_rep,
                hml_rep=dataset_args.hml_rep,
                window_size_s=dataset_args.window_size_s,
                fps=dataset_args.fps,
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


class TMRDataset(BaseMotionDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        motion_rep: str = "full",
        hml_rep: str = "gprvc",
        motion_min_length_s=2,
        motion_max_length_s=10,
        window_size_s=None,
        fps: int = 30,
        split: str = "train",
    ):
        super().__init__(dataset_root, MotionRep(motion_rep), hml_rep)
        self.dataset_name = dataset_name
        self.split = split
        self.fps = fps

        self.window_size = (
            int(window_size_s * self.fps) if window_size_s is not None else None
        )

        self.min_motion_length = motion_min_length_s * fps
        self.max_motion_length = motion_max_length_s * fps

        data_root = dataset_root

        self.text_dir = os.path.join(data_root, "texts/semantic_labels")
        self.motion_dir = os.path.join(data_root, "motion_data/new_joint_vecs")

        split_file = os.path.join(data_root, f"motion_data/{split}.txt")

        self.id_list = []
        self.text_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                if dataset_name + "/" in line:
                    try:
                        motion = np.load(os.path.join(self.motion_dir, line.strip()))
                        if motion.shape[0] < default(
                            self.window_size, self.min_motion_length
                        ):
                            continue

                        if self.dataset_name == "humanml":
                            name_list, txt_list = self.load_humanml(line.strip())

                        else:
                            name_list, txt_list = self.load_txt(line.strip())

                        self.id_list.extend(name_list)
                        self.text_list.extend(txt_list)

                    except:
                        continue

        print(
            f"Total number of motions {dataset_name}: {len(self.id_list)} and texts {len(self.text_list)}"
        )

    def __len__(self) -> int:
        return len(self.id_list)

    def load_txt(self, name):
        name = name[:-4]
        new_name = f"{name}_0_0_0"
        name_list = []
        txt_list = []

        with open(os.path.join(self.text_dir, name + ".txt"), "r") as f:
            for line in f.readlines():
                name_list.append(new_name)
                txt_list.append(line.strip())

        return name_list, txt_list

    def load_humanml(self, name):
        name = name[:-4]
        # data_dict = {}
        name_list = []
        txt_list = []
        with open(os.path.join(self.text_dir, name + ".txt"), "r") as f:
            for index, line in enumerate(f.readlines()):
                line_split = line.strip().split("#")
                caption = line_split[0]
                tokens = line_split[1].split(" ")
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag
                new_name = (
                    f"{name}_{index}_{int(f_tag * self.fps)}_{int(to_tag * self.fps)}"
                )

                name_list.append(new_name)
                txt_list.append(caption)

        return name_list, txt_list

    def load_beat(self, name):
        name = name[:-4]
        id, person_name, recording_type, start, end = name.split("_")
        if id in (list(np.arange(6, 11)) + list(np.arange(21, 31))):
            gender = "woman"
        else:
            gender = "man"

        new_name = f"{name}_0_0"
        name_list = []
        txt_list = []
        with open(
            os.path.join(
                self.text_dir.replace("semantic_labels", "body_texts"), name + ".json"
            ),
            "r",
        ) as outfile:
            frame_texts = json.load(outfile)

        emotion = frame_texts.pop("emotion")
        if emotion == "neutral":
            emotion = "a neutral tone"

        prefix = (
            f"a {gender} is giving a speech with {emotion} on "
            if recording_type == 0
            else f"a {gender} is having a conversation with {emotion} on "
        )

        items = list(frame_texts.values())

        items.insert(0, prefix)
        # sentence = (" ".join(list(dict.fromkeys(items)))).strip()
        name_list.append(new_name)
        txt_list.append(" ".join(items))

        return name_list, txt_list

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:

        name, ind, f_, to_ = self.id_list[item].rsplit("_", 3)
        f_, to_ = int(f_), int(to_)
        motion = np.load(os.path.join(self.motion_dir, name + ".npy"))

        text = self.text_list[item]

        if motion[int(f_) : math.ceil(to_)].shape[0] > default(
            self.window_size, self.min_motion_length
        ):
            motion = motion[f_:to_]

        if self.window_size is not None:
            if self.window_size == -1:
                mot_len = (motion).shape[0]
            else:
                mot_len = self.window_size

            idx = random.randint(0, motion.shape[0] - mot_len)

            motion = motion[idx : idx + mot_len]

        else:
            if motion.shape[0] > self.max_motion_length:
                idx = random.randint(0, motion.shape[0] - self.max_motion_length)
                motion = motion[idx : idx + self.max_motion_length]

                rand_len = random.randint(
                    self.min_motion_length, self.max_motion_length
                )
                idx = random.randint(0, motion.shape[0] - rand_len)
                motion = motion[idx : idx + rand_len]

        motion = motion[: (motion.shape[0] // 4) * 4]

        processed_motion = self.get_processed_motion(
            motion, motion_rep=self.motion_rep, hml_rep=self.hml_rep
        )

        return {
            "name": name,
            "motion": processed_motion,
            "text": text,
        }


def simple_collate(
    samples: List[Tuple[torch.Tensor, str]],
    conditioner: ConditionProvider,
) -> Dict[str, torch.Tensor]:

    inputs = {}
    conditions = {}

    names = []
    lens = []
    motions = []
    texts = []

    device = conditioner.device

    for sample in samples:
        names.append(sample["name"])
        motions.append(sample["motion"]())
        lens.append(len(sample["motion"]))
        texts.append(sample["text"])

    motion, mask = conditioner._get_motion_features(
        motion_list=motions,
    )
    text, text_mask = conditioner._get_text_features(
        raw_text=texts,
    )

    inputs["names"] = np.array(names)
    inputs["texts"] = np.array(texts)
    inputs["motion"] = (torch.Tensor(motion).to(device), torch.Tensor(mask).to(device))
    inputs["lens"] = torch.Tensor(lens)
    conditions["text"] = (
        torch.Tensor(text).to(device),
        torch.Tensor(text_mask).to(device),
    )

    return inputs, conditions
