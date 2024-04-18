import codecs as cs
import itertools
import json
import math
import os
import random
from glob import glob
from os.path import join as pjoin
from typing import Dict, List, Optional, Tuple

import clip
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from core import MotionRep
from core.datasets.base_dataset import BaseMotionDataset
from core.datasets.conditioner import ConditionProvider
from torch.utils import data
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

genre_dict = {
    "mBR": "Break",
    "mPO": "Pop",
    "mLO": "Lock",
    "mMH": "Middle_Hip-hop",
    "mLH": "LAstyle Hip-hop",
    "mHO": "House",
    "mWA": "Waack",
    "mKR": "Krump",
    "mJS": "Street_Jazz",
    "mJB": "Ballet_Jazz",
}
inv_genre_dict = {v: k for k, v in genre_dict.items()}

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
            MotionAudioTextDataset(
                dataset_name,
                dataset_root=dataset_args.dataset_root,
                split=split,
                motion_min_length_s=dataset_args.motion_min_length_s,
                motion_max_length_s=dataset_args.motion_max_length_s,
                audio_rep=dataset_args.audio_rep,
                motion_rep=dataset_args.motion_rep,
                hml_rep=dataset_args.hml_rep,
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


def load_dataset_gen(
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
            MotionIndicesAudioTextDataset(
                dataset_name,
                dataset_root=dataset_args.dataset_root,
                split=split,
                motion_min_length_s=dataset_args.motion_min_length_s,
                motion_max_length_s=dataset_args.motion_max_length_s,
                audio_rep=dataset_args.audio_rep,
                motion_rep=dataset_args.motion_rep,
                fps=dataset_args.fps / dataset_args.down_sampling_ratio,
                window_size_s=dataset_args.window_size_s,
                hml_rep=dataset_args.hml_rep,
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


def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
    assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
    *shape, channels, length = wav.shape
    if target_channels == 1:
        wav = wav.mean(-2, keepdim=True)
    elif target_channels == 2:
        wav = wav.expand(*shape, target_channels, length)
    elif channels == 1:
        wav = wav.expand(target_channels, -1)
    else:
        raise RuntimeError(
            f"Impossible to convert from {channels} to {target_channels}"
        )
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


def default(val, d):
    return val if val is not None else d


class MotionAudioTextDataset(BaseMotionDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        audio_rep: str = "encodec",
        motion_rep: str = "full",
        hml_rep: str = "gprvc",
        motion_min_length_s=2,
        motion_max_length_s=10,
        window_size=None,
        sampling_rate: int = 16000,
        fps: int = 30,
        split: str = "train",
        **kwargs,
    ):
        super().__init__(dataset_root, MotionRep(motion_rep), hml_rep)
        self.dataset_name = dataset_name
        self.split = split
        self.fps = fps
        self.audio_rep = audio_rep

        self.window_size = window_size

        self.min_motion_length = motion_min_length_s * fps
        self.max_motion_length = motion_max_length_s * fps

        data_root = dataset_root

        self.text_dir = os.path.join(data_root, "texts/semantic_labels")
        self.motion_dir = os.path.join(data_root, "motion_data/new_joint_vecs")

        self.audio_dir = os.path.join(data_root, "audio")

        if self.audio_rep in ["encodec", "librosa"]:
            self.sampling_rate = 30
        else:
            self.sampling_rate = sampling_rate

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
        # print(self.id_list[item])

        name, ind, f_, to_ = self.id_list[item].rsplit("_", 3)
        f_, to_ = int(f_), int(to_)
        motion = np.load(os.path.join(self.motion_dir, name + ".npy"))

        text = self.text_list[item]
        try:

            if self.audio_rep in ["wav", "clap"]:

                wav, sr = torchaudio.load(
                    os.path.join(self.audio_dir, self.audio_rep, name + ".wav")
                )
                audio_data = np.array(convert_audio(wav, sr, self.sampling_rate, 1))
            elif self.audio_rep in ["encodec", "librosa"]:
                audio_data = np.load(
                    os.path.join(self.audio_dir, self.audio_rep, name + ".npy")
                )

            motion_s = motion.shape[0] // self.fps
            audio_s = audio_data.shape[0] // self.sampling_rate

            common_len_seconds = min(motion_s, audio_s)
            motion = motion[: int(common_len_seconds * self.fps)]
            audio_data = audio_data[: int(common_len_seconds * self.sampling_rate)]

        except:
            audio_data = None

        if to_ - f_ > self.min_motion_length:
            motion = motion[f_:to_]

        processed_motion = self.get_processed_motion(
            motion, motion_rep=self.motion_rep, hml_rep=self.hml_rep
        )

        return {
            "name": name,
            "motion": processed_motion,
            "text": text,
            "audio": audio_data,
        }


class MotionIndicesAudioTextDataset(BaseMotionDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        audio_rep: str = "encodec",
        motion_rep: str = "full",
        hml_rep: str = "gprvc",
        motion_min_length_s=3,
        motion_max_length_s=10,
        window_size_s=None,
        sampling_rate: int = 16000,
        downsample_ratio=4,
        fps: int = 30,
        split: str = "train",
        **kwargs,
    ):
        super().__init__(dataset_root, MotionRep(motion_rep), hml_rep)
        self.dataset_name = dataset_name
        self.split = split
        self.fps = fps
        self.audio_rep = audio_rep
        self.downsample_ratio = downsample_ratio
        self.motion_rep = motion_rep

        self.window_size = (
            int(window_size_s * self.fps) if window_size_s is not None else None
        )

        self.min_motion_length = motion_min_length_s * fps
        self.max_motion_length = motion_max_length_s * fps

        data_root = dataset_root

        self.text_dir = os.path.join(data_root, "texts/semantic_labels")

        if self.motion_rep in ["full", "hand"]:
            self.motion_ind_dir = os.path.join(data_root, f"indices/1024/body")
        else:
            self.motion_ind_dir = os.path.join(data_root, f"indices/1024/{motion_rep}")

        self.motion_dir = os.path.join(data_root, "motion_data/new_joint_vecs")
        self.audio_dir = os.path.join(data_root, "audio")

        if self.audio_rep in ["encodec", "librosa"]:
            self.sampling_rate = 30
        elif self.audio_rep == "clap":
            self.sampling_rate = 48000
        else:
            self.sampling_rate = int(sampling_rate)

        split_file = os.path.join(data_root, f"motion_data/{split}.txt")

        self.id_list = []
        self.text_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                if dataset_name + "/" in line:
                    try:
                        # if self.motion_rep in ["full", "body"]:
                        motion = np.load(
                            os.path.join(self.motion_ind_dir, line.strip())
                        ).squeeze()
                        seq_len = motion.shape[0]
                        if self.motion_rep in ["full", "hand"]:
                            left_hand_motion = np.load(
                                os.path.join(
                                    self.motion_ind_dir.replace("body", "left_hand"),
                                    line.strip(),
                                )
                            ).squeeze()
                            right_hand_motion = np.load(
                                os.path.join(
                                    self.motion_ind_dir.replace("body", "right_hand"),
                                    line.strip(),
                                )
                            ).squeeze()

                            seq_len = min(
                                motion.shape[0],
                                left_hand_motion.shape[0],
                                right_hand_motion.shape[0],
                            )

                        if seq_len < round(
                            default(self.window_size, self.min_motion_length)
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

    def _select_common_start_idx(self, motion, audio, max_length_s):
        motion_s = motion.shape[0] // self.fps
        audio_s = audio.shape[0] // self.sampling_rate

        common_len_seconds = min(motion_s, audio_s)
        motion = motion[: int(common_len_seconds * self.fps)]
        audio = motion[: int(common_len_seconds * self.sampling_rate)]

        if common_len_seconds > max_length_s:
            subset_idx_motion = np.random.randint(
                0, motion.shape[0] - int(max_length_s * self.fps) + 1
            )

            mot_start_s = subset_idx_motion // self.fps
            subset_idx_audio = int(mot_start_s * self.sampling_rate)

        else:
            return 0, 0

        return subset_idx_audio, subset_idx_motion

    def get_windowed_data(
        self, audio_data, motion, left_hand_motion=None, right_hand_motion=None
    ):
        if self.window_size == -1:
            mot_len_s = int(motion.shape[0] // self.fps)
            audio_len_s = mot_len_s

        else:
            mot_len_s = int(self.window_size // self.fps)
            audio_len_s = mot_len_s

        if audio_data is None:

            subset_idx_motion = random.randint(
                0, max(0, motion.shape[0] - int(mot_len_s * self.fps))
            )

        else:
            subset_idx_audio, subset_idx_motion = self._select_common_start_idx(
                motion, audio_data, mot_len_s
            )

            audio_data = audio_data[
                subset_idx_audio : subset_idx_audio
                + int(audio_len_s * self.sampling_rate)
            ]

        motion = motion[
            subset_idx_motion : subset_idx_motion + int(mot_len_s * self.fps)
        ]
        if self.motion_rep in ["full", "hand"]:
            left_hand_motion = left_hand_motion[
                subset_idx_motion : subset_idx_motion + int(mot_len_s * self.fps)
            ]
            right_hand_motion = right_hand_motion[
                subset_idx_motion : subset_idx_motion + int(mot_len_s * self.fps)
            ]

            return audio_data, motion, left_hand_motion, right_hand_motion

        return audio_data, motion

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:

        name, ind, f_, to_ = self.id_list[item].rsplit("_", 3)
        f_, to_ = int(f_), int(to_)
        motion = (
            np.load(os.path.join(self.motion_ind_dir, name + ".npy"))
            .squeeze()
            .reshape(-1, 1)
        )

        if self.motion_rep in ["full", "hand"]:
            left_hand_motion = (
                np.load(
                    os.path.join(
                        self.motion_ind_dir.replace("body", "left_hand"), name + ".npy"
                    )
                )
                .squeeze()
                .reshape(-1, 1)
            )
            right_hand_motion = (
                np.load(
                    os.path.join(
                        self.motion_ind_dir.replace("body", "right_hand"), name + ".npy"
                    )
                )
                .squeeze()
                .reshape(-1, 1)
            )

            min_length = min(
                motion.shape[0], left_hand_motion.shape[0], right_hand_motion.shape[0]
            )
            motion = motion[:min_length]
            left_hand_motion = left_hand_motion[:min_length]
            right_hand_motion = right_hand_motion[:min_length]

        text = self.text_list[item]
        audio_name = name
        try:

            if "aist" in audio_name:
                for i in genre_dict.values():
                    if i in audio_name:
                        audio_name = (
                            f"aist/{(inv_genre_dict[i])}{str(random.randint(0, 5))}"
                        )

            if self.audio_rep in ["wav", "clap"]:

                audio_data, _ = librosa.load(
                    os.path.join(self.audio_dir, "wav", audio_name + ".wav"),
                    sr=self.sampling_rate,
                )  # sample rate should be 48000
                audio_data = audio_data.reshape(-1, 1)  ## T 1
            elif self.audio_rep in ["encodec", "librosa"]:
                audio_data = np.load(
                    os.path.join(self.audio_dir, self.audio_rep, audio_name + ".npy")
                )

            motion_s = (motion.shape[0]) // self.fps
            audio_s = audio_data.shape[0] // self.sampling_rate

            common_len_seconds = min(motion_s, audio_s)
            motion = motion[: int((common_len_seconds * self.fps))]
            if self.motion_rep == "full":
                left_hand_motion = left_hand_motion[
                    : int((common_len_seconds * self.fps))
                ]
                right_hand_motion = right_hand_motion[
                    : int((common_len_seconds * self.fps))
                ]

            audio_data = audio_data[: int(common_len_seconds * self.sampling_rate)]

        except:
            audio_data = None

        if motion[int(f_) : math.ceil(to_)].shape[0] > default(
            self.window_size, self.min_motion_length
        ):
            motion = motion[int(f_) : math.ceil(to_)]
            if self.motion_rep in ["full", "hand"]:
                left_hand_motion = left_hand_motion[int(f_) : math.ceil(to_)]
                right_hand_motion = right_hand_motion[int(f_) : math.ceil(to_)]

        if self.window_size is not None:

            if self.motion_rep in ["full", "hand"]:

                audio_data, motion, left_hand_motion, right_hand_motion = (
                    self.get_windowed_data(
                        audio_data, motion, left_hand_motion, right_hand_motion
                    )
                )

            else:
                audio_data, motion = self.get_windowed_data(audio_data, motion)

        if self.motion_rep == "full":
            final_motion = [
                motion.reshape(-1, 1),
                left_hand_motion.reshape(-1, 1),
                right_hand_motion.reshape(-1, 1),
            ]
        elif self.motion_rep == "hand":
            final_motion = [
                left_hand_motion.reshape(-1, 1),
                right_hand_motion.reshape(-1, 1),
            ]

        else:

            final_motion = motion  ## n 1/3

        return {
            "name": name,
            "motion": final_motion,
            "motion_rep": self.motion_rep,
            "text": text,
            "audio": audio_data,
        }


def simple_collate(
    samples: List[Tuple[torch.Tensor, str]],
    conditioner: ConditionProvider,
    permute: bool = False,
) -> Dict[str, torch.Tensor]:
    motions = []
    texts = []
    audios = []
    names = []

    for sample in samples:
        names.append(sample["name"])

        if isinstance(sample["motion"], list):

            motions.append(np.concatenate(sample["motion"], -1))
        else:
            motions.append(sample["motion"])

        texts.append(sample["text"])
        audios.append(sample["audio"])

    inputs, conditions = conditioner(
        raw_audio=audios,
        raw_motion=motions,
        raw_text=texts,
    )

    if permute:
        inputs["motion"] = (
            inputs["motion"][0].permute(0, 2, 1).to(torch.long),
            inputs["motion"][1].to(torch.bool),
        )  # [B, T, K] -> [B, K , T]

    inputs["names"] = np.array(names)
    inputs["texts"] = np.array(texts)

    return inputs, conditions
