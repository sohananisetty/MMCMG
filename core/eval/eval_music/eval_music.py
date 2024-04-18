import math
import os

import librosa
import numpy as np
import torch
from core import MotionRep
from core.datasets.base_dataset import BaseMotionDataset
from core.models.generation.muse2 import generate_animation
from tqdm import tqdm
from utils.aist_metrics.calculate_beat_scores import (alignment_score,
                                                      motion_peak_onehot)
from utils.aist_metrics.calculate_fid_scores import (
    calculate_avg_distance, calculate_frechet_distance,
    calculate_frechet_feature_distance, extract_feature)
from utils.aist_metrics.features import kinetic, manual
from utils.motion_processing.hml_process import recover_from_ric


@torch.no_grad()
def evaluate_music_motion_vqvae(
    val_loader,  ##bs 1
    net,
    audio_feature_dir="/srv/hays-lab/scratch/sanisetty3/motionx/audio/librosa",
):

    result_features = {"kinetic": [], "manual": []}
    real_features = {"kinetic": [], "manual": []}

    beat_scores_real = []
    beat_scores_pred = []

    for aist_batch in tqdm(val_loader):

        mot_len = aist_batch["lens"][0]
        motion_name = aist_batch["names"][0]
        gt_motion = aist_batch["motion"][0]
        audio_path = os.path.join(
            audio_feature_dir,
            f"{motion_name}.npy",
        )
        print(audio_path)

        vqvae_output = net(
            motion=gt_motion,
            # mask=mask,
        )

        keypoints3d_gt = recover_from_ric(gt_motion[0, :mot_len], 22).cpu().numpy()
        keypoints3d_pred = (
            recover_from_ric(vqvae_output.decoded_motion[0, :mot_len], 22).cpu().numpy()
        )

        real_features["kinetic"].append(extract_feature(keypoints3d_gt, "kinetic"))
        real_features["manual"].append(extract_feature(keypoints3d_gt, "manual"))

        result_features["kinetic"].append(extract_feature(keypoints3d_pred, "kinetic"))
        result_features["manual"].append(extract_feature(keypoints3d_pred, "manual"))

        motion_beats = motion_peak_onehot(keypoints3d_gt)
        # get real data music beats
        audio_name = motion_name.split("_")[-2]

        audio_feature = np.load(audio_path)
        audio_beats = audio_feature[:mot_len, -1]  # last dim is the music beats
        # get beat alignment scores
        beat_score = alignment_score(audio_beats, motion_beats, sigma=1)
        beat_scores_real.append(beat_score)

        motion_beats = motion_peak_onehot(keypoints3d_pred)
        beat_score_pred = alignment_score(audio_beats, motion_beats, sigma=1)
        beat_scores_pred.append(beat_score_pred)

    FID_k, Dist_k = calculate_frechet_feature_distance(
        real_features["kinetic"], result_features["kinetic"]
    )
    FID_g, Dist_g = calculate_frechet_feature_distance(
        real_features["manual"], result_features["manual"]
    )

    FID_k2, Dist_k2 = calculate_frechet_feature_distance(
        real_features["kinetic"], real_features["kinetic"]
    )
    FID_g2, Dist_g2 = calculate_frechet_feature_distance(
        real_features["manual"], real_features["manual"]
    )

    print("FID_k: ", FID_k, "Diversity_k:", Dist_k)
    print("FID_g: ", FID_g, "Diversity_g:", Dist_g)

    print("FID_k_real: ", FID_k2, "Diversity_k_real:", Dist_k2)
    print("FID_g_real: ", FID_g2, "Diversity_g_real:", Dist_g2)

    print("\nBeat score on real data: %.3f\n" % (np.mean(beat_scores_real)))
    print("\nBeat score on reconstructed data: %.3f\n" % (np.mean(beat_scores_pred)))

    best_fid_k = FID_k if FID_k < best_fid_k else best_fid_k
    best_fid_g = FID_g if FID_g < best_fid_g else best_fid_g
    best_div_k = Dist_k if Dist_k > best_div_k else best_div_k
    best_div_g = Dist_g if Dist_g > best_div_g else best_div_g

    best_beat_align = (
        np.mean(beat_scores_real)
        if np.mean(beat_scores_real) > best_beat_align
        else best_beat_align
    )

    return best_fid_k, best_fid_g, best_div_k, best_div_g, best_beat_align


@torch.no_grad()
def evaluation_transformer(
    val_loader,
    condition_provider,
    bkn_to_motion,
    motion_generator,
    audio_feature_dir="/srv/hays-lab/scratch/sanisetty3/motionx/audio/librosa",
):

    result_features = {"kinetic": [], "manual": []}
    real_features = {"kinetic": [], "manual": []}
    base_dset = BaseMotionDataset(motion_rep=MotionRep.BODY, hml_rep="gpvc")

    beat_scores_real = []
    beat_scores_pred = []

    for inputs in tqdm(val_loader):

        gt_motion = inputs["motion"][0]
        motion_name = inputs["names"][0]
        text_ = inputs["texts"][0]
        audio_path = os.path.join(
            audio_feature_dir,
            f"{motion_name}.npy",
        )
        audio_feature = np.load(audio_path)
        bs, seq_len, d = inputs["motion"][0].shape

        # text_ = inputs["texts"][0]
        duration_s = math.ceil(seq_len / 30)

        all_ids = generate_animation(
            motion_generator,
            condition_provider,
            temperature=0.6,
            overlap=10,
            duration_s=duration_s,
            text=text_,
            aud_clip=audio_path.replace("librosa", "wav").replace("npy", "wav"),
            use_token_critic=True,
            timesteps=24,
        )

        pred_motion = bkn_to_motion(all_ids[:, :1], base_dset)

        keypoints3d_gt = recover_from_ric(gt_motion.detach().cpu(), 22)[0].numpy()
        keypoints3d_pred = recover_from_ric(pred_motion.detach().cpu(), 22)[0].numpy()

        # try:

        real_features["kinetic"].append(extract_feature(keypoints3d_gt, "kinetic"))
        real_features["manual"].append(extract_feature(keypoints3d_gt, "manual"))

        result_features["kinetic"].append(extract_feature(keypoints3d_pred, "kinetic"))
        result_features["manual"].append(extract_feature(keypoints3d_pred, "manual"))
        # except:
        #     continue

        motion_beats = motion_peak_onehot(keypoints3d_gt)

        audio_beats = audio_feature[:seq_len, -1]  # last dim is the music beats
        # get beat alignment scores
        beat_score = alignment_score(audio_beats, motion_beats, sigma=1)
        beat_scores_real.append(beat_score)

        motion_beats = motion_peak_onehot(keypoints3d_pred)
        beat_score_pred = alignment_score(audio_beats, motion_beats, sigma=1)
        beat_scores_pred.append(beat_score_pred)

    FID_k, Dist_k = calculate_frechet_feature_distance(
        real_features["kinetic"], result_features["kinetic"]
    )
    FID_g, Dist_g = calculate_frechet_feature_distance(
        real_features["manual"], result_features["manual"]
    )

    print("FID_k: ", FID_k, "Diversity_k:", Dist_k)
    print("FID_g: ", FID_g, "Diversity_g:", Dist_g)
    print("Beat score on real data: %.3f\n" % (np.mean(beat_scores_real)))
    print("Beat score on generated data: %.3f\n" % (np.mean(beat_scores_pred)))

    best_fid_k = FID_k
    best_fid_g = FID_g
    best_div_k = Dist_k
    best_div_g = Dist_g

    best_beat_align = (
        np.mean(beat_scores_real)
        if np.mean(beat_scores_real) > best_beat_align
        else best_beat_align
    )

    return best_fid_k, best_fid_g, best_div_k, best_div_g, best_beat_align
