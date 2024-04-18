import math

import torch
from core import MotionRep
from core.datasets.base_dataset import BaseMotionDataset
from core.eval.eval_text.helpers import (calculate_activation_statistics,
                                         calculate_diversity,
                                         calculate_frechet_distance,
                                         calculate_R_precision)
from core.models.generation.muse2 import generate_animation
from tqdm import tqdm


def get_latents(inputs, conditions, tmr):
    text_conds = conditions["text"]
    text_x_dict = {"x": text_conds[0], "mask": text_conds[1].to(torch.bool)}
    motion_x_dict = {"x": inputs[0], "mask": inputs[1].to(torch.bool)}
    motion_mask = motion_x_dict["mask"]
    text_mask = text_x_dict["mask"]
    t_motions, t_latents, t_dists = tmr(text_x_dict, mask=motion_mask, return_all=True)

    # motion -> motion
    m_motions, m_latents, m_dists = tmr(
        motion_x_dict, mask=motion_mask, return_all=True
    )
    return t_latents, m_latents


@torch.no_grad()
def evaluation_vqvae(
    val_loader,
    motion_vqvae,
    tmr,
):
    motion_vqvae.eval()
    tmr.eval()
    nb_sample = 0

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    motion_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    for inputs, conditions in tqdm(val_loader):

        with torch.no_grad():
            bs = inputs["motion"][0].shape[0]

            t_latents, m_latents = get_latents(inputs["motion"], conditions, tmr)

            # pred_pose_eval = torch.zeros_like(inputs["motion"][0])
            # for k in range(bs):
            #     lenn = int(inputs["lens"][k])
            #     vqvae_output = motion_vqvae(
            #         motion=inputs["motion"][0][k : k + 1, :lenn],
            #     )
            #     pred_pose_eval[k : k + 1, :lenn] = vqvae_output.decoded_motion
            pred_pose_eval = (
                motion_vqvae(inputs["motion"][0]).decoded_motion
                * inputs["motion"][1][..., None]
            )
            t_latents_pred, m_latents_pred = get_latents(
                (pred_pose_eval, inputs["motion"][1]), conditions, tmr
            )

            motion_list.append(m_latents)
            motion_pred_list.append(m_latents_pred)

            temp_R, temp_match = calculate_R_precision(
                t_latents.cpu().numpy(), m_latents.cpu().numpy(), top_k=3, sum_all=True
            )
            R_precision_real += temp_R
            matching_score_real += temp_match
            temp_R, temp_match = calculate_R_precision(
                t_latents_pred.cpu().numpy(),
                m_latents_pred.cpu().numpy(),
                top_k=3,
                sum_all=True,
            )
            R_precision += temp_R
            matching_score_pred += temp_match
            nb_sample += bs

    motion_annotation_np = torch.cat(motion_list).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"-->  FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    real_metrics = (0.0, diversity_real, R_precision_real, matching_score_real)
    pred_metrics = (fid, diversity, R_precision, matching_score_pred)

    return real_metrics, pred_metrics


@torch.no_grad()
def evaluation_transformer(
    val_loader, condition_provider, bkn_to_motion, motion_generator, tmr
):
    motion_generator.eval()
    tmr.eval()
    nb_sample = 0
    base_dset = BaseMotionDataset(motion_rep=MotionRep.BODY, hml_rep="gpvc")

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    for inputs, conditions in val_loader:

        motion_list = []
        motion_pred_list = []
        R_precision_real = 0
        R_precision = 0
        matching_score_real = 0
        matching_score_pred = 0
        with torch.no_grad():
            bs = inputs["motion"][0].shape[0]

            t_latents, m_latents = get_latents(inputs["motion"], conditions, tmr)

            pred_pose_eval = torch.zeros_like(inputs["motion"][0])
            for k in range(bs):
                lenn = int(inputs["lens"][k])
                text_ = inputs["texts"][k]
                duration_s = math.ceil(lenn / 30)
                all_ids_body = generate_animation(
                    motion_generator,
                    condition_provider,
                    temperature=0.6,
                    overlap=10,
                    duration_s=duration_s,
                    text=text_,
                    use_token_critic=True,
                    timesteps=24,
                )
                gen_motion = bkn_to_motion(all_ids_body[:, :1], base_dset)
                pred_pose_eval[k : k + 1, :lenn] = gen_motion()[:lenn][None]

            t_latents_pred, m_latents_pred = get_latents(
                (pred_pose_eval, inputs["motion"][1]), conditions, tmr
            )

            motion_list.append(m_latents)
            motion_pred_list.append(m_latents_pred)

            temp_R, temp_match = calculate_R_precision(
                t_latents.cpu().numpy(), m_latents.cpu().numpy(), top_k=3, sum_all=True
            )
            R_precision_real += temp_R
            matching_score_real += temp_match
            temp_R, temp_match = calculate_R_precision(
                t_latents_pred.cpu().numpy(),
                m_latents_pred.cpu().numpy(),
                top_k=3,
                sum_all=True,
            )
            R_precision += temp_R
            matching_score_pred += temp_match
            nb_sample += bs

        break

    motion_annotation_np = torch.cat(motion_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 2
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 2)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    print(gt_mu.shape, gt_cov.shape, mu.shape, cov.shape)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"-->  FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    real_metrics = (0.0, diversity_real, R_precision_real, matching_score_real)
    pred_metrics = (fid, diversity, R_precision, matching_score_pred)

    return real_metrics, pred_metrics
