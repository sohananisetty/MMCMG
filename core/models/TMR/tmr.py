from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .losses import InfoNCE_with_filtering
from .metrics import all_contrastive_metrics
from .temos import TEMOS


# x.T will be deprecated in pytorch
def transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))


def get_sim_matrix(x, y):
    x_logits = torch.nn.functional.normalize(x, dim=-1)
    y_logits = torch.nn.functional.normalize(y, dim=-1)
    sim_matrix = x_logits @ transpose(y_logits)
    return sim_matrix


# Scores are between 0 and 1
def get_score_matrix(x, y):
    sim_matrix = get_sim_matrix(x, y)
    scores = sim_matrix / 2 + 0.5
    return scores


class TMR(TEMOS):
    r"""TMR: Text-to-Motion Retrieval
    Using Contrastive 3D Human Motion Synthesis
    Find more information about the model on the following website:
    https://mathis.petrovich.fr/tmr

    Args:
        motion_encoder: a module to encode the input motion features in the latent space (required).
        text_encoder: a module to encode the text embeddings in the latent space (required).
        motion_decoder: a module to decode the latent vector into motion features (required).
        vae: a boolean to make the model probabilistic (required).
        fact: a scaling factor for sampling the VAE (optional).
        sample_mean: sample the mean vector instead of random sampling (optional).
        lmd: dictionary of losses weights (optional).
        lr: learninig rate for the optimizer (optional).
        temperature: temperature of the softmax in the contrastive loss (optional).
        threshold_selfsim: threshold used to filter wrong negatives for the contrastive loss (optional).
        threshold_selfsim_metrics: threshold used to filter wrong negatives for the metrics (optional).
    """

    def __init__(
        self,
        motion_encoder: nn.Module,
        text_encoder: nn.Module,
        motion_decoder: nn.Module,
        vae: bool,
        fact: Optional[float] = None,
        sample_mean: Optional[bool] = False,
        recons=1.0,
        latent=1.0e-5,
        kl=1.0e-5,
        contrastive=0.1,
        lr: float = 1e-4,
        temperature: float = 0.7,
        threshold_selfsim: float = 0.80,
        threshold_selfsim_metrics: float = 0.95,
    ) -> None:
        # Initialize module like TEMOS
        super().__init__(
            motion_encoder=motion_encoder,
            text_encoder=text_encoder,
            motion_decoder=motion_decoder,
            vae=vae,
            fact=fact,
            sample_mean=sample_mean,
            lmd={
                "recons": recons,
                "latent": latent,
                "kl": kl,
                "contrastive": contrastive,
            },
            lr=lr,
        )

        # adding the contrastive loss
        self.contrastive_loss_fn = InfoNCE_with_filtering(
            temperature=temperature, threshold_selfsim=threshold_selfsim
        )
        self.threshold_selfsim_metrics = threshold_selfsim_metrics

        # store validation values to compute retrieval metrics
        # on the whole validation set
        self.validation_step_t_latents = []
        self.validation_step_m_latents = []
        self.validation_step_sent_emb = []

    def mean_pooling(self, token_embeddings, attention_mask):

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )  ## b d

    def compute_loss(self, inputs: Tuple, conditions: Dict, return_all=False) -> Dict:

        text_conds = conditions["text"]

        text_x_dict = {"x": text_conds[0], "mask": text_conds[1].to(torch.bool)}
        motion_x_dict = {"x": inputs[0], "mask": inputs[1].to(torch.bool)}
        # batch["text_x_dict"]
        # motion_x_dict = batch["motion_x_dict"]

        motion_mask = motion_x_dict["mask"]
        text_mask = text_x_dict["mask"]
        ref_motions = motion_x_dict["x"]

        # sentence embeddings

        if conditions.get("sent_emb", None) is None:
            sent_emb = self.mean_pooling(text_conds[0], text_conds[1].to(torch.bool))
            if len(sent_emb.shape) > 2:
                sent_emb = sent_emb.squeeze(1)  ##b d
        else:
            sent_emb = conditions["sent_emb"][0]

        # text -> motion
        t_motions, t_latents, t_dists = self(
            text_x_dict, mask=motion_mask, return_all=True
        )

        # motion -> motion
        m_motions, m_latents, m_dists = self(
            motion_x_dict, mask=motion_mask, return_all=True
        )

        # Store all losses
        losses = {}

        # Reconstructions losses
        # fmt: off
        losses["recons"] = (
            + self.reconstruction_loss_fn(t_motions, ref_motions) # text -> motion
            + self.reconstruction_loss_fn(m_motions, ref_motions) # motion -> motion
        )
        # fmt: on

        # VAE losses
        if self.vae:
            # Create a centred normal distribution to compare with
            # logvar = 0 -> std = 1
            ref_mus = torch.zeros_like(m_dists[0])
            ref_logvar = torch.zeros_like(m_dists[1])
            ref_dists = (ref_mus, ref_logvar)

            losses["kl"] = (
                self.kl_loss_fn(t_dists, m_dists)  # text_to_motion
                + self.kl_loss_fn(m_dists, t_dists)  # motion_to_text
                + self.kl_loss_fn(m_dists, ref_dists)  # motion
                + self.kl_loss_fn(t_dists, ref_dists)  # text
            )

        # Latent manifold loss
        losses["latent"] = self.latent_loss_fn(t_latents, m_latents)

        # TMR: adding the contrastive loss
        losses["contrastive"] = self.contrastive_loss_fn(t_latents, m_latents, sent_emb)

        # Weighted average of the losses
        losses["loss"] = sum(
            self.lmd[x] * val for x, val in losses.items() if x in self.lmd
        )

        # Used for the validation step
        if return_all:
            return losses, t_latents, m_latents

        return losses

    def validation_step(self, inputs: Tuple, conditions: Dict) -> Tensor:
        bs = len(inputs[0])
        losses, t_latents, m_latents = self.compute_loss(
            inputs, conditions, return_all=True
        )

        # Store the latent vectors
        self.validation_step_t_latents.append(t_latents)
        self.validation_step_m_latents.append(m_latents)
        self.validation_step_sent_emb.append(conditions["sent_emb"][0])

        return losses

    def on_validation_epoch_end(self):
        # Compute contrastive metrics on the whole batch
        t_latents = torch.cat(self.validation_step_t_latents)
        m_latents = torch.cat(self.validation_step_m_latents)
        sent_emb = torch.cat(self.validation_step_sent_emb)

        # Compute the similarity matrix
        sim_matrix = get_sim_matrix(t_latents, m_latents).cpu().numpy()

        contrastive_metrics = all_contrastive_metrics(
            sim_matrix,
            emb=sent_emb.cpu().numpy(),
            threshold=self.threshold_selfsim_metrics,
        )

        self.validation_step_t_latents.clear()
        self.validation_step_m_latents.clear()
        self.validation_step_sent_emb.clear()

        return contrastive_metrics
