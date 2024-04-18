import math
import random

import numpy as np
import torch
import torch.nn as nn
import utils.rotation_conversions as geometry
from core import VQVAEOutput
from core.models.utils import default
from core.models.VQVAE.encdec import Decoder, Encoder
from core.models.VQVAE.quantization.vector_quantize import VectorQuantize
from core.models.VQVAE.quantization.vector_quantize_simple import \
    QuantizeEMAReset
from einops import pack, rearrange


class VQVAE(nn.Module):
    def __init__(
        self,
        input_dim=263,
        nb_code=1024,
        code_dim=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
        kmeans_iters=None,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.encoder = Encoder(
            input_dim,
            code_dim,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        self.decoder = Decoder(
            input_dim,
            code_dim,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        self.quantizer = QuantizeEMAReset(
            nb_code, code_dim, code_dim, kmeans_iters=kmeans_iters
        )

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x, mask=None):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        if mask is not None:

            downsampled_motion_mask = torch.nn.functional.max_pool1d(
                mask.float(),
                1,
                stride=4,
            )
            x_encoder = x_encoder * downsampled_motion_mask[:, None, :]

        # x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)

        code_idx = self.quantizer.encode(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx

    def forward(self, x, mask=None, temperature=0.0):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        if mask is not None:

            downsampled_motion_mask = torch.nn.functional.max_pool1d(
                mask.float(),
                1,
                stride=4,
            )
            x_encoder = x_encoder * downsampled_motion_mask[:, None, :]

        ## quantization
        x_quantized, code_idx, loss, perplexity = self.quantizer(
            x_encoder, return_idx=True, temperature=temperature
        )

        if mask is not None:

            x_quantized = x_quantized * downsampled_motion_mask[:, None, :]

        ## decoder
        x_decoder = self.decoder(x_quantized)

        if mask is not None:

            x_decoder = x_decoder * mask[:, None, :]

        x_out = self.postprocess(x_decoder)
        return VQVAEOutput(
            decoded_motion=x_out,
            indices=code_idx,
            commit_loss=loss.sum(),
            perplexity=perplexity,
            quantized_motion=x_quantized.clone().permute(0, 2, 1),
        )

    def forward_decoder(self, indices, mask=None):
        x_quantized = self.quantizer.dequantize(indices)
        ##need (bs, Jx3, T)
        x_quantized = x_quantized.permute(0, 2, 1).contiguous()
        if mask is not None:

            x_quantized = x_quantized * mask[:, None, :]

        # decoder
        x_decoder = self.decoder(x_quantized)
        if mask is not None:
            upsampled_motion_mask = nn.functional.interpolate(
                mask[:, None, :].to(torch.float), scale_factor=4
            ).to(torch.bool)

            x_decoder = x_decoder * upsampled_motion_mask

        x_out = self.postprocess(x_decoder)
        return x_out


class VQVAE2(nn.Module):
    def __init__(
        self,
        input_dim=263,
        nb_code=1024,
        code_dim=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.encoder = Encoder(
            input_dim,
            code_dim,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        self.decoder = Decoder(
            input_dim,
            code_dim,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        # self.quantizer = QuantizeEMAReset(nb_code, code_dim, code_dim)
        self.quantizer = VectorQuantize(
            dim=code_dim,
            codebook_dim=code_dim,
            codebook_size=nb_code,  # codebook size
            kmeans_init=True,  # set to True
            kmeans_iters=100,
            threshold_ema_dead_code=10,
            stochastic_sample_codes=True,
            sample_codebook_temp=0.2,
            affine_param=True,
            sync_update_v=0.2,
            sync_codebook=False,
            channel_last=False,
        )

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x, mask=None):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        if mask is not None:

            downsampled_motion_mask = torch.nn.functional.max_pool1d(
                mask.float(),
                1,
                stride=4,
            )
            x_encoder = x_encoder * downsampled_motion_mask[:, None, :]

        # x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx

    def forward(self, x, mask=None, temperature=None):

        if self.training is False:
            freeze_codebook = True
        else:
            freeze_codebook = False
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        if mask is not None:

            downsampled_motion_mask = torch.nn.functional.max_pool1d(
                mask.float(),
                1,
                stride=4,
            )
            x_encoder = x_encoder * downsampled_motion_mask[:, None, :]

        ## quantization
        x_quantized, code_idx, loss = self.quantizer(
            x=x_encoder,
            mask=mask,
            sample_codebook_temp=temperature,
            freeze_codebook=freeze_codebook,
        )

        if mask is not None:

            x_quantized = x_quantized * downsampled_motion_mask[:, None, :]

        ## decoder
        x_decoder = self.decoder(x_quantized)

        if mask is not None:

            x_decoder = x_decoder * mask[:, None, :]

        x_out = self.postprocess(x_decoder)
        return VQVAEOutput(
            decoded_motion=x_out,
            indices=code_idx,
            commit_loss=loss.sum(),
            quantized_motion=x_quantized.clone().permute(0, 2, 1),
        )

    def forward_decoder(self, indices, mask=None):
        x_quantized = self.quantizer.dequantize(indices)
        ##need (bs, Jx3, T)
        x_quantized = x_quantized.permute(0, 2, 1).contiguous()
        if mask is not None:

            x_quantized = x_quantized * mask[:, None, :]

        # decoder
        x_decoder = self.decoder(x_quantized)
        if mask is not None:
            upsampled_motion_mask = nn.functional.interpolate(
                mask[:, None, :].to(torch.float), scale_factor=4
            ).to(torch.bool)

            x_decoder = x_decoder * upsampled_motion_mask

        x_out = self.postprocess(x_decoder)
        return x_out


class HumanVQVAE(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()

        self.nb_joints = args.nb_joints
        self.sample_codebook_temp = args.sample_codebook_temp
        self.vqvae = VQVAE(
            input_dim=args.motion_dim,
            nb_code=args.codebook_size,
            code_dim=args.codebook_dim,
            down_t=args.down_sampling_ratio // 2,
            stride_t=args.down_sampling_ratio // 2,
            width=args.dim,
            depth=args.depth,
            dilation_growth_rate=3,
            activation="relu",
            norm=None,
            kmeans_iters=args.kmeans_iters,
        )

    def load(self, path):
        pkg = torch.load(path, map_location="cuda")
        self.load_state_dict(pkg["model"])

    def freeze(self):
        # self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, motion):
        b, t, c = motion.size()
        code_idx = self.vqvae.encode(motion)  # (N, T)
        return code_idx

    def forward(self, motion, mask=None, sample_temperature=None):
        return self.vqvae(
            motion,
            mask,
            temperature=default(sample_temperature, self.sample_codebook_temp),
        )

    def decode(self, indices, mask=None):
        x_out = self.vqvae.forward_decoder(indices, mask)
        return x_out


class HumanVQVAE2(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()

        self.nb_joints = args.nb_joints
        self.sample_codebook_temp = args.sample_codebook_temp
        self.vqvae = VQVAE2(
            input_dim=args.motion_dim,
            nb_code=args.codebook_size,
            code_dim=args.codebook_dim,
            down_t=args.down_sampling_ratio // 2,
            stride_t=args.down_sampling_ratio // 2,
            width=args.dim,
            depth=args.depth,
            dilation_growth_rate=3,
            activation="relu",
            norm=None,
        )

    def load(self, path):
        pkg = torch.load(str(path), map_location="cuda")
        self.vqvae.quantizer._codebook.batch_mean = pkg["model"][
            "vqvae.quantizer._codebook.batch_mean"
        ]
        self.vqvae.quantizer._codebook.batch_variance = pkg["model"][
            "vqvae.quantizer._codebook.batch_variance"
        ]
        self.load_state_dict(pkg["model"])

        print("loaded model with ", pkg["total_loss"].item(), pkg["steps"], "steps")

    def freeze(self):
        # self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, motion):
        b, t, c = motion.size()
        code_idx = self.vqvae.encode(motion)  # (N, T)
        return code_idx

    def forward(self, motion, mask=None, temperature=0.0):
        return self.vqvae(
            motion, mask, temperature=default(temperature, self.sample_codebook_temp)
        )

    def decode(self, indices, mask=None):
        ##indices: b n
        x_out = self.vqvae.forward_decoder(indices, mask)
        return x_out


from scipy.ndimage import gaussian_filter1d


class TranslationVQVAE(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()

        self.vqvae = VQVAE2(
            input_dim=args.motion_dim,
            nb_code=args.codebook_size,
            code_dim=args.codebook_dim,
            down_t=args.down_sampling_ratio // 2,
            stride_t=args.down_sampling_ratio // 2,
            width=args.dim,
            depth=args.depth,
            dilation_growth_rate=3,
            activation="relu",
            norm=None,
            # kmeans_iters=args.kmeans_iters,
        )

    # def load(self, path):
    #     pkg = torch.load(path, map_location="cuda")
    #     self.load_state_dict(pkg["model"])

    def load(self, path):
        pkg = torch.load(str(path), map_location="cuda")
        self.vqvae.quantizer._codebook.batch_mean = pkg["model"][
            "vqvae.quantizer._codebook.batch_mean"
        ]
        self.vqvae.quantizer._codebook.batch_variance = pkg["model"][
            "vqvae.quantizer._codebook.batch_variance"
        ]
        self.load_state_dict(pkg["model"])

        print("loaded model with ", pkg["total_loss"].item(), pkg["steps"], "steps")

    def freeze(self):
        # self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, motion):
        b, t, c = motion.size()
        code_idx = self.vqvae.encode(motion)  # (N, T)
        return code_idx

    def cosine_schedule(self, t):
        return torch.cos(t * math.pi * 0.5)

    def mask_augment(self, motion, perc_n=0.1):
        b, n, d = motion.shape
        device = motion.device
        rand_time = torch.zeros((b,), device=device).float().uniform_(1 - perc_n, 1)
        rand_mask_probs = torch.cos(rand_time * math.pi * 0.5)
        num_masked_n = (n * rand_mask_probs).round().clamp(min=1)
        num_masked_d = torch.Tensor(np.random.choice([0, 1, 2], b)).to(device)

        to_mask = random.random() > 0.4
        if to_mask:

            batch_randperm1 = torch.rand((b, n), device=device).argsort(dim=-1)
            batch_randperm2 = torch.rand((b, d), device=device).argsort(dim=-1)

            mask1 = ~(batch_randperm1 < rearrange(num_masked_n, "b -> b 1"))
            mask2 = ~(batch_randperm2 < rearrange(num_masked_d, "b -> b 1"))

            # motion = motion * mask1[:, :, None]
            motion = motion * mask2[:, None, :]

        return motion

    def cossin2quat(self, r_cs: torch.Tensor):
        mat = torch.zeros(
            (
                r_cs.shape[:-1]
                + (
                    3,
                    3,
                )
            )
        ).to(r_cs.device)
        mat[..., 1, 1] = 1
        mat[..., 0, 0] = r_cs[..., 0]
        mat[..., 0, 2] = r_cs[..., 1]
        mat[..., 2, 0] = -r_cs[..., 1]
        mat[..., 2, 2] = r_cs[..., 0]
        r_rot_quat = geometry.matrix_to_quaternion(mat)
        return r_rot_quat

    @torch.no_grad()
    def predict(self, rel_pos):
        ## traj xy rel positions transformed
        # self.vqvae = self.vqvae.eval()

        orient = torch.zeros_like(rel_pos)[..., :2]
        motion = torch.cat([rel_pos, orient], -1)

        # print(motion.shape)

        out = self.vqvae(motion)
        pred_orient = out.decoded_motion[..., -2:].clamp(min=-1, max=1)
        pred_orient[:, 0, 0] = 1
        pred_orient[:, 0, 1] = 0
        pred_orient[..., 1:2] = torch.Tensor(
            gaussian_filter1d(
                pred_orient[..., 1:2].cpu().numpy(), 4, axis=-1, mode="nearest"
            )
        )
        pred_quat = self.cossin2quat(pred_orient)
        return pred_quat

    def forward(self, motion, mask=None):
        r_rot = motion[..., :4]
        rel_pos = motion[..., 4:]
        if rel_pos.shape[-1] == 3:
            rel_pos = rel_pos[..., [0, 2]]
        r_rot = geometry.quaternion_to_matrix(r_rot)[..., [0, 0], [0, 2]]

        motion = torch.cat([rel_pos, r_rot], -1)

        if self.training:
            # motion = self.mask_augment(motion)
            b, n, d = motion.shape
            device = motion.device
            num_masked_d = torch.Tensor(
                np.random.choice([0, 1, 2], size=b, p=[0.4, 0.3, 0.3])
            ).to(device)
            batch_randperm2 = torch.rand((b, 2), device=device).argsort(dim=-1)
            mask2 = ~(batch_randperm2 < rearrange(num_masked_d, "b -> b 1"))
            aug_motion = motion.clone()
            aug_motion[..., 2:] = aug_motion[..., 2:] * mask2[:, None, :]
            return self.vqvae(aug_motion, mask)

        return self.vqvae(motion, mask)

    def decode(self, indices, mask=None):
        x_out = self.vqvae.forward_decoder(indices, mask)
        return x_out
