import os
import random

import torch
import torch.nn as nn
from core import VQVAEOutput
from core.models.utils import default
from core.models.VQVAE.encdec import Decoder, Encoder
from core.models.VQVAE.quantization.residual_vector_quantize_simple import \
    ResidualVQ
from core.models.VQVAE.quantization.vector_quantize_simple import \
    QuantizeEMAReset
from core.models.VQVAE.resnet import Resnet1D
from core.models.VQVAE.vqvae import HumanVQVAE


class RVQVAE(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        nb_code=1024,
        code_dim=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        num_quantizers=2,
        quantize_dropout_prob=0.2,
        activation="relu",
        shared_codebook=False,
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
        # nn.Linear(input_dim, code_dim)
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
        rvqvae_config = {
            "num_quantizers": num_quantizers,
            "shared_codebook": shared_codebook,
            "quantize_dropout_prob": quantize_dropout_prob,
            "quantize_dropout_cutoff_index": 0,
            "nb_code": nb_code,
            "code_dim": code_dim,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        # print(x_encoder.shape)
        quantized_out, code_idx, all_codes = self.quantizer.quantize(
            x_encoder, return_latent=True
        )
        # (N, T, Q)
        return quantized_out, code_idx, all_codes

    def forward(self, x, sample_codebook_temp=0.0):
        # x b n d
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)

        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(
            x_encoder, sample_codebook_temp=sample_codebook_temp
        )  ## b d n , b n q

        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)  ## b n d

        return VQVAEOutput(
            decoded_motion=x_out,
            indices=code_idx,
            commit_loss=commit_loss.sum(),
            perplexity=perplexity,
        )

        # return x_out, commit_loss, perplexity

    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0).permute(0, 2, 1)

        # decoder
        x_decoder = self.decoder(x)
        x_out = self.postprocess(x_decoder)
        return x_out


class HumanRVQVAE(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()

        self.args = args

        self.nb_joints = args.nb_joints
        self.sample_codebook_temp = args.sample_codebook_temp

        self.rvqvae = RVQVAE(
            input_dim=args.motion_dim,
            nb_code=args.codebook_size,
            code_dim=args.codebook_dim,
            down_t=args.down_sampling_ratio // 2,
            stride_t=args.down_sampling_ratio // 2,
            width=args.dim,
            depth=args.depth,
            dilation_growth_rate=3,
            num_quantizers=args.num_quantizers,
            quantize_dropout_prob=args.quantize_dropout_prob,
            activation="relu",
            shared_codebook=args.shared_codebook,
            norm=None,
        )

    def load(self, path):
        pkg = torch.load(path, map_location="cuda")
        self.load_state_dict(pkg["model"])

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, motion):
        b, t, c = motion.size()
        _, code_idx, all_codes = self.rvqvae.encode(motion)  # (N, T)
        return code_idx

    def forward(self, motion, temperature=None):

        return self.rvqvae(
            motion, sample_codebook_temp=default(temperature, self.sample_codebook_temp)
        )

    def forward_decoder(self, indices):
        # indices shape 'b n q'
        x_out = self.rvqvae.forward_decoder(indices)
        return x_out
