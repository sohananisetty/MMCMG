import math
import typing as tp
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from random import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from core.datasets.conditioner import ClassifierFreeGuidanceDropout, ConditionFuser
from core.models.attend import CustomMHA
from core.models.positional_embeddings import (
    AlibiPositionalBias,
    ScaledSinusoidalEmbedding,
)
from core.models.utils import FeedForward, LayerNorm, default, exists, get_obj_from_str
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
from tqdm.auto import tqdm
from core import MotionTokenizerParams


ConditionType = Tuple[torch.Tensor, torch.Tensor]
ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]


@dataclass
class MuseOutput:
    loss: torch.Tensor = None
    logits: torch.Tensor = None
    embed: torch.Tensor = None
    ce_per_codebook: List[torch.Tensor] = None


# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def l2norm(t):
    return F.normalize(t, dim=-1)


def make_copy(condition):
    copy_conditions = {}
    for condition_modality, (embedding, mask) in condition.items():
        copy_conditions[condition_modality] = (embedding.clone(), mask.clone())

    return copy_conditions


def get_mask_subset_prob(mask, prob, min_mask=0):
    batch, K, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim=-1, keepdim=True) * prob).clamp(min=min_mask)
    logits = torch.rand((batch, K, seq), device=device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim=-1).argsort(dim=-1).float()

    num_padding = (~mask).sum(dim=-1, keepdim=True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask


# classes
class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1) * (dim**-0.5))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x / norm.clamp(min=self.eps) * self.g


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g


def FeedForward(
    dim,
    mult=4,
    dropout=0.1,
):
    """https://arxiv.org/abs/2110.09456"""

    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        LayerNorm(inner_dim),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


class FiLM(nn.Module):
    def __init__(self, dim, dim_cond):
        super().__init__()
        self.to_cond = nn.Linear(dim_cond, dim * 2, bias=False)

    def forward(self, x, cond):
        gamma, beta = self.to_cond(cond).chunk(2, dim=-1)
        return x * gamma + beta


class ScaledEmbedding(nn.Embedding):
    """Boost learning rate for embeddings (with `scale`)."""

    def __init__(self, *args, lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def make_optim_group(self):
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        return group


class TransformerBlockMuseCustom(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        heads: int = 8,
        attn_dropout: float = 0.0,
        flash: bool = False,
        depth: int = 1,
        ff_mult: int = 4,
        film_skip: int = 1,
        causal: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.causal = causal

        for d in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        CustomMHA(
                            dim=dim,
                            heads=heads,
                            dropout=attn_dropout,
                            add_null_kv=False,
                            causal=causal,
                            flash=flash,
                        ),
                        CustomMHA(
                            dim=dim,
                            heads=heads,
                            dropout=attn_dropout,
                            add_null_kv=True,
                            flash=flash,
                            causal=False,
                        ),
                        (
                            FiLM(dim, dim)
                            if (
                                film_skip != 0 and d % film_skip == 0 and d != depth - 1
                            )
                            else nn.Identity()
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=attn_dropout),
                    ]
                )
            )

        self.norm_sa = LayerNorm(dim)
        self.norm_cross = LayerNorm(dim)
        self.norm_film = LayerNorm(dim)
        self.norm_out = LayerNorm(dim)
        self.norm_film_context = LayerNorm(dim)

    def forward(
        self,
        x,
        mask=None,
        context=None,
        context_mask=None,
        film_context=None,
        rel_pos=None,
    ):
        for attn, cross_attn, film, ff1 in self.layers:

            x = self.norm_sa(x)
            x = (
                attn(
                    q=x,
                    k=x,
                    v=x,
                    key_padding_mask=mask,
                    rel_pos=rel_pos if self.causal else None,
                )
                + x
            )

            x = (
                cross_attn(
                    q=self.norm_cross(x),
                    k=context,
                    v=context,
                    key_padding_mask=context_mask,
                )
                + x
            )

            if not isinstance(film, nn.Identity):

                x = (
                    film(
                        self.norm_film(x),
                        cond=self.norm_film_context(film_context),
                    )
                    + x
                )

            x = self.norm_out(ff1(x) + x)

        return x


class TransformerBlockMuseSpatialCustom(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        heads: int = 8,
        attn_dropout: float = 0.0,
        flash: bool = False,
        depth: int = 1,
        ff_mult: int = 4,
        causal: bool = False,
        film_skip: int = 0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.causal = causal

        for d in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        CustomMHA(
                            dim=dim,
                            heads=heads,
                            dropout=attn_dropout,
                            add_null_kv=False,
                            causal=causal,
                            flash=flash,
                        ),
                        CustomMHA(
                            dim=dim,
                            heads=heads,
                            dropout=attn_dropout,
                            add_null_kv=False,
                            causal=causal,
                            flash=flash,
                        ),
                        CustomMHA(
                            dim=dim,
                            heads=heads,
                            dropout=attn_dropout,
                            add_null_kv=True,
                            flash=flash,
                            causal=False,
                        ),
                        (
                            FiLM(dim, dim)
                            if (
                                film_skip != 0 and d % film_skip == 0 and d != depth - 1
                            )
                            else nn.Identity()
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=attn_dropout),
                    ]
                )
            )

        self.norm_spatial_attn = LayerNorm(dim)
        self.norm_temporal_attn = LayerNorm(dim)
        self.norm_cross = LayerNorm(dim)
        self.norm_out = LayerNorm(dim)
        self.norm_film = LayerNorm(dim)
        self.norm_film_context = LayerNorm(dim)

    def forward(
        self,
        x,
        mask=None,
        context=None,
        context_mask=None,
        film_context=None,
        rel_pos=None,
    ):

        b, n, c, d = x.shape
        x = rearrange(x, "b n c d -> (b c) n d", b=b, n=n, c=c, d=d)

        mask = None if mask is None else repeat(mask, "b n -> (b c) n", c=c)
        film_context = (
            None
            if film_context is None
            else repeat(film_context, "b n d -> (b c) n d", c=c)
        )
        context = (
            None if context is None else repeat(context, "b n d -> (b c) n d", c=c)
        )
        context_mask = (
            None
            if context_mask is None
            else repeat(context_mask, "b n -> (b c) n", c=c)
        )

        for sp_attn, tp_attn, cross_attn, film, ff1 in self.layers:

            if not isinstance(sp_attn, nn.Identity):
                x = rearrange(x, "(b c) n d -> (b n) c d", b=b, n=n, c=c, d=d)

                x = self.norm_spatial_attn(x)
                x = (
                    sp_attn(
                        q=x,
                        k=x,
                        v=x,
                        key_padding_mask=mask,
                        rel_pos=rel_pos if self.causal else None,
                    )
                    + x
                )
                x = rearrange(x, "(b n) c d -> (b c) n d", b=b, n=n, c=c, d=d)

            x = self.norm_temporal_attn(x)
            x = (
                tp_attn(
                    q=x,
                    k=x,
                    v=x,
                    key_padding_mask=mask,
                    rel_pos=rel_pos if self.causal else None,
                )
                + x
            )

            x = (
                cross_attn(
                    q=self.norm_cross(x),
                    k=(context),
                    v=(context),
                    key_padding_mask=context_mask,
                )
                + x
            )

            if not isinstance(film, nn.Identity) and film_context is not None:

                x = (
                    film(
                        self.norm_film(x),
                        cond=self.norm_film_context(film_context),
                    )
                    + x
                )

            x = self.norm_out(ff1(x) + x)

        x = rearrange(x, "(b c) n d -> b c n d", b=b, n=n, c=c, d=d)

        return x


class TransformerBlockMuse(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        heads: int = 8,
        attn_dropout: float = 0.0,
        flash: bool = False,
        depth: int = 1,
        ff_mult: int = 4,
        causal: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.MultiheadAttention(
                            embed_dim=dim,
                            num_heads=heads,
                            dropout=attn_dropout,
                            bias=False,
                            batch_first=True,
                        ),
                        nn.MultiheadAttention(
                            embed_dim=dim,
                            num_heads=heads,
                            dropout=attn_dropout,
                            bias=False,
                            batch_first=True,
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=attn_dropout),
                    ]
                )
            )

        self.norm_sa = nn.LayerNorm(dim, eps=1e-5, bias=False)
        self.norm_cross = nn.LayerNorm(dim, eps=1e-5, bias=False)
        self.norm_out = nn.LayerNorm(dim, eps=1e-5, bias=False)

    def forward(self, x, mask=None, context=None, context_mask=None, rel_pos=None):
        for attn, cross_attn, ff1 in self.layers:

            x = self.norm_sa(x)

            x = (
                attn(
                    query=x,
                    key=x,
                    value=x,
                    key_padding_mask=mask,
                    need_weights=False,
                )[0]
                + x
            )

            x = (
                cross_attn(
                    query=self.norm_cross(x),
                    key=context,
                    value=context,
                    key_padding_mask=context_mask,
                    need_weights=False,
                )[0]
                + x
            )

            x = self.norm_out(ff1(x) + x)

        return x


@dataclass
class LMOutput:
    # The logits are already re-aligned with the input codes
    # hence no extra shift is required, e.g. when computing CE
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor  # [B, K, T]


class MLMModel(nn.Module):
    def __init__(
        self,
        fuser: ConditionFuser,
        num_tokens,
        dim,
        heads,
        n_q=3,
        rel_pos=False,
        var_len=True,
        custom=True,
        spatial=False,
        film=False,
        film_skip=1,
        flatten=False,
        quality_emb=False,
        emb_lr: tp.Optional[float] = None,
        bias_proj=False,
        emb_dropout=0.0,
        cond_dropout=0.0,
        post_emb_norm: bool = False,
        audio_input_dim: int = 128,
        text_input_dim: int = 768,
        proj_input=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.n_q = n_q
        self.audio_input_dim = audio_input_dim
        self.text_input_dim = text_input_dim
        self.cond_dropout = cond_dropout
        self.var_len = var_len
        self.quality_emb = quality_emb
        self.spatial = spatial
        self.flatten = flatten
        self.film_skip = film_skip
        self.film = film_skip > 0

        ## if flatten num_tokens = sum of all num_tokens

        self.set_token_ids(num_tokens, False)

        self.token_emb = nn.ModuleList(
            [ScaledEmbedding(self.vocab_size, self.dim, lr=emb_lr) for _ in range(n_q)]
        )

        if quality_emb:

            self.qual_emb = nn.Embedding(2, self.dim)

        self.pos_emb = ScaledSinusoidalEmbedding(dim=self.dim)
        self.rel_pos_bias = None
        if rel_pos:

            self.rel_pos_bias = AlibiPositionalBias(heads=heads // 2, total_heads=heads)

        if custom:
            if self.spatial:
                self.transformer_blocks = TransformerBlockMuseSpatialCustom(
                    dim=dim, heads=heads, film_skip=film_skip, **kwargs
                )
            else:

                self.transformer_blocks = TransformerBlockMuseCustom(
                    dim=dim, heads=heads, film_skip=film_skip, **kwargs
                )
        else:
            self.transformer_blocks = TransformerBlockMuse(
                dim=dim, heads=heads, **kwargs
            )
        self.norm = LayerNorm(dim)
        self.post_emb_norm = LayerNorm(dim) if post_emb_norm else nn.Identity()

        self.condition_fuser = fuser
        self.translation_present = (
            self.condition_fuser.cond2fuse.get("translation", None) is not None
        )

        self.cfg_dropout = ClassifierFreeGuidanceDropout(p=self.cond_dropout)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_input = (
            nn.Linear(n_q * self.dim, self.dim, bias=False)
            if proj_input
            else nn.Identity()
        )

        self.project_audio = (
            nn.Linear(self.audio_input_dim, self.dim, bias=False)
            if self.audio_input_dim != self.dim
            else nn.Identity()
        )
        self.downsample_audio = nn.Sequential(
            Rearrange("b n d -> b d n"),
            nn.Conv1d(
                self.audio_input_dim,
                self.audio_input_dim,
                3,
                2,
                1,
                dilation=1,
                bias=False,
            ),
            nn.Conv1d(self.audio_input_dim, self.audio_input_dim, 3, 2, 1, bias=False),
            Rearrange("b d n -> b n d"),
        )  ## downsample factor 4 hard coded. assert audio sampling rate is equal to fps

        self.project_text = (
            nn.Linear(self.text_input_dim, self.dim, bias=False)
            if self.text_input_dim != self.dim
            else nn.Identity()
        )

        self.linears = (
            nn.Linear(dim, self.num_tokens, bias=bias_proj)
            if self.spatial
            else nn.ModuleList(
                [nn.Linear(dim, self.num_tokens, bias=bias_proj) for _ in range(n_q)]
            )
        )
        self.out_norm = LayerNorm(dim)

    def set_token_ids(self, num_tokens, add_special_token=False):
        motion_tokenizer_params = MotionTokenizerParams(num_tokens, add_special_token)
        self.num_tokens = motion_tokenizer_params.num_tokens
        self.mask_token_id = motion_tokenizer_params.mask_token_id
        self.pad_token_id = motion_tokenizer_params.pad_token_id
        self.special_token_id = motion_tokenizer_params.special_token_id
        self.vocab_size = motion_tokenizer_params.vocab_size

    @property
    def num_codebooks(self) -> int:
        return self.n_q

    def _compute_cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None,
        ignore_index=-100,
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        """Compute cross entropy between multi-codebook targets and model's logits.
        The cross entropy is computed per codebook to provide codebook-level cross entropy.
        Valid timesteps for each of the codebook are pulled from the mask, where invalid
        timesteps are set to 0.

        Args:
            logits (torch.Tensor): Model's logits of shape [B, K, T, card].
            targets (torch.Tensor): Target codes, of shape [B, K, T].
            mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
        Returns:
            ce (torch.Tensor): Cross entropy averaged over the codebooks
            ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
        """
        B, K, T = targets.shape
        assert logits.shape[:-1] == targets.shape
        # assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook: tp.List[torch.Tensor] = []
        for k in range(K):
            logits_k = (
                logits[:, k, ...].contiguous().view(-1, logits.size(-1))
            )  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            if mask is not None:
                mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
                targets_k = targets_k[mask_k]
                logits_k = logits_k[mask_k]
            q_ce = F.cross_entropy(logits_k, targets_k, ignore_index=ignore_index)
            ce += q_ce
            ce_per_codebook.append(q_ce)
        # average cross entropy across codebooks
        ce = ce / K
        return ce, ce_per_codebook

    def _prepare_inputs(self, input, conditions):
        new_conditions = {}

        if conditions.get("audio", None) is not None:
            audio_embed = self.downsample_audio(conditions["audio"][0])
            audio_embed = self.project_audio(audio_embed)
            new_conditions["audio"] = (audio_embed, conditions["audio"][1])

        if conditions.get("text", None) is not None:
            text_embed = self.project_text(conditions["text"][0])
            new_conditions["text"] = (text_embed, conditions["text"][1])

        rem_keys = [x for x in list(conditions.keys()) if x not in ["audio", "text"]]

        for k in rem_keys:
            new_conditions[k] = conditions[k]

        inputs_, cross_inputs = self.condition_fuser(input, new_conditions)

        if self.film:
            film_cond_mask = (
                F.interpolate(
                    conditions["audio"][1].unsqueeze(1).to(torch.float),
                    size=input[0].shape[1],
                )
                .squeeze(1)
                .to(torch.bool)
            )

            film_cond = audio_embed * film_cond_mask[..., None]

            return inputs_, cross_inputs, film_cond

        return inputs_, cross_inputs

    def forward_with_cond_scale(
        self,
        inputs: Dict[str, ConditionType],
        conditions: Dict[str, ConditionType],
        cond_scale=3.0,
        return_embed=False,
        **kwargs,
    ):
        if cond_scale == 1:
            out = self.forward(
                inputs,
                conditions,
                cond_drop_prob=0.0,
                **kwargs,
            )
            if return_embed:
                return out.logits, pos_out.embed

            return out.logits

        pos_out = self.forward(inputs, conditions, cond_drop_prob=0.0, **kwargs)

        null_out = self.forward(inputs, conditions, cond_drop_prob=1.0, **kwargs)

        scaled_logits = (
            null_out.logits + (pos_out.logits - null_out.logits) * cond_scale
        )

        if return_embed:
            return scaled_logits, pos_out.embed

        return scaled_logits

    def forward_with_neg_prompt(
        self,
        inputs: Dict[str, ConditionType],
        conditions: Dict[str, ConditionType],
        neg_conditions: Dict[str, ConditionType],
        cond_scale=3.0,
        return_embed=False,
        **kwargs,
    ):
        neg_out = self.forward(
            inputs,
            neg_conditions,
            cond_drop_prob=0.0,
            **kwargs,
        )
        pos_out = self.forward(
            inputs,
            conditions,
            cond_drop_prob=0.0,
            **kwargs,
        )

        scaled_logits = neg_out.logits + (pos_out.logits - neg_out.logits) * cond_scale

        if return_embed:
            return scaled_logits, pos_out.embed

        return scaled_logits

    def forward(
        self,
        inputs: ConditionTensors,
        conditions: ConditionTensors,
        labels: Optional[torch.IntTensor] = None,
        cond_drop_prob: float = None,
        ignore_index: int = -100,
        quality_list=None,
        return_embed=False,
    ):

        sequence = inputs[0]
        sequence_mask = inputs[1]
        B, K, S = sequence.shape

        if self.quality_emb is True and quality_list is None:
            quality_list = torch.ones(B, dtype=torch.long, device=sequence.device)

        assert (
            K == self.num_codebooks
        ), "Sequence shape must match the specified number of codebooks"

        if self.spatial:
            input_ = torch.stack(
                [self.token_emb[k](sequence[:, k]) for k in range(K)], -1
            )

            input_ = rearrange(input_, "b n d c -> b n c d")

        else:

            if isinstance(self.project_input, nn.Identity):

                input_ = sum([self.token_emb[k](sequence[:, k]) for k in range(K)])
            else:

                input_ = torch.stack(
                    [self.token_emb[k](sequence[:, k]) for k in range(K)], -1
                )
                input_ = input_.reshape(B, S, -1)

                input_ = self.project_input(input_)

        # classifier free guidance
        cond_drop_prob = default(cond_drop_prob, self.cond_dropout)
        if cond_drop_prob > 0.0:
            conditions = self.cfg_dropout(conditions, cond_drop_prob)
        # fuse conditions

        processed_inputs = self._prepare_inputs((input_, sequence_mask), conditions)

        if self.film:
            input_, cross_attention_input, film_input_ = processed_inputs
        else:
            input_, cross_attention_input = processed_inputs

        x_ = input_[0]
        x_padding_mask = input_[1]
        context = cross_attention_input[0]
        context_padding_mask = cross_attention_input[1]

        if self.spatial:
            # x_ = rearrange(x_, "b n (c d) -> b n c d", c=K)
            pos_emb_spatial = self.pos_emb(pos=torch.arange(K, device=x_.device))
            pos_emb_tmp = self.pos_emb(x_)
            x_ = x_ + pos_emb_spatial[None, None, :, :]
            x_ = x_ + pos_emb_tmp[None, :, None, :]
        else:
            x_ = x_ + self.pos_emb(x_)

        x_ = self.post_emb_norm(x_)
        x_ = self.emb_dropout(x_)

        if quality_list is not None and self.quality_emb == True:
            x_ = torch.cat([self.qual_emb(quality_list).unsqueeze(1), x_], dim=-2)
            x_padding_mask = F.pad(x_padding_mask, (1, 0), value=True)

        if self.film:
            embed = self.transformer_blocks(
                x=x_,
                mask=x_padding_mask if self.var_len else None,
                context=context,
                context_mask=context_padding_mask,
                film_context=film_input_,
                rel_pos=self.rel_pos_bias,
            )

        else:

            embed = self.transformer_blocks(
                x=x_,
                mask=x_padding_mask if self.var_len else None,
                context=context,
                context_mask=context_padding_mask,
                rel_pos=self.rel_pos_bias,
            )
        if self.out_norm:
            embed = self.out_norm(embed)

        if return_embed:
            return MuseOutput(embed=embed)

        if self.spatial:
            logits = self.linears(embed)

        else:

            logits = torch.stack([self.linears[k](embed) for k in range(K)], dim=1)

        if (
            len(self.condition_fuser.fuse2cond.get("prepend", [])) > 0
            or self.quality_emb
        ):
            logits = logits[:, :, -S:]

        if not exists(labels):
            return MuseOutput(logits=logits, embed=embed)  ## B K N num_tokens

        if self.num_codebooks == 1:

            loss = F.cross_entropy(
                rearrange(logits.squeeze(1), "b n c -> b c n"),
                labels.squeeze(1),
                ignore_index=ignore_index,
            )
            return MuseOutput(loss=loss, logits=logits, embed=embed)

        else:
            loss, ce_per_codebook = self._compute_cross_entropy(
                logits,
                labels,
                ignore_index=ignore_index,
            )
            return MuseOutput(
                loss=loss, logits=logits, embed=embed, ce_per_codebook=ce_per_codebook
            )


# sampling helpers


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(-1, ind, val)
    return probs


def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


# noise schedules


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


# main maskgit classes


class SelfCritic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.spatial = self.net.spatial
        self.num_codebooks = self.net.num_codebooks
        self.to_preds = (
            nn.Linear(net.dim, 1)
            if self.spatial
            else nn.ModuleList(
                [nn.Linear(net.dim, 1) for _ in range(self.num_codebooks)]
            )
        )
        # self.to_preds = nn.ModuleList(
        #     [nn.Linear(net.dim, 1) for _ in range(self.num_codebooks)]
        # )

    def forward_with_cond_scale(
        self,
        inputs: Dict[str, ConditionType],
        conditions: Dict[str, ConditionType],
        cond_scale=3.0,
        **kwargs,
    ):

        logits = self.forward(
            inputs, conditions=conditions, cond_drop_prob=0.0, **kwargs
        )

        if cond_scale == 1:
            return logits

        null_logits = self.forward(
            inputs, conditions=conditions, cond_drop_prob=1.0, **kwargs
        )

        return null_logits + (logits - null_logits) * cond_scale

    def forward_with_neg_prompt(
        self,
        inputs: Dict[str, ConditionType],
        conditions: Dict[str, ConditionType],
        neg_conditions: Dict[str, ConditionType],
        cond_scale=3.0,
        **kwargs,
    ):

        logits = self.forward(
            inputs, conditions=conditions, cond_drop_prob=0.0, **kwargs
        )

        if cond_scale == 1:
            return logits

        neg_logits = self.forward(
            inputs, conditions=neg_conditions, cond_drop_prob=0.0, **kwargs
        )

        return neg_logits + (logits - neg_logits) * cond_scale

    def _compute_cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None,
        ignore_index=-100,
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:

        B, K, T = targets.shape
        assert logits.shape == targets.shape
        # assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook: tp.List[torch.Tensor] = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1)  # [B x T]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            if mask is not None:
                mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
                targets_k = targets_k[mask_k]
                logits_k = logits_k[mask_k]
            q_ce = F.binary_cross_entropy_with_logits(logits_k, targets_k)
            ce += q_ce
            ce_per_codebook.append(q_ce)
        # average cross entropy across codebooks
        ce = ce / K
        return ce, ce_per_codebook

    def forward(
        self,
        inputs: ConditionTensors,
        conditions: ConditionTensors,
        labels: Optional[torch.IntTensor] = None,
        **kwargs,
    ):
        out = self.net(inputs, conditions, return_embed=True, **kwargs)
        if self.spatial:
            logits = self.to_preds(out.embed)

        else:

            logits = torch.stack(
                [self.to_preds[k](out.embed) for k in range(self.num_codebooks)], dim=1
            )

        if not exists(labels):
            return logits

        logits = rearrange(logits, "... 1 -> ...")

        if self.num_codebooks == 1:

            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(1),
                labels.squeeze(1),
            )

        else:
            loss, ce_per_codebook = self._compute_cross_entropy(
                logits,
                labels,
            )
            return loss, ce_per_codebook
        return loss, None


class MotionMuse(nn.Module):
    def __init__(
        self,
        tranformer_config,
        fuse_config,
        noise_schedule: Callable = cosine_schedule,
    ):
        super().__init__()
        self.token_critic = None
        # percentage of tokens to be [mask]ed to remain the same token, so that transformer produces better embeddings across all tokens as done in original BERT paper
        self.no_mask_token_prob = tranformer_config.pop("no_mask_token_prob")
        self.critic_loss_weight = tranformer_config.pop("critic_loss_weight")
        self.flatten = tranformer_config.flatten
        self.spatial = tranformer_config.spatial
        self.flatten_interleave = True
        self.self_token_critic = True if self.critic_loss_weight > 0 else False

        fuse_method = fuse_config.pop("fuse_method")
        if isinstance(fuse_method, list):
            fuse_method = fuse_method[0]
        condition_fuser = ConditionFuser(fuse_method, **fuse_config)

        if (
            len(condition_fuser.fuse2cond["cross"]) > 0
            and tranformer_config.cond_dropout > 0
        ):
            assert (
                tranformer_config.custom == True
            ), "when audio is cross attention, you need custom attention"

        self.model = MLMModel(
            fuser=condition_fuser,
            **tranformer_config,
        )

        if self.self_token_critic:
            self.token_critic = SelfCritic(self.model)

        self.mask_token_id = self.model.mask_token_id
        self.noise_schedule = noise_schedule
        self.num_codeboks = self.model.num_codebooks

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        state_dict = torch.load(str(path))
        self.load_state_dict(state_dict["model"])

    def muse_mask(self, motion_ids: torch.Tensor, ignore_index: int = -100):
        batch, K, T = motion_ids.shape
        motion_ids = motion_ids.contiguous().view(batch, 1, -1)
        seq_len = K * T
        device = motion_ids.device

        code_ids = motion_ids.clone()

        rand_time = uniform((batch,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)

        num_token_masked = (seq_len * 1 * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((batch, 1 * seq_len), device=device).argsort(dim=-1)
        mask = batch_randperm < rearrange(num_token_masked, "b -> b 1")
        mask = mask.reshape(batch, 1, seq_len)
        mask[code_ids == self.model.pad_token_id] = False

        labels = torch.where(mask, code_ids, ignore_index)

        if self.no_mask_token_prob > 0.0:
            no_mask_mask = get_mask_subset_prob(mask, self.no_mask_token_prob)
            mask &= ~no_mask_mask

        x = torch.where(mask, self.mask_token_id, code_ids)
        x = x.view(batch, K, T)
        labels = labels.reshape(batch, K, T)

        return x, labels, mask.reshape(batch, K, T)

    def bert_muse_mask(self, motion_ids: torch.Tensor, ignore_index: int = -100):
        batch, K, T = motion_ids.shape
        motion_ids = motion_ids.contiguous().view(batch, 1, -1)
        seq_len = K * T
        device = motion_ids.device

        code_ids = motion_ids.clone()

        rand_time = uniform((batch,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * 1 * rand_mask_probs).round().clamp(min=1)

        # mask_id = self.mask_token_id
        batch_randperm = torch.rand((batch, 1 * seq_len), device=device).argsort(dim=-1)
        mask = batch_randperm < rearrange(num_token_masked, "b -> b 1")
        mask = mask.reshape(batch, 1, seq_len)
        mask[code_ids == self.model.pad_token_id] = False

        # mask_id = self.transformer.mask_token_id
        labels = torch.where(mask, code_ids, ignore_index)
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(code_ids.shape, 0.8)).bool().to(device) & mask
        )
        code_ids[indices_replaced] = self.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(code_ids.shape, 0.5)).bool().to(device)
            & mask
            & ~indices_replaced
        )
        random_words = torch.randint(
            self.model.num_tokens, code_ids.shape, dtype=torch.long
        ).to(device)
        code_ids[indices_random] = random_words[indices_random]

        code_ids = code_ids.view(batch, K, T)
        labels = labels.reshape(batch, K, T)

        return code_ids, labels, mask.reshape(batch, K, T)

    def get_null_context(self, B, device, dtype=torch.float):
        cond_list = list(self.model.condition_fuser.cond2fuse.keys())

        conditions = {}
        for cond_type in cond_list:
            cond = torch.zeros(
                (B, 1, self.model.audio_input_dim),
                device=device,
                dtype=dtype,
            )
            cond_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
            conditions[cond_type] = (cond, cond_mask)

        return conditions

    def flatten_input(self, inputs, conditions=None, interleave=False):
        motion, mask = inputs
        b, q, n = motion.shape
        offset = (
            torch.LongTensor([i * self.model.num_tokens // q for i in range(q)])[
                None, :, None
            ]
            .repeat(b, 1, n)
            .to(motion.device)
        )

        motion = motion + offset

        if interleave:
            flat_mask = mask.repeat_interleave(q, dim=1)
            flat_motion = torch.zeros_like(motion).reshape(b, -1)
            inds = [np.arange(i, n * q, q) for i in range(q)]

            for i, ind in enumerate(inds):
                flat_motion[..., ind] = motion[:, i]

        else:
            flat_mask = mask.repeat(1, q)
            flat_motion = motion.reshape(b, -1)

        if conditions is not None:
            cond = self.flatten_cond(conditions["audio"], q, interleave)
            conditions["audio"] = cond
            return (flat_motion[:, None, :], flat_mask), conditions

        return (flat_motion[:, None, :], flat_mask)

    def flatten_cond(self, inputs, q=2, interleave=False):
        cond, mask = inputs
        b, n, d = cond.shape
        cond = cond.permute(0, 2, 1)  ##b d n
        cond = cond[:, :, None].repeat(1, 1, q, 1)
        if interleave:

            flat_mask = mask.repeat_interleave(q, dim=1)
            flat_cond = torch.zeros_like(cond).reshape(b, d, -1)
            inds = [np.arange(i, n * q, q) for i in range(q)]

            for i, ind in enumerate(inds):
                flat_cond[..., ind] = cond[..., i, :]

        else:
            flat_mask = mask.repeat(1, q)
            flat_cond = cond.reshape(b, d, -1)

        flat_cond = flat_cond.permute(0, 2, 1)  ##b nq d

        return (flat_cond, flat_mask)

    def unflatten(self, ids, q=2, interleave=True):
        b, _, n = ids.shape
        l = n // q
        device = ids.device
        unflat = torch.zeros((b, q, l), dtype=torch.long, device=device)
        if interleave:
            inds = [np.arange(i, n, q) for i in range(q)]
        else:
            inds = [
                np.arange(
                    i * l,
                    (i + 1) * l,
                )
                for i in range(q)
            ]

        for i, ind in enumerate(inds):
            unflat[:, i, :] = ids[..., ind].squeeze()

        offset = (
            torch.LongTensor([i * (self.model.num_tokens // q) for i in range(q)])[
                None, :, None
            ]
            .repeat(b, 1, l)
            .to(device)
        )
        unflat = unflat - offset
        for i in range(q):
            unflat[:, i] = unflat[:, i].clamp(min=0, max=self.model.num_tokens // q - 1)

        return unflat

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        conditions: Dict[str, ConditionType],
        neg_conditions: Optional[Dict[str, ConditionType]] = None,
        prime_frames=None,
        duration_s: int = 8,
        temperature=1.0,
        topk_filter_thres=0.9,
        can_remask_prev_masked=False,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        cond_scale=3,
        critic_noise_scale=1,
        force_not_use_token_critic=False,
    ):

        has_prime = exists(prime_frames)
        if has_prime:
            prime_token_ids = prime_frames
            prime_token_length = prime_token_ids.shape[-1]

        # begin with all image token ids masked

        # assert self.num_codeboks == 1, "only 1 codebook supported  for now"
        down_sample_ratio = 4
        fps = 30

        device = next(self.parameters()).device
        duration = int(duration_s * (fps / down_sample_ratio))
        timesteps = timesteps * self.num_codeboks

        seq_len = duration
        try:
            if conditions.get("text", None) is not None:

                batch_size = conditions["text"][0].shape[0]
                if (
                    neg_conditions is not None
                    and neg_conditions.get("text", None) is not None
                ):
                    assert (
                        neg_conditions["text"][0].shape[0] == batch_size
                    ), "negetive text conditions should have same number as positive "

            elif conditions.get("audio", None) is not None:
                batch_size = conditions["audio"][0].shape[0]
                conditions["audio"][0] = conditions["audio"][0][:, :seq_len]
                conditions["audio"][1] = conditions["audio"][1][:, :seq_len]
                # seq_len = int(conditions["audio"][0].shape[1] // down_sample_ratio)
                if (
                    neg_conditions is not None
                    and neg_conditions.get("audio", None) is not None
                ):
                    assert (
                        neg_conditions["audio"][0].shape[0] == batch_size
                    ), "negetive audio conditions should have same number as positive "

            else:
                rem_keys = [
                    x for x in list(conditions.keys()) if x not in ["audio", "text"]
                ]
                batch_size = conditions[rem_keys[0]][0].shape[0]
                # seq_len = int(conditions[rem_keys[0]][0].shape[1])
                if (
                    neg_conditions is not None
                    and neg_conditions.get(rem_keys[0], None) is not None
                ):
                    assert (
                        neg_conditions[rem_keys[0]][0].shape[0] == batch_size
                    ), "negetive audio conditions should have same number as positive "

        except:
            print("using null condition")
            batch_size = 1
            conditions = self.get_null_context(batch_size, device)
            cond_scale = 1

        shape = (batch_size, self.num_codeboks, seq_len)

        if self.flatten:
            q = 2
            shape = (batch_size, self.num_codeboks, seq_len * q)
            conditions = make_copy(conditions)
            if conditions["audio"] is not None:
                cond = self.flatten_cond(
                    conditions["audio"], q, self.flatten_interleave
                )
                conditions["audio"] = cond

        ids = torch.full(shape, self.mask_token_id, dtype=torch.long, device=device)
        mask = torch.ones_like(ids)[:, 0, :].to(torch.bool)
        scores = torch.zeros(shape, dtype=torch.float32, device=device)

        starting_temperature = temperature

        demask_fn = self.model.forward_with_cond_scale
        use_token_critic = exists(self.token_critic) and not force_not_use_token_critic

        if use_token_critic:
            token_critic_fn = self.token_critic.forward_with_cond_scale

        # negative prompting, as in paper

        if exists(neg_conditions):

            demask_fn = partial(
                self.model.forward_with_neg_prompt,
                neg_conditions=neg_conditions,
            )

            if use_token_critic:
                token_critic_fn = partial(
                    self.token_critic.forward_with_neg_prompt,
                    neg_conditions=neg_conditions,
                )

        for timestep, steps_until_x0 in tqdm(
            zip(
                torch.linspace(0, 1, timesteps, device=device),
                reversed(range(timesteps)),
            ),
            total=timesteps,
        ):

            # print(steps_until_x0)
            is_last_step = steps_until_x0 == 0

            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len * 1).item()), 1)

            masked_indices = scores.topk(num_token_masked, dim=-1).indices

            ids = ids.scatter(-1, masked_indices, self.mask_token_id)

            # print(ids.shape, mask.shape)
            if has_prime:
                # ids = torch.cat(prime_token_ids , ids)
                ids[..., :prime_token_length] = prime_token_ids
                scores[..., :prime_token_length] = -1e5

            logits, embed = demask_fn(
                inputs=(ids, mask.squeeze(1)),
                conditions=conditions,
                cond_scale=cond_scale,
                return_embed=True,
            )

            filtered_logits = top_k(logits, topk_filter_thres)

            temperature = starting_temperature * (
                steps_until_x0 / timesteps
            )  # temperature is annealed

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            is_mask = ids == self.mask_token_id

            ids = torch.where(is_mask, pred_ids, ids)

            if use_token_critic and not is_last_step:
                scores = token_critic_fn(
                    inputs=(ids, mask.squeeze(1)),
                    conditions=conditions,
                    cond_scale=cond_scale,
                )

                scores = rearrange(scores, "... 1 -> ...")

                scores = scores + (
                    uniform(scores.shape, device=device) - 0.5
                ) * critic_noise_scale * (steps_until_x0 / timesteps)
            else:

                probs_without_temperature = logits.softmax(dim=-1)

                scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
                scores = rearrange(scores, "... 1 -> ...")

                if not can_remask_prev_masked:
                    scores = scores.masked_fill(~is_mask, -1e5)
                else:
                    assert (
                        self.no_mask_token_prob > 0.0
                    ), "without training with some of the non-masked tokens forced to predict, not sure if the logits will be meaningful for these token"

        # get ids

        if self.flatten:
            ids = self.unflatten(ids, q=q, interleave=self.flatten_interleave)

        return ids

    def forward(
        self,
        inputs: Dict[str, ConditionType],
        conditions: Dict[str, ConditionType],
        ignore_index: int = -100,
        cond_drop_prob=None,
        quality_list=None,
        sample_temperature=None,
        train_critic=True,
    ) -> LMOutput:

        if self.flatten:
            conditions = make_copy(conditions)
            inputs, conditions = self.flatten_input(
                inputs, conditions, self.flatten_interleave
            )

        motions_ids = inputs[0].to(torch.long)
        input_mask = inputs[1].to(torch.bool)

        B, K, T = motions_ids.shape

        x, labels, muse_mask = self.muse_mask(motions_ids, ignore_index)

        out: LMOutput = self.model(
            inputs=(x, input_mask),
            conditions=conditions,
            labels=labels,
            cond_drop_prob=cond_drop_prob,
            ignore_index=ignore_index,
            quality_list=quality_list,
        )

        if not exists(self.token_critic) or train_critic == False:
            return out

        sampled_ids = gumbel_sample(
            out.logits, temperature=default(sample_temperature, random())
        )

        # muse_mask = torch.where(x == self.mask_token_id, 1.0, 0.0).to(torch.bool)

        critic_input = torch.where(muse_mask, sampled_ids, x)
        critic_labels = (x != critic_input).float()

        bce_loss, bce_per_codebook = self.token_critic(
            inputs=(critic_input, input_mask),
            conditions=conditions,
            labels=critic_labels,
            cond_drop_prob=cond_drop_prob,
        )

        out.loss = out.loss + self.critic_loss_weight * bce_loss
        if bce_per_codebook is not None:
            for q in range(self.num_codeboks):
                out.ce_per_codebook[q] += self.critic_loss_weight * bce_per_codebook[q]

        return out


def generate_animation(
    motion_gen: MotionMuse,
    condition_provider,
    duration_s: int = 4,
    aud_file: Optional[str] = None,
    text: Optional[Union[List[str], str]] = None,
    neg_text=None,
    overlap=5,
    timesteps=24,
    use_token_critic=True,
    cond_scale=8,
    temperature=0.8,
):

    _, conditions = condition_provider(
        raw_audio=aud_file, raw_text=text, audio_max_length_s=-1
    )

    neg_conditions = None
    if neg_text is not None:
        _, neg_conditions = condition_provider(
            raw_audio=None, raw_text=text, audio_max_length_s=-1
        )
    if (
        aud_file is not None
        and (conditions["audio"][0].shape[1] // condition_provider.sampling_rate)
        < duration_s
    ):
        duration_s = conditions["audio"][0].shape[1] // condition_provider.sampling_rate

    if isinstance(text, list):
        num_iters = len(text)
    else:
        num_iters = math.ceil((duration_s * 7.5 - 30) / (30 - overlap) + 1)

    all_ids = []
    prime_frames = None
    for i in range(num_iters):
        new_conditions = {}

        if aud_file is not None:

            st = max(0, (30 - overlap) * i - overlap)
            end = st + 120
            new_conditions["audio"] = (
                conditions["audio"][0][:, st:end],
                conditions["audio"][1][:, st:end],
            )

        else:
            new_conditions["audio"] = conditions["audio"]

        if isinstance(text, list):
            new_conditions["text"] = (
                conditions["text"][0][i : i + 1],
                conditions["text"][1][i : i + 1],
            )
        else:
            new_conditions["text"] = conditions["text"]

        gen_ids = motion_gen.generate(
            conditions=new_conditions,
            neg_conditions=neg_conditions,
            prime_frames=prime_frames,
            duration_s=4,
            temperature=temperature,
            timesteps=timesteps,
            cond_scale=cond_scale,
            force_not_use_token_critic=~use_token_critic,
        )
        prime_frames = gen_ids[..., -overlap:]

        if i == 0:
            all_ids.append(gen_ids)
        else:
            all_ids.append(gen_ids[..., overlap:])

    all_ids = torch.cat(all_ids, -1)

    if isinstance(text, list):
        return all_ids
    return all_ids[..., : math.ceil(duration_s * 7.5)]
