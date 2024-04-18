from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Dict, List, Optional

import torch


class PositionalEmbeddingType(Enum):
    REL = "core.models.positional_embeddings.RelativePositionBias"
    SINE = "core.models.positional_embeddings.ScaledSinusoidalEmbedding"
    ALIBI = "core.models.positional_embeddings.AlibiPositionalBias"
    ABS = "core.models.positional_embeddings.AbsolutePositionalEmbedding"
    SHAW = "core.models.positional_embeddings.ShawRelativePositionalEmbedding"


class MotionRep(Enum):
    FULL = "full"
    BODY = "body"
    HAND = "hand"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"


class TextRep(Enum):
    POOLED_TEXT_EMBED = "pooled_text_embed"
    FULL_TEXT_EMBED = "full_text_embed"
    NONE = "none"


class AudioRep(Enum):
    ENCODEC = "encodec"
    LIBROSA = "librosa"
    WAV = "wav"
    CLAP = "clap"
    NONE = "none"


@dataclass
class VQVAEOutput:
    decoded_motion: torch.Tensor
    quantized_motion: torch.Tensor = None
    indices: torch.Tensor = None
    commit_loss: torch.Tensor = None
    perplexity: torch.Tensor = None


@dataclass
class MotionTokenizerParams:
    num_tokens: int = 1024
    add_additional_id: bool = False

    @property
    def pad_token_id(self):
        return self.num_tokens

    @property
    def mask_token_id(self):
        return self.pad_token_id + 1

    @property
    def special_token_id(self):
        return self.mask_token_id + 1

    @property
    def vocab_size(self):
        if self.add_additional_id:
            return self.num_tokens + 3
        return self.num_tokens + 2
