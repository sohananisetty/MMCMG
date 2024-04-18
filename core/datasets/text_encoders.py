import logging
import re
import typing as tp
import warnings
from abc import ABC, abstractmethod
from typing import List, Union

import clip
import torch
import torch.nn as nn
from transformers import (AutoModel, AutoTokenizer, BertConfig,
                          BertForMaskedLM, BertModel,
                          ClapTextModelWithProjection,
                          CLIPTextModelWithProjection, CLIPTokenizer, T5Config,
                          T5EncoderModel, T5Tokenizer)

ConditionType = tp.Tuple[torch.Tensor, torch.Tensor]  # condition, mask
CACHE_DIR = "/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/pretrained/"


class TorchAutocast:
    """TorchAutocast utility class.
    Allows you to enable and disable autocast. This is specially useful
    when dealing with different architectures and clusters with different
    levels of support.

    Args:
        enabled (bool): Whether to enable torch.autocast or not.
        args: Additional args for torch.autocast.
        kwargs: Additional kwargs for torch.autocast
    """

    def __init__(self, enabled: bool, *args, **kwargs):
        self.autocast = torch.autocast(*args, **kwargs) if enabled else None

    def __enter__(self):
        if self.autocast is None:
            return
        try:
            self.autocast.__enter__()
        except RuntimeError:
            device = self.autocast.device
            dtype = self.autocast.fast_dtype
            raise RuntimeError(
                f"There was an error autocasting with dtype={dtype} device={device}\n"
                "If you are on the FAIR Cluster, you might need to use autocast_dtype=float16"
            )

    def __exit__(self, *args, **kwargs):
        if self.autocast is None:
            return
        self.autocast.__exit__(*args, **kwargs)


class BaseTextConditioner(ABC, nn.Module):
    """Base model for all text conditioner modules."""

    def __init__(self):
        super().__init__()

    def mean_pooling(self, token_embeddings, attention_mask):

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    @abstractmethod
    def tokenize(self, *args, **kwargs) -> tp.Dict[str, torch.Tensor]:
        """Should be any part of the processing that will lead to a synchronization
        point, e.g. BPE tokenization with transfer to the GPU.

        The returned value will be saved and return later when calling forward().
        """
        raise NotImplementedError()

    @abstractmethod
    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        raise NotImplementedError()

    @abstractmethod
    def get_text_embedding(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:

        raise NotImplementedError()


class T5Conditioner(BaseTextConditioner):
    """T5-based TextConditioner.

    Args:
        name (str): Name of the T5 model.
        output_dim (int): Output dim of the conditioner.
        finetune (bool): Whether to fine-tune T5 at train time.
        device (str): Device for T5 Conditioner.
        autocast_dtype (tp.Optional[str], optional): Autocast dtype.
        word_dropout (float, optional): Word dropout probability.
        normalize_text (bool, optional): Whether to apply text normalization.
    """

    MODELS = [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
    ]
    MODELS_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }

    def __init__(
        self,
        name: str = "t5-base",
        device: str = "cuda",
        autocast_dtype: tp.Optional[str] = "float32",
        word_dropout: float = 0.0,
    ):
        assert (
            name in self.MODELS
        ), f"Unrecognized t5 model name (should in {self.MODELS})"
        super().__init__()
        self.dim = self.MODELS_DIMS[name]
        self.device = device
        self.name = name
        self.word_dropout = word_dropout
        if autocast_dtype is None or self.device == "cpu":
            self.autocast = TorchAutocast(enabled=False)
            # if self.device != "cpu":
            #     logger.warning("T5 has no autocast, this might lead to NaN")
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            # logger.info(f"T5 will be evaluated with autocast as {autocast_dtype}")
            self.autocast = TorchAutocast(
                enabled=True, device_type=self.device, dtype=dtype
            )
        # Let's disable logging temporarily because T5 will vomit some errors otherwise.
        # thanks https://gist.github.com/simon-weber/7853144
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.t5_tokenizer = T5Tokenizer.from_pretrained(
                    name, cache_dir=CACHE_DIR
                )
                self.t5 = (
                    T5EncoderModel.from_pretrained(name, cache_dir=CACHE_DIR)
                    .eval()
                    .to(device)
                )
            finally:
                logging.disable(previous_level)

        self.t5 = self.t5.eval()
        self.freeze()

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, torch.Tensor]:
        # if current sample doesn't have a certain attribute, replace with empty string'
        if x is not None and isinstance(x, str):
            x = [x]
        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.t5_tokenizer(
            entries,
            return_tensors="pt",
            padding=True,
            truncation=False,
            # max_length=256,
        ).to(self.device)
        inputs["attention_mask"][
            empty_idx, :
        ] = 0  # zero-out index where the input is non-existant
        return inputs

    def freeze(self):
        for p in self.t5.parameters():
            p.requires_grad = False

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        mask = inputs["attention_mask"]
        with torch.no_grad(), self.autocast:
            embeds = self.t5(**inputs).last_hidden_state
        embeds = embeds * mask.unsqueeze(-1)
        return embeds, mask

    def get_text_embedding(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:

        mask = inputs["attention_mask"]

        with torch.no_grad(), self.autocast:
            embeds = self.t5(**inputs).last_hidden_state

        encoding = self.mean_pooling(embeds, mask)
        encoding = nn.functional.normalize(encoding, p=2, dim=1)

        mask = mask[:, 0:1]

        return encoding, mask


class ClipConditioner(BaseTextConditioner):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    MODELS = ["openai/clip-vit-large-patch14", "openai/clip-vit-base-patch32"]
    MODELS_DIMS = {
        "openai/clip-vit-large-patch14": 768,
        "openai/clip-vit-base-patch32": 512,
    }

    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version, cache_dir=CACHE_DIR)
        self.transformer = (
            CLIPTextModelWithProjection.from_pretrained(version, cache_dir=CACHE_DIR)
            .to(device)
            .eval()
        )
        self.dim = self.MODELS_DIMS[version]
        # self.transformer.config.projection_dim  ##768
        self.device = device
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, torch.Tensor]:
        if x is not None and isinstance(x, str):
            x = [x]
        entries = [xi if xi is not None else "" for xi in x]

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.tokenizer(
            entries,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        inputs["attention_mask"][empty_idx, :] = 0

        return inputs

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        mask = inputs["attention_mask"]
        with torch.no_grad():
            embeds = self.transformer(**inputs).last_hidden_state
        embeds = embeds * mask.unsqueeze(-1)
        return embeds, mask

    def get_text_embedding(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:

        mask = inputs["attention_mask"]
        mask = mask[:, 0:1]

        with torch.no_grad():
            embeds = self.transformer(**inputs).text_embeds

        encoding = embeds * mask

        return encoding, mask


class ClipConditioner2(nn.Module):
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name: str = "ViT-L/14",
    ):

        MODELS = ["ViT-B/32", "ViT-L/14"]

        super().__init__()
        self.MODELS_DIMS = {
            "ViT-L/14": 768,
            "ViT-B/32": 512,
        }
        self.device = device
        self.encoder, self.preprocess = clip.load(name)
        clip.model.convert_weights(self.encoder)

        self.encoder = self.encoder.eval()
        self.dim = self.MODELS_DIMS[name]
        self.freeze()

    # @property
    # def device(self):
    #     return next(self.encoder.parameters()).device

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, torch.Tensor]:
        if x is not None and isinstance(x, str):
            x = [x]
        entries = [xi if xi is not None else "" for xi in x]

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = {}

        inputs["input_ids"] = clip.tokenize(entries, truncate=True).to(self.device)
        inputs["attention_mask"] = torch.ones((len(inputs), 1)).to(self.device)  ## B

        inputs["attention_mask"][empty_idx, :] = 0

        return inputs

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        return self.get_text_embedding(inputs)

    def get_text_embedding(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:

        mask = inputs["attention_mask"]

        with torch.no_grad():
            embeds = self.encoder.encode_text(inputs["input_ids"]).float()

        embeds = embeds * mask.unsqueeze(-1)

        return embeds, mask

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = False


class BERTConditioner(BaseTextConditioner):
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name: str = "bert-base-uncased",
    ):
        super().__init__()
        self.device = device

        self.config = BertConfig.from_pretrained(name, cache_dir=CACHE_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.encoder = (
            BertModel.from_pretrained(name, cache_dir=CACHE_DIR).to(device).eval()
        )

        self.freeze()
        self.dim = self.config.hidden_size

    # @property
    # def device(self):
    #     return next(self.encoder.parameters()).device

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def tokenize(self, x: Union[str, List[str]]):

        if x is not None and isinstance(x, str):
            x = [x]
        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.tokenizer(
            entries,
            truncation=True,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        inputs["attention_mask"][empty_idx, :] = 0

        return inputs

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        # return self.get_text_embedding(inputs)
        mask = inputs["attention_mask"]

        with torch.no_grad():
            embeds = self.encoding(**inputs).last_hidden_state

        encoding = embeds * mask.unsqueeze(-1)

        return encoding, mask

    def get_text_embedding(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:

        mask = inputs["attention_mask"]
        mask = mask[:, 0:1]

        with torch.no_grad():
            embeds = self.encoding(**inputs).last_hidden_state

        encoding = embeds[:, 0] * mask.unsqueeze(-1)

        return encoding, mask


class MPNETConditioner(BaseTextConditioner):
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name: str = "sentence-transformers/all-mpnet-base-v2",
    ):
        super().__init__()
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.encoder = (
            AutoModel.from_pretrained(name, cache_dir=CACHE_DIR).to(device).eval()
        )
        self.config = self.encoder.config
        self.dim = self.config.hidden_size

        self.freeze()

    # @property
    # def device(self):
    #     return next(self.encoder.parameters()).device

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def tokenize(self, x: Union[str, List[str]]):

        if x is not None and isinstance(x, str):
            x = [x]
        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.tokenizer(
            entries,
            truncation=True,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        inputs["attention_mask"][empty_idx, :] = 0

        return inputs

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        mask = inputs["attention_mask"]
        with torch.no_grad():
            embeds = self.encoder(**inputs).last_hidden_state
        embeds = embeds * mask.unsqueeze(-1)
        return embeds, mask

    def get_text_embedding(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:

        mask = inputs["attention_mask"]

        with torch.no_grad():
            embeds = self.encoder(**inputs).last_hidden_state

        encoding = self.mean_pooling(embeds, mask)
        encoding = nn.functional.normalize(encoding, p=2, dim=1)

        mask = mask[:, 0:1]

        return encoding, mask


class ClapTextConditioner(BaseTextConditioner):
    """Uses the CLAP transformer encoder for text (from Hugging Face)"""

    def __init__(self, version="laion/larger_clap_music_and_speech", device="cuda"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(version, cache_dir=CACHE_DIR)
        self.transformer = (
            ClapTextModelWithProjection.from_pretrained(version, cache_dir=CACHE_DIR)
            .to(device)
            .eval()
        )
        self.dim = self.transformer.config.projection_dim  ##512
        self.device = device
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, torch.Tensor]:
        if x is not None and isinstance(x, str):
            x = [x]
        entries = [xi if xi is not None else "" for xi in x]

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.tokenizer(
            entries,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        inputs["attention_mask"][empty_idx, :] = 0

        return inputs

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        mask = inputs["attention_mask"]
        with torch.no_grad():
            embeds = self.transformer(**inputs).last_hidden_state
        embeds = embeds * mask.unsqueeze(-1)
        return embeds, mask

    def get_text_embedding(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:

        mask = inputs["attention_mask"]
        mask = mask[:, 0:1]

        with torch.no_grad():
            embeds = self.transformer(**inputs).text_embeds

        encoding = embeds * mask

        return encoding, mask


def getTextConditioner(text_conditioner_name, device="cuda"):
    if "t5" in text_conditioner_name:

        text_encoder = T5Conditioner(text_conditioner_name, device=device)
        text_dim = text_encoder.dim

    elif "bert" in text_conditioner_name:
        text_encoder = BERTConditioner(text_conditioner_name, device=device)
        text_dim = text_encoder.dim

    elif "clip" in text_conditioner_name:
        text_encoder = ClipConditioner(text_conditioner_name, device=device)
        text_dim = text_encoder.dim

    elif "clap" in text_conditioner_name:
        text_encoder = ClapTextConditioner(text_conditioner_name, device=device)
        text_dim = text_encoder.dim

    return text_encoder, int(text_dim)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1

    """
    re_attention = re.compile(
        r"""
    \(|
    \[|
    :\s*([+-]?[.\d]+[\.]*)\s*\)|
    \)|
    ]|
    [^\\()\[\]:]+|
    """,
        re.X,
    )

    weight_pattern = r"[+-]?\d*\.?\d+"

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            weight = re.findall(weight_pattern, weight)[0]
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:

            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res
