import typing as tp
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import (AutoProcessor, ClapAudioModelWithProjection,
                          ClapProcessor, EncodecModel)

ConditionType = tp.Tuple[torch.Tensor, torch.Tensor]  # condition, mask


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


class EncodecConditioner(nn.Module):
    def __init__(self, target_bandwidth=6, target_sr=9600, device="cuda"):
        super().__init__()
        self.encoder = (
            EncodecModel.from_pretrained(
                "facebook/encodec_24khz", cache_dir="./pretrained/"
            )
            .to(device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(
            "facebook/encodec_24khz", cache_dir="./pretrained/"
        )
        self.target_sr = target_sr
        self.device = device
        self.target_bandwidth = target_bandwidth
        self.dim = self.encoder.config.codebook_dim
        self.freeze()

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def _load_wav(self, pth: str, target_sr: Optional[int] = None):
        target_sr = target_sr if target_sr is not None else self.target_sr
        wav, sr = torchaudio.load(pth)
        audio_sample = convert_audio(wav, sr, target_sr, 1)  ## channel time

        return audio_sample

    def _decode(
        self,
        audio_codes: torch.Tensor,
        audio_scales: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        def _decode_frame(
            codes: torch.Tensor, scale: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            codes = codes.transpose(0, 1)
            embeddings = self.encoder.quantizer.decode(codes)
            # if scale is not None:
            #     embeddings = embeddings * scale.view(-1, 1, 1)

            return embeddings

        chunk_length = self.encoder.config.chunk_length
        if chunk_length is None:
            if len(audio_codes) != 1:
                raise ValueError(f"Expected one frame, got {len(audio_codes)}")
            quantized_values = _decode_frame(audio_codes[0], audio_scales[0])
        else:
            decoded_frames = []
            for code, scale in zip(audio_codes, audio_scales):

                frames = _decode_frame(code, scale)
                decoded_frames.append(frames)

            quantized_values = self.encoder._linear_overlap_add(
                decoded_frames, self.encoder.config.chunk_stride or 1
            )

        quantized_values = quantized_values.contiguous().permute(
            0, 2, 1
        )  ## b d n -> b n d

        # truncate based on padding mask
        if padding_mask is not None:
            scale_mask = padding_mask.shape[-1] // quantized_values.shape[1]
            padding_mask = (
                torch.nn.functional.interpolate(
                    padding_mask[:, None, :].float(), scale_factor=(1 / scale_mask)
                )
                .squeeze(1)
                .to(torch.bool)
            )

            quantized_values = (
                quantized_values * padding_mask[:, : quantized_values.shape[0], None]
            )

        return quantized_values, padding_mask

    def forward(
        self,
        path_or_wav: tp.Union[str, tp.List[str], np.ndarray, tp.List[np.ndarray]],
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        if not isinstance(path_or_wav, list):
            path_or_wav = [path_or_wav]

        if not isinstance(path_or_wav[0], np.ndarray):
            wavs = []
            for pth in path_or_wav:
                audio_sample = self._load_wav(pth)
                wavs.append(audio_sample[0].numpy())
            inputs = self.processor(
                raw_audio=wavs,
                sampling_rate=self.processor.sampling_rate,
                return_tensors="pt",
            )  ## B chn T

        else:
            wavs = []
            for wav in path_or_wav:
                assert len(wav.shape) == 2, "should be chn T"
                wavs.append(wav[0])
            inputs = self.processor(
                raw_audio=wavs,
                sampling_rate=self.processor.sampling_rate,
                return_tensors="pt",
            )  ## B chn T

        with torch.no_grad():
            encoder_outputs = self.encoder.encode(
                inputs["input_values"].to(self.device),
                inputs["padding_mask"].to(self.device),
                bandwidth=self.target_bandwidth,
            )
            embs, red_masks = self._decode(
                encoder_outputs.audio_codes,
                encoder_outputs.audio_scales,
                inputs["padding_mask"].to(self.device),
            )

        out_embs = []
        for emb, msk in zip(embs, red_masks):
            len_emb = torch.sum(msk)
            out_embs.append(emb[:len_emb])

        if len(out_embs) == 1:
            return out_embs[0]

        return out_embs  ## B N d


class LibrosaConditioner(nn.Module):
    def __init__(self, fps=30, device="cuda"):
        super().__init__()

        self.device = device
        self.fps = fps
        self.HOP_LENGTH = 512
        self.SR = self.fps * self.HOP_LENGTH
        self.EPS = 1e-6
        self.dim = 35

    def get_audio_feat(self, data: np.ndarray):
        envelope = librosa.onset.onset_strength(y=data, sr=self.SR)  # (seq_len,)
        mfcc = librosa.feature.mfcc(y=data, sr=self.SR, n_mfcc=20).T  # (seq_len, 20)
        chroma = librosa.feature.chroma_cens(
            y=data, sr=self.SR, hop_length=self.HOP_LENGTH, n_chroma=12
        ).T  # (seq_len, 12)

        peak_idxs = librosa.onset.onset_detect(
            onset_envelope=envelope.flatten(), sr=self.SR, hop_length=self.HOP_LENGTH
        )
        peak_onehot = np.zeros_like(envelope, dtype=np.float32)
        peak_onehot[peak_idxs] = 1.0  # (seq_len,)

        tempo, beat_idxs = librosa.beat.beat_track(
            onset_envelope=envelope,
            sr=self.SR,
            hop_length=self.HOP_LENGTH,
            tightness=100,
        )
        beat_onehot = np.zeros_like(envelope, dtype=np.float32)
        beat_onehot[beat_idxs] = 1.0  # (seq_len,)

        audio_feature = np.concatenate(
            [
                envelope[:, None],
                mfcc,
                chroma,
                peak_onehot[:, None],
                beat_onehot[:, None],
            ],
            axis=-1,
        )
        # audio_feat[audio_name] = audio_feature

        return audio_feature

    def _load_wav(self, pth: str, target_sr: Optional[int] = None):
        target_sr = target_sr if target_sr is not None else self.SR
        audio_sample, _ = librosa.load(pth, sr=target_sr)
        return audio_sample

    def forward(
        self,
        path_or_wav: tp.Union[str, tp.List[str], np.ndarray, tp.List[np.ndarray]],
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        if not isinstance(path_or_wav, list):
            path_or_wav = [path_or_wav]

        audio_features = []
        if not isinstance(path_or_wav[0], torch.Tensor):
            for path in path_or_wav:
                data = self._load_wav(path)
                audio_features.append(
                    torch.Tensor(self.get_audio_feat(data)).to(self.device)
                )

        else:
            for wav in path_or_wav:
                assert len(wav.shape) == 2, "should be chn T"
                audio_features.append(
                    torch.Tensor(self.get_audio_feat(wav[0])).to(self.device)
                )

        # embs = np.stack(audio_features, 0)

        if len(audio_features) == 1:
            return audio_features[0]

        return audio_features  ## B N d


class ClapAudioConditioner(nn.Module):
    """Uses the CLAP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self,
        version="laion/larger_clap_music_and_speech",
        target_sr=9600,
        device="cuda",
    ):
        super().__init__()
        self.processor = ClapProcessor.from_pretrained(
            version, cache_dir="./pretrained/"
        )
        self.encoder = (
            ClapAudioModelWithProjection.from_pretrained(
                version, cache_dir="./pretrained/"
            )
            .to(device)
            .eval()
        )
        self.dim = self.encoder.config.projection_dim  ##512
        self.device = device
        self.target_sr = self.processor.feature_extractor.sampling_rate
        self.freeze()

    def freeze(self):
        self.encoder = self.encoder.eval()
        for param in self.parameters():
            param.requires_grad = False

    def _load_wav(self, pth: str, target_sr: Optional[int] = None):
        target_sr = target_sr if target_sr is not None else self.target_sr
        wav, sr = torchaudio.load(pth)
        audio_sample = convert_audio(wav, sr, target_sr, 1)  ## channel time

        return audio_sample

    def forward(
        self,
        path_or_wav: tp.Union[str, tp.List[str], np.ndarray, tp.List[np.ndarray]],
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        if not isinstance(path_or_wav, list):
            path_or_wav = [path_or_wav]

        if not isinstance(path_or_wav[0], np.ndarray):
            wavs = []
            for pth in path_or_wav:
                audio_sample = self._load_wav(pth)

                wavs.append(audio_sample[0].numpy())
            inputs = self.processor(
                audios=wavs,
                sampling_rate=self.processor.feature_extractor.sampling_rate,
                return_tensors="pt",
            ).to(self.device)

        else:
            wavs = []
            for wav in path_or_wav:
                assert len(wav.shape) == 2, "should be chn T"

                wavs.append(wav[0])
            inputs = self.processor(
                audios=wavs,
                sampling_rate=self.processor.feature_extractor.sampling_rate,
                return_tensors="pt",
            ).to(self.device)

        with torch.no_grad():
            encoder_outputs = self.encoder(**inputs)
            out_embs = encoder_outputs.audio_embeds

        if len(out_embs) == 1:
            return out_embs[0][None]

        return out_embs[:, None, :]  ## B d


def getAudioConditioner(audio_rep, device="cuda"):
    if audio_rep == "encodec":
        audio_encoder = EncodecConditioner(device=device)
        audio_dim = audio_encoder.dim
    elif audio_rep == "librosa":
        audio_encoder = LibrosaConditioner(device=device)
        audio_dim = audio_encoder.dim
    elif audio_rep == "clap":
        audio_encoder = ClapAudioConditioner(device=device)
        audio_dim = audio_encoder.dim

    return audio_encoder, int(audio_dim)
