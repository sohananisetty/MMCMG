import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.utils import default
from einops import pack, rearrange, reduce, repeat, unpack

# from vector_quantize_pytorch import ResidualVQ

# Borrow from vector_quantize_pytorch


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(logits, temperature=1.0, stochastic=False, dim=-1, training=True):
    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim=dim)

    return ind


def batched_sample_vectors(samples, num):
    def sample_vectors(samples, num):
        num_samples, device = samples.shape[0], samples.device
        if num_samples >= num:
            indices = torch.randperm(num_samples, device=device)[:num]
        else:
            indices = torch.randint(0, num_samples, (num,), device=device)

        return samples[indices]

    return sample_vectors(samples, num)
    # return torch.stack(
    #     [sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0
    # )


def batched_bincount(x, *, minlength):
    dtype, device = x.dtype, x.device
    target = torch.zeros(minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


def kmeans(
    samples,
    num_clusters,
    num_iters=10,
    sample_fn=batched_sample_vectors,
):
    dim, dtype, device = (
        samples.shape[-1],
        samples.dtype,
        samples.device,
    )

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        dists = -torch.cdist(samples, means, p=2)

        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)

        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, "... -> ... 1")

        means = torch.where(rearrange(zero_mask, "... -> ... 1"), means, new_means)

    return means, bins


class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, encdec_dim=None, kmeans_iters=None, mu=0.99):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.kmeans_iters = kmeans_iters
        self.encdec_dim = default(encdec_dim, code_dim)
        self.mu = mu  ##TO_DO
        self.requires_projection = code_dim != encdec_dim
        self.project_in = (
            nn.Linear(self.encdec_dim, code_dim)
            if code_dim != self.encdec_dim
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(code_dim, self.encdec_dim)
            if code_dim != self.encdec_dim
            else nn.Identity()
        )
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer(
            "codebook",
            torch.zeros(self.nb_code, self.code_dim, requires_grad=False).cuda(),
        )

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):

        if self.kmeans_iters is None:

            out = self._tile(x)
            self.codebook = out[: self.nb_code]
        else:
            embed, cluster_size = kmeans(
                x,
                self.nb_code,
                self.kmeans_iters,
            )
            self.codebook = embed * rearrange(cluster_size, "... -> ... 1")
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    def quantize(self, x, sample_codebook_temp=0.0):
        # N X C -> C X N
        k_w = self.codebook.t()
        # x: NT X C
        # NT X N
        distance = (
            torch.sum(x**2, dim=-1, keepdim=True)
            - 2 * torch.matmul(x, k_w)
            + torch.sum(k_w**2, dim=0, keepdim=True)
        )  # (N * L, b)

        # code_idx = torch.argmin(distance, dim=-1)

        code_idx = gumbel_sample(
            -distance,
            dim=-1,
            temperature=sample_codebook_temp,
            stochastic=True,
            training=self.training,
        )

        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)

        return x

    def get_codebook_entry(self, indices):
        return self.dequantize(indices)

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(
            self.nb_code, code_idx.shape[0], device=code_idx.device
        )  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(
            self.nb_code, x.shape[0], device=x.device
        )  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, c
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[: self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1.0 - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1.0 - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(
            self.nb_code, self.code_dim
        ) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1 - usage) * code_rand

        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def encode(self, x, temperature=0.0):
        shape = x.shape

        need_transpose = (
            True
            if (shape[-1] != self.encdec_dim or shape[-1] != self.code_dim)
            else False
        )

        if need_transpose:
            x = rearrange(x, "n c t -> (n t) c")
            N, width, T = shape
        else:
            x = rearrange(x, "n t c -> (n t) c")
            N, T, width = shape

        x = self.project_in(x)

        code_idx = self.quantize(x, temperature)

        return code_idx

    def forward(self, x, return_idx=False, temperature=0.0):
        shape = x.shape

        need_transpose = (
            True
            if (shape[-1] != self.encdec_dim or shape[-1] != self.code_dim)
            else False
        )

        if need_transpose:
            x = rearrange(x, "n c t -> (n t) c")
            N, width, T = shape
        else:
            x = rearrange(x, "n t c -> (n t) c")
            N, T, width = shape

        x = self.project_in(x)

        if self.training and not self.init:
            self.init_codebook(x)

        code_idx = self.quantize(x, temperature)
        x_d = self.dequantize(code_idx)  ## N T C

        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        x_d = self.project_out(x_d)

        if need_transpose:
            x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        else:
            x_d = x_d.view(N, T, -1).contiguous()

        # Postprocess
        code_idx = code_idx.view(N, T).contiguous()

        # print(code_idx[0])
        if return_idx:
            return x_d, code_idx, commit_loss, perplexity
        return x_d, commit_loss, perplexity


class QuantizeEMA(QuantizeEMAReset):
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(
            self.nb_code, x.shape[0], device=x.device
        )  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, c
        code_count = code_onehot.sum(dim=-1)  # nb_code

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1.0 - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1.0 - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(
            self.nb_code, self.code_dim
        ) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1 - usage) * self.codebook

        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity
