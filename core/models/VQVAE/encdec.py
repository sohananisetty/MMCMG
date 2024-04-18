import torch.nn as nn
from core.models.VQVAE.resnet import Resnet1D


class Encoder(nn.Module):
    def __init__(
        self,
        input_emb_width=3,
        codebook_dim=512,
        down_t=2,  ## 2 -> 4 downasample
        stride_t=2,
        width=None,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if width is None:
            width = codebook_dim

        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(
                    width, depth, dilation_growth_rate, activation=activation, norm=norm
                ),
            )
            blocks.append(block)

        if codebook_dim != width:
            blocks.append(nn.Conv1d(width, codebook_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_emb_width=3,
        codebook_dim=512,
        down_t=2,
        stride_t=2,
        width=None,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        blocks = []
        if width is None:
            width = codebook_dim

        if codebook_dim != width:
            blocks.append(nn.Conv1d(codebook_dim, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(
                    width,
                    depth,
                    dilation_growth_rate,
                    reverse_dilation=True,
                    activation=activation,
                    norm=norm,
                ),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(width, out_dim, 3, 1, 1),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return x

    # .permute(0, 2, 1)
