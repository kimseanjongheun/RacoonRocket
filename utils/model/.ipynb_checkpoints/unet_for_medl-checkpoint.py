# coding=utf-8
# Copyright (c) DIRECT Contributors

# Code borrowed / edited from: https://github.com/facebookresearch/fastMRI/blob/
import math
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
import fastmri

from mri_transforms_medl import *


class ConvBlock(nn.Module):
    """U-Net convolutional block.

    It consists of two convolution layers each followed by instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_probability: float):
        """Inits ConvBlock.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        dropout_probability: float
            Dropout probability.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_probability = dropout_probability

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_probability),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_probability),
        )

    def forward(self, input_data: torch.Tensor):
        """Performs the forward pass of ConvBlock.

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.layers(input_data)

    def __repr__(self):
        """Representation of ConvBlock."""
        return (
            f"ConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"dropout_probability={self.dropout_probability})"
        )


class TransposeConvBlock(nn.Module):
    """U-Net Transpose Convolutional Block.

    It consists of one convolution transpose layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Inits TransposeConvBlock.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, input_data: torch.Tensor):
        """Performs forward pass of TransposeConvBlock.

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.layers(input_data)

    def __repr__(self):
        """Representation of TransposeConvBlock."""
        return f"ConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels})"


class UnetModel2d(nn.Module): # DR block에 대응한다.
    """PyTorch implementation of a U-Net model based on [1]_.

    References
    ----------

    .. [1] Ronneberger, Olaf, et al. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, edited by Nassir Navab et al., Springer International Publishing, 2015, pp. 234–41. Springer Link, https://doi.org/10.1007/978-3-319-24574-4_28.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_filters: int = 32,
        num_pool_layers: int = 4,
        dropout_probability: float = 0.0,
    ):
        """Inits UnetModel2d.

        Parameters
        ----------
        in_channels: int
            Number of input channels to the u-net.
        out_channels: int
            Number of output channels to the u-net.
        num_filters: int
            Number of output channels of the first convolutional layer.
        num_pool_layers: int
            Number of down-sampling and up-sampling layers (depth).
        dropout_probability: float
            Dropout probability.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_pool_layers = num_pool_layers
        self.dropout_probability = dropout_probability

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_channels, num_filters, dropout_probability)])
        ch = num_filters
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, dropout_probability)]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, dropout_probability)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
            self.up_conv += [ConvBlock(ch * 2, ch, dropout_probability)]
            ch //= 2

        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
        self.up_conv += [
            nn.Sequential(
                ConvBlock(ch * 2, ch, dropout_probability),
                nn.Conv2d(ch, self.out_channels, kernel_size=1, stride=1),
            )
        ]

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def forward(self, input_data: torch.Tensor, sens_maps = None):
        """Performs forward pass of UnetModel2d.

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        # input_data.size() = (x, atb_k, mask, csm)
        # (BS, height, width, complex=2)
        # (BS, coils, height, width, complex=2)
        # (BS, 1, height, width, 1): int
        # (BS, coils, height, width, complex=2)

        if input_data.dim() == 5:
            input_data = self.sens_reduce(input_data, sens_maps)
            input_data = input_data.squeeze(dim=1)
            input_data = input_data.permute(0, 3, 1, 2) # input의 차원 위치를 바꿀 때 사용함
        elif input_data.dim() == 4:
            input_data = input_data.permute(0, 3, 1, 2) # input의 차원 위치를 바꿀 때 사용함
        else:
            raise ValueError(f"dim {input_data.dim()} of input_data is not supported.")
            
        stack = []
        output = input_data

        # Apply down-sampling layers
        for _, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
        
        output = self.conv(output)

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # Reflect pad on the right/bottom if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # Padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # Padding bottom
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        output = output.permute(0, 2, 3, 1).contiguous()
        return output


class NormUnetModel2d(nn.Module):
    """Implementation of a Normalized U-Net model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
        norm_groups: int = 2,
    ):
        """Inits NromUnetModel2d.

        Parameters
        ----------
        in_channels: int
            Number of input channels to the u-net.
        out_channels: int
            Number of output channels to the u-net.
        num_filters: int
            Number of output channels of the first convolutional layer.
        num_pool_layers: int
            Number of down-sampling and up-sampling layers (depth).
        dropout_probability: float
            Dropout probability.
        norm_groups: int,
            Number of normalization groups.
        """
        super().__init__()

        self.unet2d = UnetModel2d(
            in_channels=in_channels,
            out_channels=out_channels,
            num_filters=num_filters,
            num_pool_layers=num_pool_layers,
            dropout_probability=dropout_probability,
        )

        self.norm_groups = norm_groups

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    @staticmethod
    def norm(input_data: torch.Tensor, groups: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs group normalization."""
        # group norm
        b, c, h, w = input_data.shape
        input_data = input_data.reshape(b, groups, -1)

        mean = input_data.mean(-1, keepdim=True)
        std = input_data.std(-1, keepdim=True)

        output = (input_data - mean) / std
        output = output.reshape(b, c, h, w)

        return output, mean, std

    @staticmethod
    def unnorm(input_data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, groups: int) -> torch.Tensor:
        b, c, h, w = input_data.shape
        input_data = input_data.reshape(b, groups, -1)
        return (input_data * std + mean).reshape(b, c, h, w)

    @staticmethod
    def pad(input_data: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = input_data.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]

        output = F.pad(input_data, w_pad + h_pad)
        return output, (h_pad, w_pad, h_mult, w_mult)

    @staticmethod
    def unpad(
        input_data: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:

        return input_data[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of NormUnetModel2d.

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        print(input_data.size())
        input_data = self.complex_to_chan_dim(input_data)
        print(input_data.size())
        output, mean, std = self.norm(input_data, self.norm_groups)
        output, pad_sizes = self.pad(output)

        print(output.size())
        output = self.unet2d(output)

        output = self.unpad(output, *pad_sizes)
        output = self.unnorm(output, mean, std, self.norm_groups)
        output = self.chan_complex_to_last_dim(output)

        return output


class Unet2d(nn.Module):
    """PyTorch implementation of a U-Net model for MRI Reconstruction."""

    def __init__(
        self,
        num_filters: int = 64,
        num_pool_layers: int = 4,
        dropout_probability: float = 0.0,
        skip_connection: bool = True,
        normalized: bool = True,
        image_initialization: str = "sense",
        **kwargs,
    ):
        """Inits Unet2d.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        num_filters: int
            Number of first layer filters.
        num_pool_layers: int
            Number of pooling layers.
        dropout_probability: float
            Dropout probability.
        skip_connection: bool
            If True, skip connection is used for the output. Default: False.
        normalized: bool
            If True, Normalized Unet is used. Default: False.
        image_initialization: str
            Type of image initialization. Default: "zero-filled".
        kwargs: dict
        """
        super().__init__()
        extra_keys = kwargs.keys()
        for extra_key in extra_keys:
            if extra_key not in [
                "sensitivity_map_model",
                "model_name",
            ]:
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")
        if normalized:
            self.unet = NormUnetModel2d(
                in_channels=2,
                out_channels=2,
                num_filters=num_filters,
                num_pool_layers=num_pool_layers,
                dropout_probability=dropout_probability,
            )
        else:
            self.unet = UnetModel2d(
                in_channels=2,
                out_channels=2,
                num_filters=num_filters,
                num_pool_layers=num_pool_layers,
                dropout_probability=dropout_probability,
            )
        self.forward_operator = fft2_2c
        self.backward_operator = ifft2_2c
        self.skip_connection = skip_connection
        self.image_initialization = image_initialization
        self._coil_dim = 1
        self._spatial_dims = (2, 3)

    def compute_sense_init(self, kspace, sensitivity_map):
        r"""Computes sense initialization :math:`x_{\text{SENSE}}`:

        .. math::
            x_{\text{SENSE}} = \sum_{k=1}^{n_c} {S^{k}}^* \times y^k

        where :math:`y^k` denotes the data from coil :math:`k`.

        Parameters
        ----------
        kspace: torch.Tensor
            k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).

        Returns
        -------
        input_image: torch.Tensor
            Sense initialization :math:`x_{\text{SENSE}}`.
        """
        input_image = complex_multiplication(
            conjugate(sensitivity_map),
            self.backward_operator(kspace, dim=self._spatial_dims),
        )
        input_image = input_image.sum(self._coil_dim)
        return input_image

    def forward(
        self,
        atb,
        masked_kspace: torch.Tensor,
        mask,
        sensitivity_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes forward pass of Unet2d.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2). Default: None.

        Returns
        -------
        output: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        if self.image_initialization == "sense":
            if sensitivity_map is None:
                raise ValueError("Expected sensitivity_map not to be None with 'sense' image_initialization.")
            input_image = self.compute_sense_init(
                kspace=masked_kspace,
                sensitivity_map=sensitivity_map,
            )
        elif self.image_initialization == "zero_filled":
            input_image = self.backward_operator(masked_kspace).sum(self._coil_dim)
        else:
            raise ValueError(
                f"Unknown image_initialization. Expected `sense` or `zero_filled`. "
                f"Got {self.image_initialization}."
            )

        # input_image = atb
        # output = self.unet(input_image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        output = self.unet(input_image)
        if self.skip_connection:
            output += input_image
        return output

