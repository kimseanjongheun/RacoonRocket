"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, Tuple

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms

from unet_for_medl import UnetModel2d
from unet import Unet
from utils.common.utils import center_crop
from mri_transforms_medl import *


class gd(nn.Module): # DC block에 대응한다.
    def __init__(self):
        super(gd, self).__init__()
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

    def _forward_operator(self, image, sampling_mask, sensitivity_map):  # PFS
        forward = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=image.dtype).to(image.device),
            fft2_2c(expand_operator(image, sensitivity_map, self._coil_dim), dim=self._spatial_dims),
        )
        return forward

    def _backward_operator(self, kspace, sampling_mask, sensitivity_map):  # (PFS)^(-1)
        backward = reduce_operator(
            ifft2_2c(
                torch.where(
                    sampling_mask == 0,
                    torch.tensor([0.0], dtype=kspace.dtype).to(kspace.device),
                    kspace,
                ),
                self._spatial_dims,
            ),
            sensitivity_map,
            self._coil_dim,
        )
        return backward

    def forward(self, kspace_pred, masked_kspace, mask, sens_maps):
        Ax = self._forward_operator(kspace_pred, mask, sens_maps)
        ATAx_y = self._backward_operator(Ax - masked_kspace, mask, sens_maps)
        r = kspace_pred - self.lambda_step * ATAx_y

        return r



class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        # 복소수 채널로 나눈다고 가정 (e.g. real+imag)
        assert c % 2 == 0, "Channel 수는 반드시 짝수여야 합니다 (real + imag)"

        # group: [b, 2, c//2, h, w]
        x_group = x.view(b, 2, c // 2, h, w)

            # flatten each group
        x_flat = x_group.view(b, 2, -1)  # shape: [b, 2, c//2 * h * w]
        # 평균과 표준편차 계산
        mean = x_flat.mean(dim=2, keepdim=True)  # shape: [b, 2, 1]
        std = x_flat.std(dim=2, keepdim=True)    # shape: [b, 2, 1]
        # broadcasting을 위해 shape 복원
        mean = mean.view(b, 2, 1, 1, 1)           # [b, 2, 1, 1, 1]
        std = std.view(b, 2, 1, 1, 1)
        # normalize
        x_normed = (x_group - mean) / (std + 1e-8)
        # 다시 원래 shape로 복원
        x_out = x_normed.view(b, c, h, w)
        return x_out, mean.view(b, 2, 1, 1), std.view(b, 2, 1, 1)


    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:

        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 변수가 끝부분이 (real, image, real, image .. 4 + 2 * i 개 존재함)
        # 얘를 앞쪽으로 바꿔주면 될듯?
        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)
        
        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # get low frequency line locations and mask them out
        squeezed_mask = mask[:, 0, 0, :, 0]
        cent = squeezed_mask.shape[1] // 2
        # running argmin returns the first non-zero
        left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
        right = torch.argmin(squeezed_mask[:, cent:], dim=1)
        num_low_freqs = torch.max(
            2 * torch.min(left, right), torch.ones_like(left)
        )  # force a symmetric center unless 1
        pad = (mask.shape[-2] - num_low_freqs + 1) // 2

        x = transforms.batched_mask_center(masked_kspace, pad, pad + num_low_freqs)

        # convert to image space
        x = fastmri.ifft2c(x)
        x, b = self.chans_to_batch_dim(x)

        # estimate sensitivities
        x = self.norm_unet(x)
        x = self.batch_chans_to_chan_dim(x, b)
        x = self.divide_root_sum_of_squares(x)

        return x


class MedlNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 9,
        iterations=(3, 3, 3),
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
        """
        super().__init__()

        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.cascades = nn.ModuleList()
        if isinstance(iterations, int):
            self.cascades.append(MedlNetBlock(iters=iterations))
        else:
            for i in range(len(iterations)):
                self.cascades.append(MedlNetBlock(iters=iterations[i]))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:        
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()
        out_list = []
        for cascade in self.cascades:            
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps) + kspace_pred

            # inter_kspace_pred = self.sens_reduce(image_pred.unsqueeze(dim=1), sens_maps)
            x = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
            x = center_crop(x, 384, 384)
            out_list.append(x)
            
        return out_list # list 변수임



class MedlNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, iters=3):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()
        self.iters = iters
        self.cnn = nn.ModuleList()
        self.gd_blocks = nn.ModuleList()
        for i in range(self.iters):
            self.cnn.append(UnetModel2d(
                        in_channels=4+i*2,
                        out_channels=2,
                        num_filters=18,
                        num_pool_layers=4,
                        dropout_probability=0.0,
                    ))
            self.gd_blocks.append(gd())
        self.reg = UnetModel2d(
                        in_channels=2,
                        out_channels=2,
                        num_filters=18,
                        num_pool_layers=4,
                        dropout_probability=0.0,
                    )


    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )
    def complex_to_chan(self, x):
        # x: [B, C, H, W, 2] → [B, 2*C, H, W]
        return x.permute(0, 4, 1, 2, 3).reshape(x.size(0), 2 * x.size(1), x.size(2), x.size(3))

    def forward(self, kspace_pred, ref_kspace, mask, sens_maps): # x: image_pred of MedlNet
        gds = []
        current_x = kspace_pred

        x = self.sens_reduce(current_x, sens_maps)
        x = x.squeeze(dim=1)
        
        for i in range(self.iters):
                    
            x = self.gd_blocks[i](x, ref_kspace, mask, sens_maps)  
            gds.append(x)
            x = self.cnn[i](torch.cat((x, *gds), dim=-1))

        result = self.reg(x)
        result = result.unsqueeze(dim=1)
        result = self.sens_expand(result, sens_maps)
        return result
