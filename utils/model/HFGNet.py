import torch
import torch.nn as nn
import torch.nn.functional as F

from fastmri.data import transforms
from unet import Unet
from utils.common.utils import center_crop
from typing import List, Tuple
import fastmri
import math

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
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

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
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

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

class ComplexChannelAttention(nn.Module):
    """
    채널별 복소수 attention을 적용하는 블록.
    입력 feature의 amplitude(진폭)를 기반으로 채널별 가중치를 학습하여 곱해줌.
    복소수 입력([B, C, H, W, 2])을 받아 채널 attention을 수행.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, C, H, W, 2]
        x_amp = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
        y = self.avg_pool(x_amp)
        y = y.view(y.size(0), -1)
        weights = self.fc(y).view(x.size(0), x.size(1), 1, 1, 1)
        return x * weights

class RCABlock(nn.Module):
    """
    Residual Channel Attention Block (복소수 버전).
    입력 feature에 대해 두 번의 convolution과 channel attention을 적용하고, residual 연결을 더함.
    복소수 입력([B, C, H, W, 2])을 받아 복소수 attention을 수행.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)  # real+imag concat
        self.relu = nn.ReLU(inplace=True) # 논문에는 안 나와있긴 하다.
        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1)  # output: [real, imag]
        self.ca = ComplexChannelAttention(channels)

    def forward(self, x):  # x: [B, C, H, W, 2]
        B, C, H, W, _ = x.shape
        x_ri = x.permute(0, 1, 4, 2, 3).reshape(B, C * 2, H, W)  # [B, 2C, H, W]
        out = self.relu(self.conv1(x_ri))
        out = self.conv2(out)  # [B, 2C, H, W]
        out = out.view(B, C, 2, H, W).permute(0, 1, 3, 4, 2)  # [B, C, H, W, 2]
        out = self.ca(out)
        return x + out  # residual


class RSAFusionBlock(nn.Module):
    """
    이미지 도메인과 k-space 도메인에서 각각 RCABlock을 적용한 후,
    원본 입력, 이미지 attention, k-space attention 결과를 채널 방향으로 합쳐주는 블록.
    복소수 입력([B, C, H, W, 2])을 받아 [B, C, H, W, 2] 출력.
    """
    def __init__(self, channels):
        super().__init__()
        self.rsa_image = RCABlock(channels)
        self.rsa_kspace = RCABlock(channels)
        self.fusion_conv = nn.Conv2d(channels * 2 * 3, channels * 2, kernel_size=1)

    def forward(self, x):  # x: [B, C, H, W, 2]
        rsa_img = self.rsa_image(x)
        kspace = fastmri.fft2c(x)
        rsa_k = self.rsa_kspace(kspace)
        rsa_k_img = fastmri.ifft2c(rsa_k)

        # concat along channel dim (real+imag = 2)
        out = torch.cat([x, rsa_img, rsa_k_img], dim=1)  # [B, 3C, H, W, 2]
        B, C3, H, W, _ = out.shape
        out_ri = out.permute(0, 1, 4, 2, 3).reshape(B, C3 * 2, H, W)
        out = self.fusion_conv(out_ri)  # [B, 2C, H, W]
        out = out.view(B, C3 // 3, 2, H, W).permute(0, 1, 3, 4, 2)
        return out

'''
여기를 기준으로 위에 정의된 부분이 RSA block이다. 이를 통해서 f_HF를 만든 다음, 
'''

class ComplexConv2D(nn.Module):
    """
    복소수 2D convolution 연산을 수행하는 블록.
    실수/허수 파트를 분리하여 각각 conv 연산 후 복소수로 합침.
    입력/출력 shape: [B, C, H, W, 2]
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.real_conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.imag_conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)

    def forward(self, x):  # x: [B, C, H, W, 2]
        x_r, x_i = x[..., 0], x[..., 1]  # real & imag
        real = self.real_conv(x_r) - self.imag_conv(x_i)
        imag = self.real_conv(x_i) + self.imag_conv(x_r)
        out = torch.stack([real, imag], dim=-1)  # [B, C_out, H, W, 2]
        return out


class PseudoComplexConv2D(nn.Module):
    """
    복소수 입력을 실수 채널로 펼쳐서 일반 conv2d로 처리하는 대안적 복소수 convolution 블록.
    메모리 이슈가 있을 때 사용 가능.
    입력/출력 shape: [B, C, H, W, 2]
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size, padding=padding)

    def forward(self, x):  # x: [B, C, H, W, 2]
        B, C, H, W, _ = x.shape
        x_concat = x.permute(0, 1, 4, 2, 3).reshape(B, 2 * C, H, W)  # [B, 2C, H, W]
        out = self.conv(x_concat)  # [B, 2C_out, H, W]
        out = out.view(B, -1, 2, H, W).permute(0, 1, 3, 4, 2)  # [B, C_out, H, W, 2]
        return out


'''
위는 complex conv를 수행하는 2가지 후보를 작성했다.
'''

class HFGN_HFEN_Module(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.rsa1 = RCABlock(in_ch)
        self.rsa2 = RCABlock(in_ch)
        self.complex_conv = ComplexConv2D(in_ch, in_ch)  # 또는 PseudoComplexConv2D로 변경 가능 [메모리 이슈]

    def forward(self, x):  # x: [B, 1, H, W, 2]
        x1 = self.rsa1(x)
        x2 = self.rsa2(x1)
        r_hat = self.complex_conv(x2)
        return r_hat



'''
위는 RSA 이후를 구현한 코드이다.
'''

class FeatureExtractor(nn.Module):
    """
    입력 복소수 이미지를 받아 두 번의 복소수 convolution을 거쳐 feature를 추출하는 블록.
    입력/출력 shape: [B, C, H, W, 2]
    """
    def __init__(self, in_ch=1, out_ch=1, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = ComplexConv2D(in_ch, out_ch, kernel_size, padding)
        self.conv2 = ComplexConv2D(out_ch, out_ch, kernel_size, padding)

    def forward(self, x):  # x: [B, C, H, W, 2]
        x = self.conv1(x)  # → [B, out_ch, H, W, 2]
        x = self.conv2(x)
        return x


'''
위는 Feature Extractor (FE) block을 구현한 코드이다.
'''


class HFGNet(nn.Module):
    """
    High-Frequency residual Feature Guided Network for fastMRI reconstruction.
    RSAFusionBlock, FeatureExtractor, 여러 cascade block을 조합하여 고주파 정보를 효과적으로 복원하는 네트워크.
    복합 loss(L1, SSIM, residual, k-space residual) 기반 학습을 지원.
    l1_weight(alpha): L1 loss의 가중치
    """
    def __init__(
        self,
        num_cascades: int = 8,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 16,
        pools: int = 4,
        in_ch: int = 1,
        l1_weight: float = 0.1,
        gamma: float = 0.01,
    ):
        super().__init__()
        self.rsa_fusion = RSAFusionBlock(in_ch)
        self.fe = FeatureExtractor(in_ch, in_ch)
        self.complex_conv = ComplexConv2D(in_ch, in_ch)  # loss_hfen용 추가
        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.cascades = nn.ModuleList(
            [HFGNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )
        self.l1_weight = l1_weight
        self.gamma = gamma
    
    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, x_gt: torch.Tensor = None) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        image_pred = self.sens_reduce(kspace_pred, sens_maps)

        # RSA Fusion block 적용
        rsa_fused = self.rsa_fusion(image_pred)  # [B, in_ch, H, W, 2]

        # residual (image domain)
        r = self.complex_conv(rsa_fused)
        # residual (k-space domain)
        r_k = fastmri.fft2c(r)

        # feature extractor (FE)
        fe_x = self.fe(image_pred)  # [B, in_ch, H, W, 2]

        # reconstruction module에 들어갈 변수
        input_image = torch.cat([rsa_fused, fe_x], dim=1)  # [B, 2*in_ch, H, W, 2]

        for cascade in self.cascades:
            input_image = cascade(input_image, masked_kspace, mask, sens_maps)
        
        out = fastmri.rss(fastmri.complex_abs(input_image), dim=1)
        out = center_crop(out, 384, 384)
        return out, r, r_k, image_pred


class HFGNetBlock(nn.Module):
    """
    HFGNet의 cascade를 구성하는 단위 블록.
    내부에 regularizer(예: U-Net)와 soft data consistency를 결합하여 end-to-end로 학습.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def forward(
        self,
        input_image: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:

        # model calculation
        image_updated = self.model(input_image)

        # img -> kspace
        kspace_updated = self.sens_expand(image_updated, sens_maps)

        # data consistency
        zero = torch.zeros(1, 1, 1, 1, 1).to(kspace_updated)
        soft_dc = torch.where(mask, kspace_updated - ref_kspace, zero) * self.dc_weight
        kspace_corrected = kspace_updated - self.dc_weight * soft_dc

        # kspace -> img
        image_corrected = self.sens_reduce(kspace_corrected, sens_maps)

        return image_corrected
        