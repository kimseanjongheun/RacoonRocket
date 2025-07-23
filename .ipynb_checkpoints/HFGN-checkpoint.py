import torch
import torch.nn as nn
import torch.nn.functional as F

from fastmri.data import transforms

class ComplexChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True), 
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, C, H, W, 2] (complex-valued: last dim is real/imag)
        # Compute amplitude
        x_amp = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)  # [B, C, H, W]
        y = self.avg_pool(x_amp)  # [B, C, 1, 1]
        y = y.view(y.size(0), -1)  # [B, C]
        weights = self.fc(y).view(x.size(0), x.size(1), 1, 1, 1)  # [B, C, 1, 1, 1]
        return x * weights  # channel-wise scaling

class RCABlock(nn.Module):
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
    def __init__(self, in_ch=1, out_ch=16, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = ComplexConv2D(in_ch, out_ch, kernel_size, padding) # 또는 PseudoComplexConv2D로 변경 가능 [메모리 이슈]
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
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
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

        self.rsa1 = RCABlock(in_ch)
        self.rsa2 = RCABlock(in_ch)
        self.complex_conv = ComplexConv2D(in_ch, in_ch)  # 또는 PseudoComplexConv2D로 변경 가능 [메모리 이슈]

        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.cascades = nn.ModuleList(
            [HFGNet(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )
    
    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        image_pred = self.sens_reduce(kspace_pred, sens_maps)

        #RSA part (HFEN)
        rsa_x1 = self.rsa1(image_pred)
        rsa_x2 = self.rsa2(rsa_x1)

        # (4) HFEN supervision loss
        r_hat = self.complex_conv(rsa_x2) # assert [B, 1, H, W, 2] -> residual (r)
        r_gt = x_gt - image_pred  # [B, 1, H, W, 2] -> GT residual (r^u)
        loss_hfen = F.l1_loss(r_hat, r_gt)

        # feature extractor (FE)
        fe_x_1 = self.conv1(image_pred)  # → [B, out_ch, H, W, 2]
        fe_x_2 = self.conv2(fe_x_1)

        # reconstruction module에 들어갈 변수
        input_image = torch.cat([rsa_x2, fe_x_2], dim=1)  # [B, C1 + C2, H, W, 2]

        for cascade in self.cascades:
            input_image = cascade(input_image, masked_kspace, mask, sens_maps) # 각 cascade에는 VarnetBlock이 들어간다
        
        result = fastmri.rss(fastmri.complex_abs(input_image), dim=1)
        result = center_crop(result, 384, 384)
        return result


class HFGNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
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