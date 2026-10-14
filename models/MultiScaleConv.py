from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTemporalConv(nn.Module):
    """
    多尺度时间卷积模块，通过多个不同膨胀率的卷积捕获不同时间尺度的依赖关系
    """

    def __init__(self, model_dim: int, num_scales: int = 3, dropout: float = 0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_scales = num_scales

        # 计算每个尺度的通道数
        self.channels_per_scale = model_dim // num_scales
        remainder = model_dim % num_scales

        # 不同膨胀率的卷积层
        self.conv_layers = nn.ModuleList()
        dilation_factors = [
            2**i for i in range(num_scales)
        ]  # [1, 2, 4] for num_scales=3

        for i, dilation in enumerate(dilation_factors):
            # 最后一个尺度可能有额外的通道
            out_channels = self.channels_per_scale + (
                remainder if i == num_scales - 1 else 0
            )

            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        model_dim,
                        out_channels,
                        kernel_size=(3, 1),  # 3×1的卷积核，只在时间维度上进行卷积
                        padding=((3 - 1) * dilation // 2, 0),  # 保持时间维度不变
                        dilation=(dilation, 1),  # 只在时间维度上进行膨胀
                    ),
                    nn.GELU(),
                    nn.BatchNorm2d(out_channels),
                    nn.Dropout(dropout),
                )
            )

        # 输出投影层，将多尺度特征整合回原始维度
        self.output_proj = nn.Sequential(
            nn.Conv2d(model_dim, model_dim, kernel_size=1), nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入x: [B, T, N, D]
        输出: [B, T, N, D]
        """
        # 转换维度以适应2D卷积: [B, D, T, N]
        x = x.permute(0, 3, 1, 2)

        # 通过不同尺度的卷积层
        outputs = []
        for conv in self.conv_layers:
            outputs.append(conv(x))

        # 拼接多尺度特征
        multi_scale_feat = torch.cat(outputs, dim=1)

        # 通过输出投影层
        output = self.output_proj(multi_scale_feat)

        # 转换回原始维度: [B, T, N, D]
        output = output.permute(0, 2, 3, 1)

        return output


class MultiScaleSpatialConv(nn.Module):
    """
    多尺度空间卷积模块，通过多个不同感受野的图卷积捕获不同空间尺度的依赖关系
    """

    def __init__(self, model_dim: int, num_scales: int = 3, dropout: float = 0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_scales = num_scales

        # 计算每个尺度的通道数
        self.channels_per_scale = model_dim // num_scales
        remainder = model_dim % num_scales

        # 不同感受野的图卷积层
        self.conv_layers = nn.ModuleList()

        for i in range(num_scales):
            # 最后一个尺度可能有额外的通道
            out_channels = self.channels_per_scale + (
                remainder if i == num_scales - 1 else 0
            )

            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        model_dim,
                        out_channels,
                        kernel_size=(1, 3),  # 1×3的卷积核，只在空间维度上进行卷积
                        padding=(0, (3 - 1) * 1 // 2),  # 保持空间维度不变
                    ),
                    nn.GELU(),
                    nn.BatchNorm2d(out_channels),
                    nn.Dropout(dropout),
                )
            )

        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Conv2d(model_dim, model_dim, kernel_size=1), nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入x: [B, T, N, D]
        输出: [B, T, N, D]
        """
        # 转换维度以适应2D卷积: [B, D, T, N]
        x = x.permute(0, 3, 1, 2)

        # 通过不同尺度的卷积层
        outputs = []
        for conv in self.conv_layers:
            outputs.append(conv(x))

        # 拼接多尺度特征
        multi_scale_feat = torch.cat(outputs, dim=1)

        # 通过输出投影层
        output = self.output_proj(multi_scale_feat)

        # 转换回原始维度: [B, T, N, D]
        output = output.permute(0, 2, 3, 1)

        return output


class MultiScaleTemporalSpatialModule(nn.Module):
    """
    结合时间和空间的多尺度依赖建模模块
    """

    def __init__(self, model_dim: int, num_scales: int = 3, dropout: float = 0.1):
        super().__init__()
        self.temporal_conv = MultiScaleTemporalConv(model_dim, num_scales, dropout)
        self.spatial_conv = MultiScaleSpatialConv(model_dim, num_scales, dropout)

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim), nn.GELU(), nn.Dropout(dropout)
        )

        # 自适应加权
        self.temporal_weight = nn.Parameter(torch.ones(1))
        self.spatial_weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入x: [B, T, N, D]
        输出: [B, T, N, D]
        """
        # 时间多尺度卷积
        temp_feat = self.temporal_conv(x)

        # 空间多尺度卷积
        spat_feat = self.spatial_conv(x)

        # 自适应加权
        weights = F.softmax(
            torch.stack([self.temporal_weight, self.spatial_weight]), dim=0
        )

        # 特征融合
        fused_feat = torch.cat([temp_feat * weights[0], spat_feat * weights[1]], dim=-1)
        output = self.fusion(fused_feat)

        return output
