from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .DGConv import DGConvLayer


class TripleGraphConv(nn.Module):
    """
    分别处理静态图和动态图的图卷积模块
    """

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim or model_dim

        # 静态图卷积层
        self.forward_gcn_layers = nn.ModuleList(
            [
                DGConvLayer(
                    model_dim if i == 0 else self.hidden_dim,
                    self.hidden_dim if i < num_layers - 1 else model_dim,
                )
                for i in range(num_layers)
            ]
        )
        self.backward_gcn_layers = nn.ModuleList(
            [
                DGConvLayer(
                    model_dim if i == 0 else self.hidden_dim,
                    self.hidden_dim if i < num_layers - 1 else model_dim,
                )
                for i in range(num_layers)
            ]
        )

        # 动态图卷积层
        self.dynamic_gcn_layers = nn.ModuleList(
            [
                DGConvLayer(
                    model_dim if i == 0 else self.hidden_dim,
                    self.hidden_dim if i < num_layers - 1 else model_dim,
                )
                for i in range(num_layers)
            ]
        )

        # 特征融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        forward_adj: torch.Tensor,
        backward_adj: torch.Tensor,
        dynamic_adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            x: 输入特征 [B, T, N, D]
            forward_adj: 前向随机游走邻接矩阵 [N, N] 或 [B, T, N, N]
            backward_adj: 后向随机游走邻接矩阵 [N, N] 或 [B, T, N, N]
            dynamic_adj: 动态邻接矩阵 [B, T, N, N]

        返回:
            处理后的特征 [B, T, N, D]
        """
        B, T, N, D = x.shape

        if forward_adj.dim() == 2:
            forward_adj = forward_adj.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        else:
            forward_adj = forward_adj.expand(B, T, -1, -1)
        if backward_adj.dim() == 2:
            backward_adj = backward_adj.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        else:
            backward_adj = backward_adj.expand(B, T, -1, -1)

        # 分别处理静态图和动态图
        forward_feat = x
        backward_feat = x.clone()
        dynamic_feat = x.clone()

        # 静态图处理
        for gcn_layer in self.forward_gcn_layers:
            forward_feat = gcn_layer(forward_feat, forward_adj)

        for gcn_layer in self.backward_gcn_layers:
            backward_feat = gcn_layer(backward_feat, backward_adj)

        # 动态图处理
        for gcn_layer in self.dynamic_gcn_layers:
            dynamic_feat = gcn_layer(dynamic_feat, dynamic_adj)

        concat_feat = torch.cat([forward_feat, backward_feat, dynamic_feat], dim=-1)
        fused_feat = self.fusion_network(concat_feat)

        return fused_feat + x
