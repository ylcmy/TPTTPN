from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DGConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, K: int = 2, bias: bool = True):
        """初始化扩散图卷积层

        Args:
            in_dim: 输入特征维度
            out_dim: 输出特征维度
            K: 扩散阶数，默认为2
            bias: 是否使用偏置，默认为True
        """
        super().__init__()
        self.K = K
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 为每一阶扩散定义单独权重矩阵
        self.weights = nn.Parameter(torch.FloatTensor(K, in_dim, out_dim))
        if bias:
            self.biases = nn.Parameter(torch.FloatTensor(K, out_dim))
        else:
            self.register_parameter("biases", None)
        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        for k in range(self.K):
            nn.init.xavier_uniform_(self.weights[k])
            if self.biases is not None:
                nn.init.zeros_(self.biases[k])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: [B, T, N, D] 输入特征，B为batch size，T为时间步长，N为节点数，D为特征维度
            adj: [B, T, N, N] 已归一化的邻接矩阵

        Returns:
            [B, T, N, out_dim] 经过图卷积后的特征
        """
        B, T, N, D = x.shape
        x = x.reshape(B * T, N, D)  # [B*T, N, D]
        adj = adj.reshape(B * T, N, N)  # [B*T, N, N]

        # 计算不同阶的扩散状态并应用相应权重
        x_0 = x  # 保存初始状态作为0阶扩散
        out = torch.einsum("bni,io->bno", x_0, self.weights[0])  # 0阶扩散权重
        if self.biases is not None:
            out = out + self.biases[0]

        # 迭代计算更高阶扩散
        x_k = x_0
        for k in range(1, self.K):
            # 在当前状态基础上再扩散一步 (x_k = adj @ x_{k-1})
            x_k = torch.bmm(adj, x_k)  # [B*T, N, D]
            # 应用当前阶数对应的权重
            out_k = torch.einsum(
                "bni,io->bno", x_k, self.weights[k]
            )  # [B*T, N, out_dim]
            if self.biases is not None:
                out_k = out_k + self.biases[k]
            out = out + out_k

        out = out.reshape(B, T, N, -1)  # [B, T, N, out_dim]

        return F.gelu(out)
