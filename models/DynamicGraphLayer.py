from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGraphLayer(nn.Module):
    def __init__(
        self,
        node_num: int,
        model_dim: int,
        sparsity_method: str = "none",
        topk: int = 10,
        l1_reg: float = 0.0,
    ) -> None:
        super().__init__()
        self.memory = nn.Parameter(torch.randn(model_dim, node_num))
        nn.init.xavier_uniform_(self.memory)
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.feature_proj = nn.Linear(model_dim, model_dim)

        # 稀疏化参数
        self.sparsity_method = sparsity_method  # 'none', 'topk', 'threshold', 'both'
        self.topk = min(topk, node_num // 5)  # top-k 保留的边数量
        self.l1_reg = l1_reg  # L1 正则化系数

        self.query_proj = nn.Linear(model_dim, model_dim)
        self.key_proj = nn.Linear(model_dim, model_dim)

    def sparse_topk(self, adj: torch.Tensor) -> torch.Tensor:
        """应用 top-k 稀疏化，只保留每个节点的 top-k 个最强连接"""
        B, T, N, _ = adj.shape
        adj = adj.reshape(B * T, N, N)

        # 为每个节点保留 top-k 个连接
        topk_values, topk_indices = torch.topk(adj, k=self.topk, dim=-1)

        sparse_adj = torch.zeros_like(adj)
        sparse_adj.scatter_(2, topk_indices, topk_values)
        sparse_adj = sparse_adj / sparse_adj.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        sparse_adj = sparse_adj.reshape(B, T, N, N)

        return sparse_adj

    def l1_regularization(self, adj: torch.Tensor) -> torch.Tensor:
        """计算邻接矩阵的 L1 正则化损失"""
        return torch.mean(torch.abs(adj))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        生成动态图结构

        参数:
            x: 输入特征 [B, T, N, D]

        返回:
            动态邻接矩阵 [B, T, N, N]
        """
        # x: [B, T, N, D]
        B, T, N, D = x.shape

        x_proj = self.feature_proj(x)  # [B, T, N, D]
        x_proj = x_proj.reshape(B * T, N, -1)  # [B*T, N, D]

        # Query-Key注意力计算
        q = self.query_proj(x_proj)  # [B*T, N, D]
        k = self.key_proj(x_proj)  # [B*T, N, D]
        dynamic_adj = F.relu(torch.matmul(q, k.transpose(-1, -2)))
        dynamic_adj = torch.softmax(dynamic_adj, dim=-1)
        dynamic_adj = dynamic_adj.reshape(B, T, N, N)  # [B, T, N, N]

        dyn_adj = torch.softmax(
            F.relu(torch.einsum("btnc, cm->btnm", x, self.memory)),
            dim=-1,
        )
        dynamic_adj = self.alpha * dynamic_adj + (1 - self.alpha) * dyn_adj

        # 应用稀疏化策略
        if self.sparsity_method == "topk":
            dynamic_adj = self.sparse_topk(dynamic_adj)

        # 存储 L1 正则化损失（如果需要）
        if self.l1_reg > 0:
            if not hasattr(self, "l1_loss"):
                self.l1_loss = self.l1_regularization(dynamic_adj) * self.l1_reg
            else:
                self.l1_loss = self.l1_regularization(dynamic_adj) * self.l1_reg

        return dynamic_adj
