import logging
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .DynamicGraphLayer import DynamicGraphLayer
from .TransformerLayer import TransformerLayer
from .MultiScaleConv import MultiScaleTemporalSpatialModule
from .TripleGraphConv import TripleGraphConv
from .OutputHead import OutputHead


class TPTTPN(nn.Module):
    def __init__(
        self,
        node_num,
        input_dim,
        output_dim,
        embed_dim=24,
        adaptive_embed_dim=80,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.1,
        tod=True,
        dow=True,
        steps_per_day=288,
        lag=12,
        horizon=12,
        sparsity_method="none",
        topk=10,
        l1_reg=0.0,
        forward_adj=None,  # 前向随机游走邻接矩阵
        backward_adj=None,  # 后向随机游走邻接矩阵
        probabilistic=True,  # 是否使用概率预测头
        tensor_decomp="tt",  # 张量分解方法：none, tucker, tt
        tt_ranks=[1, 16, 1],
        tucker_ranks=[6, 16, 6, 1],
        distribution="student_t",
    ):
        super().__init__()
        self.input_dim = input_dim + (1 if tod else 0) + (1 if dow else 0)
        self.output_dim = output_dim
        self.adaptive_embed_dim = adaptive_embed_dim
        self.tod = tod
        self.dow = dow
        self.horizon = horizon
        self.steps_per_day = steps_per_day
        self.embed_dim = embed_dim
        self.model_dim = embed_dim * 3 + adaptive_embed_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.node_num = node_num
        self.probabilistic = probabilistic
        self.tensor_decomp = tensor_decomp
        self.distribution = distribution
        self.l1_reg = l1_reg
        self.forward_adj = forward_adj
        self.backward_adj = backward_adj

        # 输入特征编码
        self.input_proj = nn.Linear(self.input_dim, embed_dim)
        if tod:
            self.tod_embedding = nn.Embedding(self.steps_per_day, embed_dim)
        if dow:
            self.dow_embedding = nn.Embedding(7, embed_dim)
        if adaptive_embed_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(lag, node_num, adaptive_embed_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)

        # 时间和空间Transformer层
        self.temporal_transformer_layers = nn.ModuleList(
            [
                TransformerLayer(self.model_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        self.spatial_transformer_layers = nn.ModuleList(
            [
                TransformerLayer(self.model_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        # 多尺度时空依赖建模模块
        self.multi_scale_module = MultiScaleTemporalSpatialModule(
            self.model_dim, num_scales=3, dropout=dropout
        )

        # 交叉融合层
        self.cross_attention = nn.ModuleList(
            [
                TransformerLayer(self.model_dim, num_heads, ff_dim, dropout)
                for _ in range(1)
            ]
        )

        # 最终特征融合层 - 融合处理后的特征
        self.final_fusion = nn.Sequential(
            nn.Linear(self.model_dim * 3, self.model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 动态图
        self.dynamic_graph = DynamicGraphLayer(
            node_num,
            self.model_dim,
            sparsity_method=sparsity_method,
            topk=topk,
            l1_reg=l1_reg,
        )

        # 三图卷积模块（前向随机游走、后向随机游走、动态图）
        self.triple_graph_conv = TripleGraphConv(
            model_dim=self.model_dim,
            hidden_dim=self.model_dim,
            num_layers=2,
            dropout=dropout,
        )

        # 输出头
        self.output_head = OutputHead(
            lag=lag,
            model_dim=self.model_dim,
            horizon=horizon,
            output_dim=output_dim,
            tensor_decomp=tensor_decomp,
            probabilistic=probabilistic,
            distribution=self.distribution if probabilistic else "gaussian",
            tucker_ranks=tucker_ranks,
            tt_ranks=tt_ranks,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:  # x: [B, T, N, D]
        B, T, N, D = x.shape
        origin_x = x

        # 特征编码
        x = self.input_proj(x[..., : self.input_dim])
        features = [x]
        if self.tod:
            tod_emb = self.tod_embedding(
                (origin_x[..., -2] * self.steps_per_day).long()
            )
            features.append(tod_emb)
        if self.dow:
            dow_emb = self.dow_embedding(origin_x[..., -1].long())
            features.append(dow_emb)
        if self.adaptive_embed_dim > 0:
            self.adp_emb = self.adaptive_embedding.expand(
                size=(B, *self.adaptive_embedding.shape)
            )
            features.append(self.adp_emb)
        x = torch.cat(features, dim=-1)  # [B, T, N, model_dim]

        output = self._hybrid_cascade(x)

        # 统一通过输出头生成结果
        return self.output_head(output)

    def _hybrid_cascade(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 交叉注意力特征路径
        # 1.1 时间特征提取
        temporal_feat = x
        for temporal_layer in self.temporal_transformer_layers:
            temporal_feat = temporal_layer(temporal_feat, dim=1)

        # 1.2 空间特征提取
        spatial_feat = x.clone()
        for spatial_layer in self.spatial_transformer_layers:
            spatial_feat = spatial_layer(spatial_feat, dim=2)

        # 1.3 交叉注意力融合时间和空间特征
        for cross_layer in self.cross_attention:
            cross_feat = cross_layer(temporal_feat, spatial_feat, dim=2)

        # 2. 多尺度特征路径
        multi_scale_feat = self.multi_scale_module(x)

        # 3. 三图卷积特征路径
        forward_adj = (
            self.forward_adj.unsqueeze(0).unsqueeze(0)
            if hasattr(self, "forward_adj") and self.forward_adj is not None
            else None
        )

        backward_adj = (
            self.backward_adj.unsqueeze(0).unsqueeze(0)
            if hasattr(self, "backward_adj") and self.backward_adj is not None
            else None
        )

        dynamic_adj = self.dynamic_graph(x)

        triple_gcn_feat = self.triple_graph_conv(
            x, forward_adj, backward_adj, dynamic_adj
        )

        # 4. 融合三种特征路径的输出
        combined_feat = torch.cat(
            [cross_feat, multi_scale_feat, triple_gcn_feat], dim=-1
        )

        output = self.final_fusion(combined_feat)

        return output

    def get_l1_reg_loss(self):
        """获取图稀疏化的L1正则化损失"""
        l1_loss = 0.0
        if hasattr(self.dynamic_graph, "l1_loss"):
            l1_loss += self.dynamic_graph.l1_loss
        return l1_loss
