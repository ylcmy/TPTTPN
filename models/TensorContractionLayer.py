from typing import List
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import opt_einsum as oe


class TuckerLayer(nn.Module):
    # def __init__(self, in_features, out_features, ranks, bias=True):
    #     super().__init__()
    #     # T维度：时间，从lag映射到horizon
    #     self.factor_T = Parameter(torch.empty(in_features[0], ranks[0]))
    #     # D维度：特征，从model_dim映射到output_dim
    #     self.factor_D = Parameter(torch.empty(in_features[1], ranks[1]))
    #     # 核心张量
    #     self.core = Parameter(torch.empty(ranks[0], ranks[1], ranks[2], ranks[3]))
    #     # 输出维度
    #     self.factor_T_out = Parameter(torch.empty(ranks[2], out_features[0]))
    #     self.factor_D_out = Parameter(torch.empty(ranks[3], out_features[1]))

    #     if bias:
    #         self.bias = Parameter(torch.empty(out_features[0], out_features[1]))
    #         nn.init.zeros_(self.bias)
    #     else:
    #         self.register_parameter("bias", None)

    #     # 初始化
    #     nn.init.xavier_uniform_(self.factor_T)
    #     nn.init.xavier_uniform_(self.factor_D)
    #     nn.init.xavier_uniform_(self.core)
    #     nn.init.xavier_uniform_(self.factor_T_out)
    #     nn.init.xavier_uniform_(self.factor_D_out)

    # def forward(self, x):  # x: [B, T, N, D]
    #     # 1. 时间维度收缩: [B, T, N, D] -> [B, R0, N, D]
    #     temp1 = torch.einsum("btnf,tr->brnf", x, self.factor_T)

    #     # 2. 特征维度收缩: [B, R0, N, D] -> [B, R0, N, R1]
    #     temp2 = torch.einsum("brnf,fs->brns", temp1, self.factor_D)

    #     # 3. 与核心张量收缩: [B, R0, N, R1] -> [B, R2, N, R3]
    #     temp3 = torch.einsum("brns,rskl->bknl", temp2, self.core)

    #     # 4. 输出时间维度收缩: [B, R2, N, R3] -> [B, H, N, R3]
    #     temp4 = torch.einsum("bknl,kh->bhnl", temp3, self.factor_T_out)

    #     # 5. 输出特征维度收缩: [B, H, N, R3] -> [B, H, N, O]
    #     output = torch.einsum("bhnl,lo->bhno", temp4, self.factor_D_out)

    #     if self.bias is not None:
    #         output = output + self.bias.unsqueeze(1).unsqueeze(0)

    #     return output

    def __init__(
        self, in_features: list, out_features: list, ranks: list, bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ranks = ranks

        self.factor_T = Parameter(torch.empty(in_features[0], ranks[0]))
        self.factor_D = Parameter(torch.empty(in_features[1], ranks[1]))
        self.W = Parameter(
            torch.empty(ranks[0], ranks[1], out_features[0], out_features[1])
        )

        if bias:
            self.bias = Parameter(torch.empty(out_features[0], out_features[1]))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.factor_T)
        nn.init.xavier_uniform_(self.factor_D)
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor):  # x: [B, T, N, D]
        # 1. 时间维度收缩: [B, T, N, D] -> [B, R0, N, D]
        temp1 = torch.einsum("btnf,tr->brnf", x, self.factor_T)

        # 2. 特征维度收缩: [B, R0, N, D] -> [B, R0, N, R1]
        temp2 = torch.einsum("brnf,fs->brns", temp1, self.factor_D)

        # 3. 与权重矩阵收缩: [B, R0, N, R1] -> [B, H, N, O]
        output = torch.einsum("brns,rskl->bknl", temp2, self.W)

        if self.bias is not None:
            output = output + self.bias.unsqueeze(1).unsqueeze(0)

        return output  # [B, horizon, N, output_dim]


class TTLayer(nn.Module):
    def __init__(
        self, in_features: list, out_features: list, tt_ranks: list, bias: bool = True
    ):
        super().__init__()
        # in_features: [lag, model_dim]
        # out_features: [horizon, output_dim]

        self.in_features = in_features
        self.out_features = out_features

        # TT-核
        self.cores = nn.ParameterList(
            [
                Parameter(torch.empty(1, in_features[0], out_features[0], tt_ranks[1])),
                Parameter(torch.empty(tt_ranks[1], in_features[1], out_features[1], 1)),
            ]
        )

        if bias:
            self.bias = Parameter(torch.empty(out_features[0], out_features[1]))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        # 初始化
        for core in self.cores:
            nn.init.xavier_uniform_(core)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, N, D]
        # output = torch.einsum(
        #     "btnf,thr,rfo->bhno", x, self.cores[0].squeeze(0), self.cores[1].squeeze(-1)
        # )

        # 沿着时间维度T收缩
        temp = torch.einsum("btnf,ithr->bhnrf", x, self.cores[0])

        # 沿着特征维度D和秩维度R收缩
        output = torch.einsum("bhnrf,rfoi->bhno", temp, self.cores[1])

        # output = oe.contract(
        #     "btnf,thr,rfo->bhno", x, self.cores[0].squeeze(0), self.cores[1].squeeze(-1)
        # )

        if self.bias is not None:
            output = output + self.bias.unsqueeze(1).unsqueeze(0)

        return output  # [B, horizon, N, output_dim]
