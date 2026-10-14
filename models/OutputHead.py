from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .TensorContractionLayer import TuckerLayer, TTLayer


class OutputHead(nn.Module):
    """
    统一的输出头模块：支持
    - 点估计/概率预测（Gaussian 或 Student's t）
    - 张量分解后端：tucker / tt / 线性

    约定：
    - 输入 feature 形状为 [B, T, N, D_model]
    - 输出：
      - 概率（gaussian）：(mean, var)
      - 概率（student_t）：(mean, scale, df)
      - 点估计：pred
    """

    def __init__(
        self,
        lag: int,
        model_dim: int,
        horizon: int,
        output_dim: int,
        tensor_decomp: str = "tt",
        probabilistic: bool = True,
        distribution: str = "student_t",
        tucker_ranks: list = None,
        tt_ranks: list = None,
    ):
        super().__init__()
        self.lag = lag
        self.model_dim = model_dim
        self.horizon = horizon
        self.output_dim = output_dim
        self.tensor_decomp = tensor_decomp
        self.probabilistic = probabilistic
        self.distribution = distribution

        tucker_ranks = tucker_ranks or [6, 16, 6, 1]
        tt_ranks = tt_ranks or [1, 16, 1]

        if probabilistic:
            if tensor_decomp == "tucker":
                self.mean_proj = TuckerLayer(
                    in_features=[lag, model_dim],
                    out_features=[horizon, output_dim],
                    ranks=tucker_ranks,
                )
                if distribution == "student_t":
                    self.scale_proj = TuckerLayer(
                        in_features=[lag, model_dim],
                        out_features=[horizon, output_dim],
                        ranks=tucker_ranks,
                    )
                    self.df_proj = TuckerLayer(
                        in_features=[lag, model_dim],
                        out_features=[horizon, output_dim],
                        ranks=tucker_ranks,
                    )
                else:
                    self.logvar_proj = TuckerLayer(
                        in_features=[lag, model_dim],
                        out_features=[horizon, output_dim],
                        ranks=tucker_ranks,
                    )
            elif tensor_decomp == "tt":
                self.mean_proj = TTLayer(
                    in_features=[lag, model_dim],
                    out_features=[horizon, output_dim],
                    tt_ranks=tt_ranks,
                )
                if distribution == "student_t":
                    self.scale_proj = TTLayer(
                        in_features=[lag, model_dim],
                        out_features=[horizon, output_dim],
                        tt_ranks=tt_ranks,
                    )
                    self.df_proj = TTLayer(
                        in_features=[lag, model_dim],
                        out_features=[horizon, output_dim],
                        tt_ranks=tt_ranks,
                    )
                else:
                    self.logvar_proj = TTLayer(
                        in_features=[lag, model_dim],
                        out_features=[horizon, output_dim],
                        tt_ranks=tt_ranks,
                    )
            else:
                # 线性后端
                self.mean_proj = nn.Linear(lag * model_dim, horizon * output_dim)
                if distribution == "student_t":
                    self.scale_proj = nn.Linear(lag * model_dim, horizon * output_dim)
                    self.df_proj = nn.Linear(lag * model_dim, horizon * output_dim)
                else:
                    self.logvar_proj = nn.Linear(lag * model_dim, horizon * output_dim)
        else:
            if tensor_decomp == "tucker":
                self.output_proj = TuckerLayer(
                    in_features=[lag, model_dim],
                    out_features=[horizon, output_dim],
                    ranks=tucker_ranks,
                )
            elif tensor_decomp == "tt":
                self.output_proj = TTLayer(
                    in_features=[lag, model_dim],
                    out_features=[horizon, output_dim],
                    tt_ranks=tt_ranks,
                )
            else:
                self.output_proj = nn.Linear(lag * model_dim, horizon * output_dim)

    def forward(
        self, feat: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        B, T, N, _ = feat.shape
        if self.probabilistic:
            if self.tensor_decomp in ["tucker", "tt"]:
                mean = self.mean_proj(feat)
                if self.distribution == "student_t":
                    scale = F.softplus(self.scale_proj(feat))
                    df = F.softplus(self.df_proj(feat)) + 2.1
                    return mean, scale, df
                else:
                    var = F.softplus(self.logvar_proj(feat))
                    return mean, var
            else:
                # 线性后端：先拉平时间与特征
                x = feat.transpose(1, 2).reshape(B, N, T * self.model_dim)
                mean = (
                    self.mean_proj(x)
                    .view(B, N, self.horizon, self.output_dim)
                    .transpose(1, 2)
                )
                if self.distribution == "student_t":
                    scale = (
                        self.scale_proj(x)
                        .view(B, N, self.horizon, self.output_dim)
                        .transpose(1, 2)
                    )
                    df = (
                        self.df_proj(x)
                        .view(B, N, self.horizon, self.output_dim)
                        .transpose(1, 2)
                    )
                    scale = F.softplus(scale)
                    df = F.softplus(df) + 2.1
                    return mean, scale, df
                else:
                    var = (
                        self.logvar_proj(x)
                        .view(B, N, self.horizon, self.output_dim)
                        .transpose(1, 2)
                    )
                    var = F.softplus(var)
                    return mean, var
        else:
            if self.tensor_decomp in ["tucker", "tt"]:
                return self.output_proj(feat)
            else:
                x = feat.transpose(1, 2).reshape(B, N, T * self.model_dim)
                out = (
                    self.output_proj(x)
                    .view(B, N, self.horizon, self.output_dim)
                    .transpose(1, 2)
                )
                return out
