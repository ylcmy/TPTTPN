import numpy as np
import torch
import torch.nn as nn
import math
from torch.distributions import StudentT, Normal
from scipy import stats


def masked_mae_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MaskedMAELoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):
        return masked_mae_loss(preds, labels, null_val)


def masked_huber_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.nn.functional.huber_loss(preds, labels, delta=1, reduction="none")
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MaskedHuberLoss:
    def _get_name(self, delta=0.5):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):
        return masked_huber_loss(preds, labels, null_val)


def gaussian_nll_loss(mu, var, target, null_val=0.0):
    """
    计算高斯分布的负对数似然损失

    参数:
        mu: 均值 [B, T, N, D]，已经过反归一化
        var: 方差 [B, T, N, D]，已经过反归一化
        target: 目标值 [B, T, N, D]
        null_val: 用于掩码的空值

    返回:
        负对数似然损失
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        mask = target != null_val

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # 防止方差为0或极小值
    var = torch.clamp(var, min=1e-6)

    # 计算负对数似然: 0.5 * (log(2π) + log(σ²) + (x - μ)²/σ²)
    loss = 0.5 * (torch.log(2 * np.pi * var) + torch.pow(target - mu, 2) / var)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def student_t_nll_loss(mu, scale, df, target, null_val=0.0):
    """
    计算Student's t分布的负对数似然损失
    使用PyTorch的StudentT分布类进行严格计算

    参数:
        mu: 位置参数（均值） [B, T, N, D]，已经过反归一化
        scale: 尺度参数 [B, T, N, D]，已经过反归一化
        df: 自由度参数 [B, T, N, D]
        target: 目标值 [B, T, N, D]
        null_val: 用于掩码的空值

    返回:
        负对数似然损失
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        mask = target != null_val

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # 防止参数为0或极小值
    scale = torch.clamp(scale, min=1e-6)
    df = torch.clamp(df, min=2.1)  # 确保自由度 > 2

    # 使用PyTorch的StudentT分布类
    try:
        # 创建Student's t分布
        student_t_dist = StudentT(df=df, loc=mu, scale=scale)

        # 计算负对数似然
        nll = -student_t_dist.log_prob(target)

    except Exception as e:
        # 如果PyTorch分布类出现问题，回退到手动计算
        print(f"Warning: PyTorch StudentT failed, using manual calculation: {e}")

        # 手动计算（作为备用）
        log_gamma_half_df_plus_1 = torch.lgamma((df + 1) / 2)
        log_gamma_half_df = torch.lgamma(df / 2)
        log_pi_df = 0.5 * torch.log(math.pi * df)
        log_scale = torch.log(scale)

        # 标准化残差的平方
        standardized_residual_sq = torch.pow(target - mu, 2) / (
            df * torch.pow(scale, 2)
        )
        log_term = ((df + 1) / 2) * torch.log(1 + standardized_residual_sq)

        # 负对数似然
        nll = (
            -log_gamma_half_df_plus_1
            + log_gamma_half_df
            + log_pi_df
            + log_scale
            + log_term
        )

    nll = nll * mask
    nll = torch.where(torch.isnan(nll), torch.zeros_like(nll), nll)

    return torch.mean(nll)


class ProbabilisticLoss:
    """
    结合负对数似然损失和均值的HuberLoss
    现在使用Student's t分布替代Gaussian分布

    参数:
        alpha: NLL损失的权重
        beta: 均值HuberLoss的权重
        distribution: 使用的分布类型 ('gaussian' 或 'student_t')
    """

    def __init__(self, alpha=1.0, beta=1.0, distribution="student_t"):
        self.alpha = alpha
        self.beta = beta
        self.distribution = distribution
        self.huber = torch.nn.HuberLoss(delta=1)
        # self.huber = MaskedHuberLoss()

    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        if self.distribution == "student_t":
            return self._student_t_loss(*args, **kwargs)
        else:
            return self._gaussian_loss(*args, **kwargs)

    def _student_t_loss(self, mean, scale, df, target, null_val=0.0):
        # Student's t分布的负对数似然损失
        nll_loss = student_t_nll_loss(mean, scale, df, target, null_val)

        # 均值的HuberLoss
        huber_loss = self.huber(mean, target)

        # 组合损失
        total_loss = self.alpha * nll_loss + self.beta * huber_loss

        return total_loss, nll_loss, huber_loss

    def _gaussian_loss(self, mean, var, target, null_val=0.0):
        nll_loss = gaussian_nll_loss(mean, var, target, null_val)

        # 均值的HuberLoss
        huber_loss = self.huber(mean, target)

        # 组合损失
        total_loss = self.alpha * nll_loss + self.beta * huber_loss

        return total_loss, nll_loss, huber_loss


def compute_prediction_interval(mean, var, confidence=0.95):
    """
    计算给定置信水平的预测区间（Gaussian分布）
    使用PyTorch的Normal分布类进行严格计算

    参数:
        mean: 均值 [B, T, N, D]，已经过反归一化
        var: 方差 [B, T, N, D]，已经过反归一化
        confidence: 置信水平，默认为0.95

    返回:
        lower: 区间下界 [B, T, N, D]
        upper: 区间上界 [B, T, N, D]
    """
    # 防止方差为0或极小值
    var = torch.clamp(var, min=1e-6)
    std = torch.sqrt(var)

    # 计算分位数
    alpha = 1 - confidence
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2

    try:
        # 使用PyTorch的Normal分布类计算精确分位数
        normal_dist = Normal(loc=mean, scale=std)

        # 计算分位数
        lower = normal_dist.icdf(torch.full_like(mean, lower_quantile))
        upper = normal_dist.icdf(torch.full_like(mean, upper_quantile))

    except Exception as e:
        # 如果PyTorch分布类出现问题，回退到标准计算
        # print(f"Warning: PyTorch Normal icdf failed, using standard calculation: {e}")

        # 标准正态分布分位数
        z_scores = {
            0.5: 0.674,
            0.8: 1.282,
            0.85: 1.440,
            0.9: 1.645,
            0.95: 1.96,
            0.96: 2.054,
            0.97: 2.170,
            0.98: 2.326,
            0.99: 2.576,
        }

        z_score = z_scores.get(confidence, 1.96)
        lower = mean - z_score * std
        upper = mean + z_score * std

    return lower, upper


def compute_student_t_prediction_interval(mean, scale, df, confidence=0.95):
    """
    计算给定置信水平的预测区间（Student's t分布）
    使用scipy的t分布进行严格计算

    参数:
        mean: 位置参数（均值） [B, T, N, D]，已经过反归一化
        scale: 尺度参数 [B, T, N, D]，已经过反归一化
        df: 自由度参数 [B, T, N, D]
        confidence: 置信水平，默认为0.95

    返回:
        lower: 区间下界 [B, T, N, D]
        upper: 区间上界 [B, T, N, D]
    """
    # 防止参数为0或极小值
    scale = torch.clamp(scale, min=1e-6)
    df = torch.clamp(df, min=2.1)

    # 计算分位数
    alpha = 1 - confidence
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2

    try:
        # 将tensor转换为numpy数组进行计算
        mean_np = mean.cpu().numpy()
        scale_np = scale.cpu().numpy()
        df_np = df.cpu().numpy()

        # 计算分位数
        lower = torch.from_numpy(
            stats.t.ppf(lower_quantile, df_np, loc=mean_np, scale=scale_np)
        ).to(mean.device)
        upper = torch.from_numpy(
            stats.t.ppf(upper_quantile, df_np, loc=mean_np, scale=scale_np)
        ).to(mean.device)

    except Exception as e:
        # print(f"Warning: scipy t.ppf failed, using approximation: {e}")

        # 使用标准Student's t分布的分位数近似
        # 对于标准化的t分布，使用查表或近似公式

        # 常用置信水平的t分位数近似（基于大样本近似）
        z_scores = {
            0.5: 0.674,
            0.8: 1.282,
            0.85: 1.440,
            0.9: 1.645,
            0.95: 1.96,
            0.96: 2.054,
            0.97: 2.170,
            0.98: 2.326,
            0.99: 2.576,
        }

        z_score = z_scores.get(confidence, 1.96)

        # 使用Cornish-Fisher展开的近似
        # t_quantile ≈ z + z³/(4*df) + 5*z⁵/(96*df²) + ...
        z = torch.tensor(z_score, device=mean.device, dtype=mean.dtype)
        correction = z**3 / (4 * df) + 5 * z**5 / (96 * df**2)
        t_quantile = z + correction

        # 计算预测区间
        margin = t_quantile * scale
        lower = mean - margin
        upper = mean + margin

    return lower, upper


def compute_picp_mpiw(mean, var, target, confidence=0.9):
    """
    计算预测区间覆盖概率(PICP)和平均预测区间宽度(MPIW)（Gaussian分布）

    参数:
        mean: 预测均值 [B, T, N, D]，已经过反归一化
        var: 预测方差 [B, T, N, D]，已经过反归一化
        target: 实际目标值 [B, T, N, D]
        confidence: 置信水平

    返回:
        picp: 预测区间覆盖概率
        mpiw: 平均预测区间宽度
    """
    lower, upper = compute_prediction_interval(mean, var, confidence)

    # 计算PICP: 落在预测区间内的点的比例
    mask = (target != 0.0).float()
    in_range = ((target >= lower) & (target <= upper)).float()
    in_range_masked = in_range * mask
    picp = torch.sum(in_range_masked) / (torch.sum(mask) + 1e-10)

    # 计算MPIW: 预测区间的平均宽度
    mpiw = torch.mean(upper - lower)

    return picp.item(), mpiw.item(), lower, upper


def compute_student_t_picp_mpiw(mean, scale, df, target, confidence=0.9):
    """
    计算预测区间覆盖概率(PICP)和平均预测区间宽度(MPIW)（Student's t分布）

    参数:
        mean: 预测均值 [B, T, N, D]，已经过反归一化
        scale: 预测尺度参数 [B, T, N, D]，已经过反归一化
        df: 预测自由度参数 [B, T, N, D]
        target: 实际目标值 [B, T, N, D]
        confidence: 置信水平

    返回:
        picp: 预测区间覆盖概率
        mpiw: 平均预测区间宽度
    """
    lower, upper = compute_student_t_prediction_interval(mean, scale, df, confidence)

    # 计算PICP: 落在预测区间内的点的比例
    mask = (target != 0.0).float()
    in_range = ((target >= lower) & (target <= upper)).float()
    in_range_masked = in_range * mask
    picp = torch.sum(in_range_masked) / (torch.sum(mask) + 1e-10)

    # 计算MPIW: 预测区间的平均宽度
    mpiw = torch.mean(upper - lower)

    return picp.item(), mpiw.item(), lower, upper
