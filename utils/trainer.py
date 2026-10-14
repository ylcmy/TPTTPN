import time

import numpy as np
import torch
from tqdm import tqdm


def _process_probabilistic_output(output, y, scaler, criterion, device):
    """处理概率模型的输出，反归一化并计算损失"""
    if len(output) == 3:  # Student's t 分布
        mean, scale, df = output
        mean = scaler.inverse_transform(mean)
        if hasattr(scaler, "std"):
            std = (
                torch.from_numpy(scaler.std).to(device).type(mean.dtype)
                if isinstance(scaler.std, np.ndarray)
                else scaler.std
            )
            scale = scale * std
        loss, nll_loss, huber_loss = criterion(mean, scale, df, y)
        dist_params = torch.stack([scale.cpu(), df.cpu()], dim=-1)
        return loss, nll_loss, huber_loss, mean, dist_params
    else:  # Gaussian 分布
        mean, var = output
        mean = scaler.inverse_transform(mean)
        if hasattr(scaler, "std"):
            std = (
                torch.from_numpy(scaler.std).to(device).type(mean.dtype)
                if isinstance(scaler.std, np.ndarray)
                else scaler.std
            )
            var = var * (std**2)
        loss, nll_loss, huber_loss = criterion(mean, var, y)
        return loss, nll_loss, huber_loss, mean, var.cpu()


def train_epoch(args, model, train_loader, criterion, optimizer, scaler, device):
    """单个epoch的训练过程"""
    model.train()
    total_loss = 0
    total_nll_loss = 0  # NLL损失累计
    total_huber_loss = 0  # Huber损失累计
    is_probabilistic = False  # 是否是概率预测模型

    pbar = tqdm(train_loader, desc="Training")
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(pbar):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        # 前向传播
        output = model(x)
        is_probabilistic = isinstance(output, tuple) and len(output) >= 2

        # 处理模型输出和损失计算
        if is_probabilistic:
            loss, nll_loss, huber_loss, _, _ = _process_probabilistic_output(
                output, y, scaler, criterion, device
            )
            total_nll_loss += nll_loss.item()
            total_huber_loss += huber_loss.item()
        else:
            pred = scaler.inverse_transform(output)
            loss = criterion(pred, y)

            loss_info = f"loss: {loss.item():.4f}"

        # 添加图结构的L1正则化损失
        if args.l1_reg > 0:
            graph_l1_loss = model.get_l1_reg_loss()
            loss = loss + graph_l1_loss

        # 反向传播
        loss.backward()
        optimizer.step()

        # 更新进度条
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{total_loss/(batch_idx+1):.4f}"})

    train_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader)

    # 返回额外的损失信息
    if is_probabilistic:
        avg_nll_loss = total_nll_loss / len(train_loader)
        avg_huber_loss = total_huber_loss / len(train_loader)
        return avg_loss, train_time, avg_nll_loss, avg_huber_loss
    else:
        return avg_loss, train_time


def validate(args, model, val_loader, criterion, scaler, device):
    """验证过程"""
    model.eval()
    total_loss = 0
    total_nll_loss = 0  # NLL损失累计
    total_huber_loss = 0  # Huber损失累计
    is_probabilistic = False  # 是否是概率预测模型

    start_time = time.time()

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validating"):
            x, y = x.to(device), y.to(device)
            output = model(x)
            is_probabilistic = isinstance(output, tuple) and len(output) >= 2

            if is_probabilistic:
                loss, nll_loss, huber_loss, _, _ = _process_probabilistic_output(
                    output, y, scaler, criterion, device
                )
                total_nll_loss += nll_loss.item()
                total_huber_loss += huber_loss.item()
            else:
                pred = scaler.inverse_transform(output)
                loss = criterion(pred, y)

            total_loss += loss.item()

    val_time = time.time() - start_time
    avg_loss = total_loss / len(val_loader)

    # 返回额外的损失信息
    if is_probabilistic:
        avg_nll_loss = total_nll_loss / len(val_loader)
        avg_huber_loss = total_huber_loss / len(val_loader)
        return avg_loss, val_time, avg_nll_loss, avg_huber_loss
    else:
        return avg_loss, val_time


def test(args, model, test_loader, criterion, scaler, device):
    """测试过程"""
    model.eval()
    predictions = []  # 保存预测均值
    distribution_params = []  # 保存分布参数（方差/尺度、自由度等）
    truths = []  # 保存真实值
    total_loss = 0
    total_nll_loss = 0  # NLL损失累计
    total_huber_loss = 0  # Huber损失累计
    is_probabilistic = False  # 是否是概率预测模型
    is_student_t = False  # 是否是Student's t分布模型

    start_time = time.time()

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(device), y.to(device)
            output = model(x)
            is_probabilistic = isinstance(output, tuple) and len(output) >= 2

            if is_probabilistic:
                is_student_t = len(output) == 3
                (
                    loss,
                    nll_loss,
                    huber_loss,
                    pred,
                    dist_params_batch,
                ) = _process_probabilistic_output(output, y, scaler, criterion, device)
                total_nll_loss += nll_loss.item()
                total_huber_loss += huber_loss.item()
                distribution_params.append(dist_params_batch)
            else:
                pred = scaler.inverse_transform(output)
                loss = criterion(pred, y)

            total_loss += loss.item()
            predictions.append(pred.cpu())
            truths.append(y.cpu())

    test_time = time.time() - start_time

    # 合并所有批次的预测结果
    predictions = torch.cat(predictions, dim=0).numpy()
    truths = torch.cat(truths, dim=0).numpy()

    # 如果存在分布参数，也合并
    if is_probabilistic:
        distribution_params = torch.cat(distribution_params, dim=0).numpy()

    # 计算基本评估指标
    from lib.metrics import All_Metrics

    mae, rmse, mape = All_Metrics(predictions, truths, mask1=0.0)

    # 构建结果指标
    metrics = {
        "test_loss": total_loss / len(test_loader),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "time": test_time,
    }

    # 添加NLL和Huber损失到指标中
    if is_probabilistic:
        metrics["nll_loss"] = total_nll_loss / len(test_loader)
        metrics["huber_loss"] = total_huber_loss / len(test_loader)
        metrics["is_student_t"] = is_student_t

    # 返回预测结果、真实值、基本指标和分布参数
    return predictions, truths, metrics, distribution_params


def compute_uncertainty_metrics(
    predictions,
    truths,
    distribution_params,
    is_student_t=False,
    confidence_levels=None,
    device="cpu",
):
    """
    计算不确定性相关指标

    参数:
        predictions: 预测值，形状为 [N, T, V, D]
        truths: 真实值，形状为 [N, T, V, D]
        distribution_params: 分布参数
            - 如果是高斯分布: 方差，形状为 [N, T, V, D]
            - 如果是t分布: 尺度和自由度，形状为 [N, T, V, D, 2]
        is_student_t: 是否是Student's t分布
        confidence_levels: 置信水平列表，默认为 [0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]
        device: 计算设备，默认为"cpu"

    返回:
        uncertainty_metrics: 不确定性指标字典
        prediction_intervals: 预测区间字典，包含各置信水平的上下界
    """
    if confidence_levels is None:
        confidence_levels = [0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]

    # 转换为PyTorch张量
    predictions = torch.tensor(predictions).to(device)
    truths = torch.tensor(truths).to(device)
    distribution_params = torch.tensor(distribution_params).to(device)

    # 初始化结果字典
    uncertainty_metrics = {}
    prediction_intervals = {
        conf: {"lower": None, "upper": None} for conf in confidence_levels
    }

    # 对每个置信水平计算指标
    for conf in confidence_levels:
        if is_student_t:
            # Student's t分布
            from utils.loss import compute_student_t_picp_mpiw

            # 解析参数
            scale = distribution_params[..., 0]
            df = distribution_params[..., 1]

            # 计算PICP和MPIW
            picp, mpiw, lower, upper = compute_student_t_picp_mpiw(
                predictions, scale, df, truths, confidence=conf
            )

        else:
            # Gaussian分布
            from utils.loss import compute_picp_mpiw

            # 解析参数
            var = distribution_params

            # 计算PICP和MPIW
            picp, mpiw, lower, upper = compute_picp_mpiw(
                predictions, var, truths, confidence=conf
            )

        # 保存结果
        uncertainty_metrics[f"picp_{conf}"] = picp
        uncertainty_metrics[f"mpiw_{conf}"] = mpiw
        prediction_intervals[conf]["lower"] = lower.cpu().numpy()
        prediction_intervals[conf]["upper"] = upper.cpu().numpy()

    return uncertainty_metrics, prediction_intervals
