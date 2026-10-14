import argparse
import os

import random
from copy import deepcopy
import ast
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchinfo import summary

from config.dataset_config import get_dataset_config
from lib.adj_generate import *
from lib.dataloader import get_dataloader
from lib.metrics import All_Metrics
from models.TPTTPN import TPTTPN
from utils import *


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置 cudnn 的随机性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    # 基本参数
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
    )
    parser.add_argument("--model", type=str, default="TPTTPN")
    parser.add_argument("--dataset", type=str, default="PEMS08")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--exp_dir", type=str)

    parser.add_argument("--num_nodes", type=int, default=170)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--tod", type=ast.literal_eval, default=True)
    parser.add_argument("--dow", type=ast.literal_eval, default=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--steps_per_day", type=int, default=288)
    parser.add_argument("--days_per_week", type=int, default=7)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--id_filename", type=str, default=None)

    # 概率预测参数
    parser.add_argument(
        "--probabilistic",
        type=ast.literal_eval,
        default=True,
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="student_t",
        choices=["gaussian", "student_t"],
    )
    parser.add_argument(
        "--nll_weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--huber_weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--tensor_decomp",
        type=str,
        default="tt",
        choices=["none", "tucker", "tt"],
    )
    parser.add_argument(
        "--tt_ranks",
        type=int,
        nargs="+",
        default=[1, 16, 1],
    )
    parser.add_argument(
        "--tucker_ranks",
        type=int,
        nargs="+",
        default=[6, 16, 6, 1],
    )

    # 图稀疏化参数
    parser.add_argument(
        "--sparsity_method",
        type=str,
        default="None",
        choices=["None", "topk"],
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
    )

    parser.add_argument("--l1_reg", type=float, default=0.0)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="reduce_on_plateau",
        choices=[
            "reduce_on_plateau",
            "multi_step",
        ],
    )
    parser.add_argument("--lr_decay_factor", type=float, default=0.2)
    parser.add_argument("--lr_decay_patience", type=int, default=4)
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=[10, 20, 30],
    )
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--seed", type=int, default=2025)

    args = parser.parse_args()
    # args.seed = random.randint(0, 1e4)

    # 检查测试模式下的实验目录和检查点
    if args.mode == "test":
        if args.exp_dir is None:
            # 自动获取最新的实验目录和检查点
            latest_exp_dir, latest_checkpoint = get_latest_exp_info(
                args.model, args.dataset
            )

            args.exp_dir = latest_exp_dir
            print(f"自动选择最新的实验目录: {args.exp_dir}")
            if latest_checkpoint is not None:
                args.checkpoint = latest_checkpoint
                print(f"自动选择最新的检查点: {args.checkpoint}")
            else:
                parser.error(f"在目录 {args.exp_dir} 中未找到有效的检查点文件")
        else:
            args.checkpoint = os.path.join(
                args.exp_dir, "checkpoints", f"{args.model}.pth"
            )

    # 获取数据集默认配置
    config = get_dataset_config(args.dataset)

    # 使用命令行参数覆盖默认配置
    for key, value in vars(args).items():
        if key not in config:
            config[key] = value

    # 将配置转换为命名空间对象
    return argparse.Namespace(**config)


def get_lr_scheduler(optimizer, args):
    """获取学习率调度器"""
    if args.lr_scheduler == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_decay_factor,
            patience=args.lr_decay_patience,
            min_lr=args.min_lr,
        )
    elif args.lr_scheduler == "multi_step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.milestones,
            gamma=args.lr_decay_factor,
        )

    return scheduler


def train(
    args,
    model,
    train_loader,
    val_loader,
    test_loader,
    scaler,
    device,
    logger,
    exp_dir,
    criterion,
):
    """训练过程"""
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_lr_scheduler(optimizer, args)

    # 训练过程
    best_val_loss = float("inf")
    best_model_state = None
    patience = args.patience
    patience_counter = 0
    total_train_time = 0
    total_val_time = 0
    total_test_time = 0

    # 记录损失和学习率
    train_losses = []
    val_losses = []
    test_losses = []
    nll_losses = []  # NLL损失历史
    huber_losses = []  # Huber损失历史
    test_rmses = []
    test_maes = []
    test_mapes = []
    learning_rates = []

    logger.info("Starting training...")
    for epoch in range(args.epochs):
        # 训练
        if args.probabilistic:
            train_loss, train_time, train_nll, train_huber = train_epoch(
                args, model, train_loader, criterion, optimizer, scaler, device
            )
            val_loss, val_time, val_nll, val_huber = validate(
                args, model, val_loader, criterion, scaler, device
            )
            # 记录NLL和Huber损失
            nll_losses.append(train_nll)
            huber_losses.append(train_huber)
        else:
            train_loss, train_time = train_epoch(
                args, model, train_loader, criterion, optimizer, scaler, device
            )
            val_loss, val_time = validate(
                args, model, val_loader, criterion, scaler, device
            )

        # 在测试集上评估
        if args.probabilistic:
            _, _, test_metrics, _ = test(
                args, model, test_loader, criterion, scaler, device
            )
        else:
            _, _, test_metrics = test(
                args, model, test_loader, criterion, scaler, device
            )

        test_loss = test_metrics["test_loss"]
        test_time = test_metrics["time"]

        # 记录损失和指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        test_rmses.append(test_metrics["rmse"])
        test_maes.append(test_metrics["mae"])
        test_mapes.append(test_metrics["mape"])

        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)

        # 累计时间
        total_train_time += train_time
        total_val_time += val_time
        total_test_time += test_time

        # 记录日志
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        if args.probabilistic:
            logger.info(
                f"Train Loss: {train_loss:.4f} (NLL: {train_nll:.4f}, Huber: {train_huber:.4f}) (Time: {train_time:.2f}s)"
            )
            logger.info(
                f"Val Loss: {val_loss:.4f} (NLL: {val_nll:.4f}, Huber: {val_huber:.4f}) (Time: {val_time:.2f}s)"
            )
            if "nll_loss" in test_metrics and "huber_loss" in test_metrics:
                logger.info(
                    f"Test Loss: {test_loss:.4f} (NLL: {test_metrics['nll_loss']:.4f}, Huber: {test_metrics['huber_loss']:.4f}), RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, MAPE: {test_metrics['mape']:.4f} (Time: {test_time:.2f}s)"
                )
            else:
                logger.info(
                    f"Test Loss: {test_loss:.4f}, RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, MAPE: {test_metrics['mape']:.4f} (Time: {test_time:.2f}s)"
                )
        else:
            logger.info(f"Train Loss: {train_loss:.4f} (Time: {train_time:.2f}s)")
            logger.info(f"Val Loss: {val_loss:.4f} (Time: {val_time:.2f}s)")
            logger.info(
                f"Test Loss: {test_loss:.4f}, RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, MAPE: {test_metrics['mape']:.4f} (Time: {test_time:.2f}s)"
            )

        logger.info(f"Learning Rate: {current_lr:.6f}")

        # 更新学习率
        if args.lr_scheduler == "reduce_on_plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # 保存最佳模型状态
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
            logger.info(f"New best model with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= patience:
            logger.info("Early stopping triggered")
            break

    # 绘制损失曲线
    loss_curve_path = os.path.join(exp_dir, "loss_curve.png")
    plot_loss_curve(train_losses, val_losses, test_losses, loss_curve_path)
    logger.info(f"Loss curve saved to {loss_curve_path}")

    # 绘制测试集指标曲线
    metrics_curve_path = os.path.join(exp_dir, "test_metrics_curve.png")
    plot_metrics_curve(test_rmses, test_maes, test_mapes, metrics_curve_path)
    logger.info(f"Test metrics curve saved to {metrics_curve_path}")

    # 记录总训练时间
    logger.info(f"Total training time: {total_train_time:.2f}s")
    logger.info(f"Total validation time: {total_val_time:.2f}s")
    logger.info(f"Total test time: {total_test_time:.2f}s")
    logger.info(f"Average epoch training time: {total_train_time/(epoch+1):.2f}s")
    logger.info(f"Average epoch validation time: {total_val_time/(epoch+1):.2f}s")
    logger.info(f"Average epoch test time: {total_test_time/(epoch+1):.2f}s")

    # 保存最佳模型
    model.load_state_dict(best_model_state)

    best_model_path = os.path.join(exp_dir, "checkpoints", f"{args.model}.pth")
    torch.save(best_model_state, best_model_path)
    logger.info(f"Best model saved to {best_model_path}")

    # 保存训练历史
    history_path = os.path.join(exp_dir, "training_history.npz")
    if args.probabilistic:
        np.savez(
            history_path,
            train_losses=train_losses,
            val_losses=val_losses,
            test_losses=test_losses,
            nll_losses=nll_losses,
            huber_losses=huber_losses,
            test_rmses=test_rmses,
            test_maes=test_maes,
            test_mapes=test_mapes,
            learning_rates=learning_rates,
            train_time=total_train_time,
            val_time=total_val_time,
            test_time=total_test_time,
        )
    else:
        np.savez(
            history_path,
            train_losses=train_losses,
            val_losses=val_losses,
            test_losses=test_losses,
            test_rmses=test_rmses,
            test_maes=test_maes,
            test_mapes=test_mapes,
            learning_rates=learning_rates,
            train_time=total_train_time,
            val_time=total_val_time,
            test_time=total_test_time,
        )
    logger.info(f"Training history saved to {history_path}")


def evaluate(args, model, test_loader, scaler, device, logger, exp_dir, criterion):
    """测试过程"""
    # 执行测试并获取结果
    if args.probabilistic:
        predictions, truths, metrics, distribution_params = test(
            args, model, test_loader, criterion, scaler, device
        )
    else:
        predictions, truths, metrics = test(
            args, model, test_loader, criterion, scaler, device
        )

    # 记录测试结果
    logger.info("\nTest Results:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")

    # 打印NLL和Huber损失
    if args.probabilistic and "nll_loss" in metrics and "huber_loss" in metrics:
        logger.info(f"NLL Loss: {metrics['nll_loss']:.4f}")
        logger.info(f"Huber Loss: {metrics['huber_loss']:.4f}")

    for i in range(predictions.shape[1]):
        mae, rmse, mape = All_Metrics(
            predictions[:, i, ...], truths[:, i, ...], mask1=0.0
        )
        logger.info(
            "Step %d RMSE = %.4f, MAE = %.4f, MAPE = %.4f"
            % (
                i + 1,
                rmse,
                mae,
                mape,
            )
        )

    # 如果是概率预测模型，记录每个置信水平的PICP和MPIW
    if args.probabilistic:
        # 单独计算不确定性指标
        from utils.trainer import compute_uncertainty_metrics

        logger.info("计算不确定性指标...")
        is_student_t = metrics.get("is_student_t", False)
        uncertainty_metrics, prediction_intervals = compute_uncertainty_metrics(
            predictions,
            truths,
            distribution_params,
            is_student_t=is_student_t,
        )

        # 将不确定性指标合并到测试指标中
        metrics.update(uncertainty_metrics)

        logger.info("\n概率预测指标:")
        for conf in [0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]:
            if f"picp_{conf}" in metrics and f"mpiw_{conf}" in metrics:
                logger.info(f"置信水平 {conf}:")
                logger.info(
                    f"PICP = {metrics[f'picp_{conf}']:.4f}, MPIW = {metrics[f'mpiw_{conf}']:.4f}"
                )

    # 保存测试结果
    results_path = os.path.join(exp_dir, "test_metrics.npy")
    np.save(results_path, metrics)
    logger.info(f"Test metrics saved to {results_path}")

    # 保存预测结果
    if args.probabilistic:
        predictions_path = os.path.join(exp_dir, "predictions.npz")
        np.savez(
            predictions_path,
            predictions=predictions,
            truths=truths,
            prediction_intervals=prediction_intervals,
            distribution_params=distribution_params,
        )
    else:
        predictions_path = os.path.join(exp_dir, "predictions.npz")
        np.savez(predictions_path, predictions=predictions, truths=truths)

    logger.info(f"Predictions saved to {predictions_path}")


def main():
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建或获取实验目录
    if args.mode == "train":
        exp_dir = get_exp_dir(args)
    else:
        exp_dir = args.exp_dir

    # 设置日志记录器
    logger = setup_logger(args, exp_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running in {args.mode} mode with config:")
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")
    logger.info(f"Device: {device}")
    logger.info(f"Experiment directory: {exp_dir}")

    # 数据加载
    train_loader, val_loader, test_loader, scaler = get_dataloader(
        args, normalizer="std", tod=args.tod, dow=args.dow
    )
    logger.info(f"Train data's mean: {scaler.mean}, std: {scaler.std}")

    # 定义损失函数
    dataset = args.dataset.upper()
    if args.probabilistic:
        # 使用概率损失函数，根据命令行参数选择分布类型
        criterion = ProbabilisticLoss(
            alpha=args.nll_weight,
            beta=args.huber_weight,
            distribution=args.distribution,
        )
        distribution_name = (
            "Student's t分布" if args.distribution == "student_t" else "高斯分布"
        )
        logger.info(
            f"使用概率损失函数 ({distribution_name}): alpha={args.nll_weight}, beta={args.huber_weight}"
        )
    else:
        if dataset in ("METR-LA", "PEMS-BAY"):
            criterion = MaskedMAELoss()
        elif dataset in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
            criterion = nn.HuberLoss()
        else:
            raise ValueError("Unsupported dataset.")

    logger.info(f"Loss function: {criterion.__class__.__name__}")

    if dataset in ("METR-LA", "PEMS-BAY"):
        adj_list, _ = load_adj(os.path.join("./data", dataset, "adj_mx.pkl"))
    else:
        # 生成静态距离邻接矩阵
        adj_filename = args.adj_filename
        id_filename = args.id_filename
        adj_mx, distance_mx = get_adjacency_matrix(
            adj_filename, args.num_nodes, id_filename
        )

        # 生成双向随机游走矩阵
        adj_list = [
            calculate_transition_matrix(adj_mx).T,
            calculate_transition_matrix(adj_mx.T).T,
        ]

    # 转换为张量
    forward_adj = torch.FloatTensor(adj_list[0].todense()).to(device)  # 前向随机游走
    backward_adj = torch.FloatTensor(adj_list[1].todense()).to(device)  # 后向随机游走

    logger.info(f"Forward adjacency matrix: {forward_adj}")
    logger.info(f"Backward adjacency matrix: {backward_adj}")

    # 初始化模型
    model = TPTTPN(
        args.num_nodes,
        args.input_dim,
        args.output_dim,
        embed_dim=args.embed_dim,
        adaptive_embed_dim=args.adaptive_embed_dim,
        tod=args.tod,
        dow=args.dow,
        steps_per_day=args.steps_per_day,
        lag=args.lag,
        horizon=args.horizon,
        sparsity_method=args.sparsity_method,
        topk=args.topk,
        l1_reg=args.l1_reg,
        forward_adj=forward_adj,  # 前向随机游走矩阵
        backward_adj=backward_adj,  # 后向随机游走矩阵
        probabilistic=args.probabilistic,  # 是否使用概率预测
        tensor_decomp=args.tensor_decomp,
        tt_ranks=args.tt_ranks,
        tucker_ranks=args.tucker_ranks,
        distribution=args.distribution,
    ).to(device)

    logger.info("Model Summary:")
    # 计算输入形状
    batch_size = args.batch_size
    seq_length = args.lag
    num_nodes = args.num_nodes
    input_dim = args.input_dim + 2 if args.tod and args.dow else args.input_dim

    model_summary = summary(
        model,
        input_size=(batch_size, seq_length, num_nodes, input_dim),
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds",
        ],
        depth=5,
        device=device,
        verbose=0,
    )

    logger.info("\n" + str(model_summary))

    if args.mode == "train":
        train(
            args,
            model,
            train_loader,
            val_loader,
            test_loader,
            scaler,
            device,
            logger,
            exp_dir,
            criterion,
        )
        # 评估模型
        evaluate(
            args,
            model,
            test_loader,
            scaler,
            device,
            logger,
            exp_dir,
            criterion,
        )

    else:
        # 测试模式
        # 加载预训练模型
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint)
            logger.info(f"成功加载检查点: {args.checkpoint}")
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return

        # 评估模型
        evaluate(
            args,
            model,
            test_loader,
            scaler,
            device,
            logger,
            exp_dir,
            criterion,
        )


if __name__ == "__main__":
    CUDA_LAUNCH_BLOCKING = 1
    main()
