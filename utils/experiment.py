import os
from datetime import datetime


def get_exp_dir(args):
    """获取实验目录路径"""
    # 创建数据集目录
    dataset_dir = os.path.join("experiments", args.dataset)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(dataset_dir, timestamp)

    # 创建实验子目录
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

    return exp_dir


def get_latest_exp_info(model_name, dataset_name):
    """获取最新的实验目录和检查点"""
    dataset_dir = os.path.join("experiments", dataset_name)
    if not os.path.exists(dataset_dir):
        return None, None

    # 获取最新的时间戳目录
    exp_dirs = [
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]
    if not exp_dirs:
        return None, None

    latest_exp_dir = os.path.join(dataset_dir, sorted(exp_dirs)[-1])

    # 查找最新的检查点
    checkpoint_dir = os.path.join(latest_exp_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return latest_exp_dir, None

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoints:
        return latest_exp_dir, None

    latest_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])

    return latest_exp_dir, latest_checkpoint
