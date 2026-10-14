"""
数据集配置文件
包含不同数据集的默认参数设置
"""

DATASET_CONFIGS = {
    "PEMS08": {
        "adj_filename": "data/PEMS08/PEMS08.csv",
        "num_nodes": 170,
        "val_ratio": 0.2,
        "batch_size": 16,
        "weight_decay": 0.0015,
        "milestones": [20, 30, 40],
        "embed_dim": 24,
        "adaptive_embed_dim": 100,
        "l1_reg": 0.001,
        "sparsity_method": "topk",
        "topk": 32,
        "tt_ranks": [1, 20, 1],
    },
}

# 通用配置参数
DEFAULT_CONFIG = {
    "input_dim": 1,
    "output_dim": 1,
    "steps_per_day": 288,
    "days_per_week": 7,
    "lag": 12,
    "horizon": 12,
    "epochs": 200,
    "lr": 0.001,
    "test_ratio": 0.2,
    "patience": 20,
}


def get_dataset_config(dataset_name):
    """
    获取指定数据集的配置参数

    Args:
        dataset_name: 数据集名称

    Returns:
        包含所有配置参数的字典
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"未知的数据集: {dataset_name}")

    config = DEFAULT_CONFIG.copy()
    config.update(DATASET_CONFIGS[dataset_name])
    return config
