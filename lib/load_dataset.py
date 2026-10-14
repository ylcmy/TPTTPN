import os
import numpy as np
import pandas as pd


# 定义数据集元数据字典
DATASET_METADATA = {
    "PEMS03": {
        "path": "data/PEMS03/PEMS03.npz",
        "slice": (slice(None), slice(None), 0),
    },
    "PEMS04": {
        "path": "data/PEMS04/PEMS04.npz",
        "slice": (slice(None), slice(None), 0),
    },
    "PEMS07": {
        "path": "data/PEMS07/PEMS07.npz",
        "slice": (slice(None), slice(None), 0),
    },
    "PEMS08": {
        "path": "data/PEMS08/PEMS08.npz",
        "slice": (slice(None), slice(None), 0),
    },
    "PEMSD7(L)": {
        "path": "data/PEMS07(L)/PEMS07L.npz",
        "slice": (slice(None), slice(None), 0),
    },
    "PEMSD7(M)": {"path": "data/PEMS07(M)/V_228.csv"},
    "METR-LA": {"path": "data/METR-LA/data.npz", "slice": (Ellipsis, 0)},
    "PEMS-BAY": {"path": "data/PEMS-BAY/data.npz", "slice": (Ellipsis, 0)},
}


def load_st_dataset(dataset):
    """
    从文件加载时空数据集

    Args:
        dataset (str): 数据集名称

    Returns:
        np.ndarray: 加载并预处理后的数据
    """
    if dataset not in DATASET_METADATA:
        raise ValueError(f"未知的数据集: {dataset}")

    metadata = DATASET_METADATA[dataset]
    data_path = os.path.join("./", metadata["path"])

    if data_path.endswith(".npz"):
        data_raw = np.load(data_path)
        # 兼容不同npz文件可能存在的不同key
        data = data_raw.get("data", data_raw[list(data_raw.keys())[0]])
        if "slice" in metadata:
            data = data[metadata["slice"]]
    elif data_path.endswith(".csv"):
        data = pd.read_csv(data_path, header=None).to_numpy()
    else:
        raise ValueError(f"不支持的文件类型: {data_path}")

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)

    print(
        f"Load {dataset} Dataset shaped: {data.shape}, "
        f"max: {data.max():.2f}, min: {data.min():.2f}, "
        f"mean: {data.mean():.2f}, std: {data.std():.2f}, "
        f"median: {np.median(data):.2f}"
    )
    return data.astype(np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load the dataset
    data = load_st_dataset("PEMS07")

    zero_ratio = np.sum(data < 5) / data.size
    print(f"0占全部数据的比例: {zero_ratio:.4f}")
