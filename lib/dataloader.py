import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import numpy as np
import torch
import torch.utils.data

from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset
from lib.normalization import (
    ColumnMinMaxScaler,
    MinMax01Scaler,
    MinMax11Scaler,
    NScaler,
    StandardScaler,
)


SCALER_REGISTRY = {
    "max01": MinMax01Scaler,
    "max11": MinMax11Scaler,
    "std": StandardScaler,
    "None": NScaler,
    "cmax": ColumnMinMaxScaler,
}


def normalize_dataset(data, normalizer, column_wise=False):
    """
    根据给定的归一化方法初始化一个scaler

    Args:
        data (np.ndarray): 用于计算归一化参数的数据 (通常是训练集)
        normalizer (str): 归一化方法的名称
        column_wise (bool): 是否按列进行归一化

    Returns:
        object: 初始化好的scaler对象
    """
    if normalizer not in SCALER_REGISTRY:
        raise ValueError(f"未知的归一化方法: {normalizer}")

    scaler_class = SCALER_REGISTRY[normalizer]

    if normalizer == "None":
        scaler = scaler_class()
        print("Does not normalize the dataset")
    elif normalizer == "cmax":
        scaler = scaler_class(data.min(axis=0), data.max(axis=0))
        print("Normalize the dataset by Column Min-Max Normalization")
    else:
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
            mean = data.mean()
            std = data.std()

        if normalizer in ["max01", "max11"]:
            scaler = scaler_class(minimum, maximum)
            print(f"Normalize the dataset by {scaler_class.__name__}")
        elif normalizer == "std":
            scaler = scaler_class(mean, std)
            print("Normalize the dataset by Standard Normalization")

    return scaler


def split_data_by_days(data, val_days, test_days, interval=30):
    """
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    """
    T = int((24 * 60) / interval)
    x = -T * int(test_days)
    test_data = data[-T * int(test_days) :]
    val_data = data[-T * int(test_days + val_days) : -T * int(test_days)]
    train_data = data[: -T * int(test_days + val_days)]
    return train_data, val_data, test_data


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio) :]
    val_data = data[
        -int(data_len * (test_ratio + val_ratio)) : -int(data_len * test_ratio)
    ]
    train_data = data[: -int(data_len * (test_ratio + val_ratio))]
    return train_data, val_data, test_data


def data_loader(X, Y, batch_size, shuffle=True):
    # cuda = True if torch.cuda.is_available() else False
    # TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    TensorFloat = torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader


def get_dataloader(args, normalizer="std", tod=False, dow=False, single=False):
    # load raw st dataset
    data = load_st_dataset(args.dataset)
    L, N, F = data.shape

    # 1. 准备时间特征
    feature_list = [data]

    if tod:
        time_ind = [
            i % args.steps_per_day / args.steps_per_day for i in range(data.shape[0])
        ]
        time_in_day = np.tile(np.array(time_ind), [1, N, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

    if dow:
        day_in_week = [
            (i // args.steps_per_day) % args.days_per_week for i in range(data.shape[0])
        ]
        day_in_week = np.tile(np.array(day_in_week), [1, N, 1]).transpose((2, 1, 0))
        feature_list.append(day_in_week)

    x, y = Add_Window_Horizon(data, args.lag, args.horizon, single)
    x_day, y_day = Add_Window_Horizon(time_in_day, args.lag, args.horizon, single)
    x_week, y_week = Add_Window_Horizon(day_in_week, args.lag, args.horizon, single)
    x, y = np.concatenate([x, x_day, x_week], axis=-1), np.concatenate(
        [y, y_day, y_week], axis=-1
    )

    # spilit dataset by days or by ratio
    if args.test_ratio > 1:
        x_train, x_val, x_test = split_data_by_days(x, args.val_ratio, args.test_ratio)
        y_train, y_val, y_test = split_data_by_days(y, args.val_ratio, args.test_ratio)
    else:
        x_train, x_val, x_test = split_data_by_ratio(x, args.val_ratio, args.test_ratio)
        y_train, y_val, y_test = split_data_by_ratio(y, args.val_ratio, args.test_ratio)

    y_train = y_train[..., :1]
    y_val = y_val[..., :1]
    y_test = y_test[..., :1]

    # #normalize st data
    scaler = normalize_dataset(x_train[..., : args.input_dim], normalizer)

    x_train[..., : args.input_dim] = scaler.transform(x_train[..., : args.input_dim])
    x_val[..., : args.input_dim] = scaler.transform(x_val[..., : args.input_dim])
    x_test[..., : args.input_dim] = scaler.transform(x_test[..., : args.input_dim])

    # 5. 构建数据加载器
    train_dataloader = data_loader(x_train, y_train, args.batch_size, shuffle=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler


if __name__ == "__main__":
    import argparse
    import os

    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="PyTorch dataloader")
    parser.add_argument("--dataset", default="PEMS03", type=str)
    parser.add_argument("--num_nodes", default=358, type=int)
    parser.add_argument("--val_ratio", default=0.2, type=float)
    parser.add_argument("--test_ratio", default=0.2, type=float)
    parser.add_argument("--lag", default=12, type=int)
    parser.add_argument("--horizon", default=12, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--input_dim", default=1, type=int, help="输入数据维度")
    parser.add_argument("--steps_per_day", default=288, type=int, help="一天的时间步数")
    parser.add_argument("--days_per_week", default=7, type=int, help="一周的天数")
    args = parser.parse_args()
