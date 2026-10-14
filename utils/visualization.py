from math import exp
import os

import dash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

# 使用内置的seaborn样式
plt.style.use("seaborn-v0_8")  # 或者使用 "seaborn-darkgrid"


def plot_loss_curve(train_losses, val_losses, test_losses=None, save_path=None):
    """绘制损失曲线，包括训练集、验证集和测试集"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.plot(epochs, val_losses, "g-", label="Validation Loss")
    if test_losses is not None:
        plt.plot(epochs, test_losses, "r-", label="Test Loss")

    plt.title("Training, Validation, and Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.close()


def plot_metrics_curve(rmses, maes, mapes, save_path=None):
    """绘制测试集上的评估指标变化曲线"""
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(rmses) + 1)

    # 创建三个子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # RMSE曲线
    ax1.plot(epochs, rmses, "r-", marker="o", markersize=3)
    ax1.set_title("Test RMSE over Epochs")
    ax1.set_ylabel("RMSE")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # MAE曲线
    ax2.plot(epochs, maes, "g-", marker="s", markersize=3)
    ax2.set_title("Test MAE over Epochs")
    ax2.set_ylabel("MAE")
    ax2.grid(True, linestyle="--", alpha=0.7)

    # MAPE曲线
    ax3.plot(epochs, mapes, "b-", marker="^", markersize=3)
    ax3.set_title("Test MAPE over Epochs")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("MAPE (%)")
    ax3.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.close()


def load_predictions(npz_file):
    """加载预测结果文件"""
    data = np.load(npz_file)
    predictions = data["predictions"]  # shape: (samples, horizon, num_nodes, features)
    truths = data["truths"]  # shape: (samples, horizon, num_nodes, features)
    # metrics = data["metrics"].item()  # 字典形式的评估指标
    return predictions, truths


def create_interactive_plot(predictions, truths, exp_dir, port=8050):
    """创建交互式预测结果可视化"""
    samples, horizon, num_nodes, features = predictions.shape

    # 初始化 Dash 应用
    app = dash.Dash(__name__)

    # 设置布局
    app.layout = html.Div(
        [
            html.H1("预测结果可视化", style={"textAlign": "center"}),
            # 控制面板
            html.Div(
                [
                    # 节点选择
                    html.Div(
                        [
                            html.Label("选择节点:", style={"marginRight": "10px"}),
                            dcc.Dropdown(
                                id="node-selector",
                                options=[
                                    {"label": f"节点 {i}", "value": i}
                                    for i in range(num_nodes)
                                ],
                                value=0,
                                style={"width": "200px"},
                            ),
                        ],
                        style={"marginBottom": "20px"},
                    ),
                ],
                style={
                    "padding": "20px",
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "5px",
                    "marginBottom": "20px",
                },
            ),
            # 图表
            dcc.Graph(id="prediction-plot"),
            # 评估指标显示
            html.Div(
                id="metrics-display",
                style={
                    "padding": "20px",
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "5px",
                },
            ),
        ],
        style={"maxWidth": "1200px", "margin": "auto", "padding": "20px"},
    )

    @app.callback(
        [Output("prediction-plot", "figure"), Output("metrics-display", "children")],
        [Input("node-selector", "value")],
    )
    def update_figure(selected_node):
        # 创建完整的时间序列
        true_values = []
        pred_values = []
        time_steps = []
        current_time = 0

        # 计算总的评估指标
        all_true = []
        all_pred = []

        for t in range(samples):
            # 对于非最后一个batch，只取第一个时间步（h=0）
            # 对于最后一个batch，取所有时间步
            if t < samples - 1:
                # 非最后一个batch，只取h=0
                h = 0
                if t + h < samples:
                    true_values.append(truths[t + h, 0, selected_node, 0])
                    pred_values.append(predictions[t, h, selected_node, 0])
                    time_steps.append(current_time)
                    current_time += 1

                    all_true.append(truths[t + h, 0, selected_node, 0])
                    all_pred.append(predictions[t, h, selected_node, 0])
            else:
                # 最后一个batch，取所有时间步
                for h in range(horizon):
                    true_values.append(truths[t, h, selected_node, 0])
                    pred_values.append(predictions[t, h, selected_node, 0])
                    time_steps.append(current_time)
                    current_time += 1

                    all_true.append(truths[t, h, selected_node, 0])
                    all_pred.append(predictions[t, h, selected_node, 0])

        # 创建图表
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=(f"节点 {selected_node} 的预测结果",),
        )

        # 添加真实值和预测值
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=true_values,
                name="真实值",
                line=dict(color="blue", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=pred_values,
                name="预测值",
                line=dict(color="red", width=2),
            )
        )

        # 更新布局
        fig.update_layout(
            height=600,
            showlegend=True,
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="x unified",
            xaxis_title="时间步",
            yaxis_title="值",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        # 更新坐标轴
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

        # 计算整体评估指标
        mae = np.mean(np.abs(np.array(all_true) - np.array(all_pred)))
        rmse = np.sqrt(np.mean((np.array(all_true) - np.array(all_pred)) ** 2))
        # 屏蔽真实值为0的数据点
        true_array = np.array(all_true)
        pred_array = np.array(all_pred)

        non_zero_mask = true_array != 0
        true_non_zero = true_array[non_zero_mask]
        pred_non_zero = pred_array[non_zero_mask]

        # 仅在存在非零真实值时计算MAPE
        if len(true_non_zero) > 0:
            mape = (
                np.mean(np.abs((true_non_zero - pred_non_zero) / true_non_zero)) * 100
            )
        else:
            mape = 0.0  # 如果所有真实值都为0，则MAPE为0

        # 创建评估指标显示
        metrics_display = html.Div(
            [
                html.H3("整体评估指标:", style={"marginBottom": "10px"}),
                html.Table(
                    [
                        html.Tr(
                            [
                                html.Td(
                                    "MAE:",
                                    style={
                                        "paddingRight": "10px",
                                        "fontWeight": "bold",
                                    },
                                ),
                                html.Td(f"{mae:.4f}"),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td(
                                    "RMSE:",
                                    style={
                                        "paddingRight": "10px",
                                        "fontWeight": "bold",
                                    },
                                ),
                                html.Td(f"{rmse:.4f}"),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td(
                                    "MAPE:",
                                    style={
                                        "paddingRight": "10px",
                                        "fontWeight": "bold",
                                    },
                                ),
                                html.Td(f"{mape:.2f}%"),
                            ]
                        ),
                    ],
                    style={"width": "100%"},
                ),
            ]
        )

        return fig, metrics_display

    try:
        print(f"启动可视化服务器在端口 {port}...")
        app.run(debug=True, port=port)
    except Exception as e:
        print(f"启动服务器时出错: {str(e)}")
        if "Address already in use" in str(e):
            print(f"端口 {port} 已被占用，请尝试使用其他端口")
            # 尝试使用其他端口
            for new_port in range(port + 1, port + 10):
                try:
                    print(f"尝试使用端口 {new_port}...")
                    app.run(debug=True, port=new_port)
                    break
                except Exception:
                    continue


def main():
    # 示例用法
    exp_dir = "experiments/PEMS04/20250720_065142"  # 根据实际情况修改
    predictions_file = os.path.join(exp_dir, "predictions.npz")

    # 加载预测结果
    predictions, truths = load_predictions(predictions_file)

    # 创建交互式可视化
    create_interactive_plot(predictions, truths, exp_dir)


if __name__ == "__main__":
    main()
