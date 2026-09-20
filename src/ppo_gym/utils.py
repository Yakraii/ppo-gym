"""项目通用的小工具(Plan §25)。

只放不属于 PPO 核心算法的通用功能:
    随机种子 / 设备解析 / checkpoint 保存加载 / metrics CSV / 曲线绘制。

约束:不要把 PPO 核心算法塞进 utils.py。
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch(含 CUDA)的随机种子,保证实验可复现(Plan §31)。

    同时打开 cudnn 确定性模式(CleanRL 的 torch_deterministic 约定),
    代价是 GPU 训练可能略慢,课程实验规模下可以接受。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True


def resolve_device(device: str) -> torch.device:
    """将 "auto" 解析为 cuda(优先)或 cpu;其余值原样返回。"""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def env_dir_name(env_name: str) -> str:
    """把环境 ID 映射成小写目录名,用于 checkpoints/results/videos。

    例如 "CartPole-v1" → "cartpole","LunarLander-v3" → "lunarlander"。
    """

    return env_name.split("-")[0].lower()


# ----------------------------------------------------------------------
# checkpoint 保存与加载(Plan §20)
# ----------------------------------------------------------------------


def save_checkpoint(
    path: str,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int | None = None,
    config: Any = None,
) -> None:
    """保存模型权重到指定路径(独立函数版本,PPO.save 之外的备用入口)。

    至少保存 {"actor": ..., "critic": ...};
    断点继续训练时额外保存 optimizer / step / config。
    """

    payload: dict[str, Any] = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if step is not None:
        payload["global_step"] = step
    if config is not None:
        payload["config"] = config

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path_obj)


def load_checkpoint(
    path: str,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    map_location: str = "cpu",
) -> dict:
    """从 checkpoint 恢复 actor / critic 权重,返回剩余字段(optimizer 等)。

    map_location="cpu" 保证在无 GPU 机器上也能加载。
    """

    checkpoint = torch.load(path, map_location=map_location)
    actor.load_state_dict(checkpoint["actor"])
    critic.load_state_dict(checkpoint["critic"])

    extras = {key: value for key, value in checkpoint.items() if key not in ("actor", "critic")}
    return extras


# ----------------------------------------------------------------------
# metrics 记录(Plan §26)
# ----------------------------------------------------------------------


def save_metrics(path: str, records: list[dict[str, Any]]) -> None:
    """把训练/评估指标写入 CSV 文件(如 results/<env>/metrics.csv)。

    records 是形如 [{"episode": 1, "return": 12.0, ...}, ...] 的列表,
    以第一条记录的 key 作为表头,目录不存在时自动创建。
    """

    if not records:
        return

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(records[0].keys())
    with path_obj.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def load_metrics(path: str) -> list[dict[str, Any]]:
    """从 CSV 读取 metrics,数值字段自动转成 float,返回字典列表。"""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            records.append(parsed)
    return records


def moving_average(data: list[float], window: int) -> list[float]:
    """计算滑动平均,用于训练曲线去噪(Plan §27:原始 reward + 移动平均)。"""

    if window <= 1 or len(data) < window:
        return list(data)

    kernel = np.ones(window) / window
    smoothed = np.convolve(data, kernel, mode="valid")
    return smoothed.tolist()


# ----------------------------------------------------------------------
# JSON 与绘图(Plan §27/§28)
# ----------------------------------------------------------------------


def save_json(path: str, data: Any) -> None:
    """保存 JSON(ensure_ascii=False,自动创建父目录),用于配置和评估结果。"""

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def plot_training_curve(metrics: list[dict[str, Any]], path: str) -> None:
    """绘制 Reward vs Episode 训练曲线(Plan §27)。

    同时绘制原始 reward 和移动平均 reward,横轴用全局步数,
    保存为 PNG。
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not metrics:
        return

    returns = [item["return"] for item in metrics]
    steps = [item["step"] for item in metrics]
    window = max(1, min(20, len(returns) // 5)) if len(returns) >= 5 else 1

    plt.figure(figsize=(8, 4))
    plt.plot(steps, returns, alpha=0.4, label="episode return")
    if window > 1:
        smooth_steps = steps[window - 1 :]
        plt.plot(smooth_steps, moving_average(returns, window), label=f"moving avg ({window})")
    plt.xlabel("environment step")
    plt.ylabel("return")
    plt.title("PPO training curve")
    plt.legend()
    plt.tight_layout()

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_obj, dpi=150)
    plt.close()


def plot_evaluation_curve(metrics: list[dict[str, Any]], path: str) -> None:
    """绘制 Evaluation Reward vs Update 评估曲线(Plan §28)。

    相比单个 episode reward,更适合观察策略性能;
    同时画出 ±std 的包络,反映评估的波动。
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not metrics:
        return

    steps = [item["step"] for item in metrics]
    means = [item["mean_reward"] for item in metrics]
    stds = [item.get("std_reward", 0.0) for item in metrics]

    plt.figure(figsize=(8, 4))
    plt.plot(steps, means, marker="o", label="evaluation mean reward")
    plt.fill_between(
        steps,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.2,
        label="±1 std",
    )
    plt.xlabel("environment step")
    plt.ylabel("evaluation reward")
    plt.title("PPO evaluation curve")
    plt.legend()
    plt.tight_layout()

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_obj, dpi=150)
    plt.close()
