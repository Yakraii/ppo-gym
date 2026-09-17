"""项目通用的小工具。

这里只放不依赖具体环境的辅助函数，避免 main.py 和 ppo.py 重复代码。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定随机种子，方便实验复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    """将 auto 解析为 cpu 或 cuda；也可直接返回用户指定的设备。"""

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def save_json(path: str | Path, data: Any) -> None:
    """保存 JSON，确保中文和目录不存在时也能正常写入。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def plot_training_curve(metrics: list[dict[str, Any]], path: str | Path) -> None:
    """绘制 episode return 曲线。

    这里延迟导入 matplotlib，因此不画图时不会影响训练。
    画图前可以运行：uv add matplotlib
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("未安装 matplotlib，请运行 `uv add matplotlib`") from exc

    if not metrics:
        return

    returns = [item["return"] for item in metrics]
    steps = [item["step"] for item in metrics]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(steps, returns, label="episode return")
    plt.xlabel("environment step")
    plt.ylabel("return")
    plt.title("PPO training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
