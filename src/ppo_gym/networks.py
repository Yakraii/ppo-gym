"""Actor 和 Critic 网络。

    State ──┬──→ Actor  MLP ──→ Action logits ──→ Categorical 分布
            └──→ Critic MLP ──→ V(s)

两个网络的优化目标不同(策略 vs 状态价值),独立网络更容易理解、调试和展示。

网络结构沿用 CleanRL 的经验(cleanrl_ppo.py):
- 两层 64 维隐藏层 + Tanh 激活,宽度可通过 hidden_sizes 配置;
- 正交初始化:隐藏层 std=sqrt(2),Critic 输出层 std=1.0,
  Actor 输出层 std=0.01(让初始策略接近均匀随机,避免一开始就过度确定)。

重要约束:网络维度必须由环境提供的信息传入(state_dim / action_dim),
不写死 CartPole 的 4 和 2,这样同一套网络可以直接迁移到 LunarLander。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """对单个 Linear 层做正交初始化(CleanRL 经验)。

    隐藏层用 std=sqrt(2)(Tanh 激活的标准选择);
    输出层的 std 由调用方显式指定(Actor 0.01 / Critic 1.0)。
    """

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: tuple[int, ...],
    output_std: float = np.sqrt(2),
) -> nn.Sequential:
    """构建一个通用的多层感知机。

    结构:Linear(正交初始化) → Tanh → ... → Linear(不接激活)。
    最后一层不加激活:Actor 需要原始 logits,Critic 需要无界的状态价值。
    output_std 只作用于最后一层,见 layer_init 的说明。
    """

    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(layer_init(nn.Linear(last_dim, hidden_dim)))
        layers.append(nn.Tanh())
        last_dim = hidden_dim
    layers.append(layer_init(nn.Linear(last_dim, output_dim), std=output_std))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """策略网络:State → Action logits。

    输出 logits 后由调用方配合 Categorical 构造离散动作分布,
    网络本身不负责采样,采样逻辑放在 PPO.select_action() 中。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
    ) -> None:
        """初始化 Actor。

        参数:
            state_dim    状态维度,由 env.observation_space.shape[0] 提供。
            action_dim   动作维度,由 env.action_space.n 提供。
            hidden_sizes 隐藏层宽度序列,默认 (64, 64)。
        """

        super().__init__()
        # 输出层 std=0.01:初始 logits 几乎全 0,策略接近均匀随机。
        self.net = build_mlp(state_dim, action_dim, hidden_sizes, output_std=0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播:输入 (batch, state_dim),输出 (batch, action_dim) 的 logits。"""

        return self.net(state)


class Critic(nn.Module):
    """价值网络:State → V(s)。

    输出维度永远是 1,表示对该状态的期望累计回报估计。
    """

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
    ) -> None:
        """初始化 Critic。state_dim 由环境提供,与 Actor 保持一致。"""

        super().__init__()
        # 输出层 std=1.0:价值本身量级不确定,不做收缩。
        self.net = build_mlp(state_dim, 1, hidden_sizes, output_std=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播:输入 (batch, state_dim),输出 (batch, 1) 的状态价值。"""

        return self.net(state)
