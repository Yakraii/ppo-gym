"""PPO 使用的 Actor-Critic 网络。

这个文件只负责“网络结构”，不负责环境、采样、训练和保存。
因此同一套网络可以同时用于 CartPole 和 LunarLander，只要传入不同的
obs_dim 和 action_dim 即可。
"""

from __future__ import annotations

from torch import nn
from torch.distributions import Categorical


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: tuple[int, ...] = (64, 64),
) -> nn.Sequential:
    """构建一个最简单的多层感知机。"""

    layers: list[nn.Module] = []
    last_dim = input_dim

    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.Tanh())
        last_dim = hidden_dim

    # 最后一层不接激活函数：Actor 输出 logits，Critic 输出状态价值。
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """离散动作空间下的 Actor-Critic。

    输入形状：
        obs: (batch_size, obs_dim)

    输出形状：
        logits: (batch_size, action_dim)
        value:  (batch_size,)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()

        # 第一版使用两个独立 MLP，逻辑最直观。
        # 后续如果希望共享底层特征，可以把这里改成共享 backbone。
        self.actor = build_mlp(obs_dim, action_dim, hidden_sizes)
        self.critic = build_mlp(obs_dim, 1, hidden_sizes)

    def forward(self, obs):
        """返回动作 logits 和状态价值。"""

        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return logits, value

    def distribution(self, obs) -> Categorical:
        """根据观测构造离散动作分布。"""

        logits, _ = self.forward(obs)
        return Categorical(logits=logits)

    def evaluate_actions(self, obs, actions):
        """PPO 更新时使用：重新计算旧动作的 log_prob、熵和价值。"""

        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)

        return dist.log_prob(actions), dist.entropy(), value
