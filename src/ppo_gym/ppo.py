"""PPO 的网络、采样缓存和训练更新。

职责划分：
- PPOConfig：保存可复用超参数。
- RolloutBuffer：保存一次 rollout 并计算 GAE/Returns。
- PPOAgent：选择动作、执行 PPO 更新、保存和加载模型。

这个文件不知道具体环境是 CartPole 还是 LunarLander，只接收 obs_dim、
action_dim 和环境产生的数据，因此算法可以复用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from .model import ActorCritic


@dataclass
class PPOConfig:
    """PPO 超参数模板；实验时可以直接修改这里的默认值。"""

    env_id: str = "CartPole-v1"
    total_steps: int = 100_000
    rollout_steps: int = 2048
    update_epochs: int = 10
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (64, 64)
    seed: int = 42
    device: str = "auto"


class RolloutBuffer:
    """保存一条 rollout，并在更新前计算 advantage 和 return。"""

    def __init__(self) -> None:
        self.observations: list[Any] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.terminated: list[float] = []
        self.truncated: list[float] = []

    def add(
        self,
        obs,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """向当前 rollout 添加一次环境交互。"""

        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.terminated.append(float(terminated))
        self.truncated.append(float(truncated))

    def __len__(self) -> int:
        return len(self.rewards)

    def _as_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        """把 Python 列表转换成训练所需张量。"""

        return {
            "observations": torch.as_tensor(
                np.asarray(self.observations), dtype=torch.float32, device=device
            ),
            "actions": torch.as_tensor(self.actions, dtype=torch.long, device=device),
            "log_probs": torch.as_tensor(
                self.log_probs, dtype=torch.float32, device=device
            ),
            "rewards": torch.as_tensor(self.rewards, dtype=torch.float32, device=device),
            "values": torch.as_tensor(self.values, dtype=torch.float32, device=device),
            "terminated": torch.as_tensor(
                self.terminated, dtype=torch.float32, device=device
            ),
            "truncated": torch.as_tensor(
                self.truncated, dtype=torch.float32, device=device
            ),
        }

    def compute_gae(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """使用 GAE(lambda) 计算 advantage 和 return。

        注意：终止和截断必须分开处理。
        - terminated=True：下一状态没有价值；
        - truncated=True：只是时间上限，下一状态仍然可以 bootstrap。
        """

        data = self._as_tensors(device)
        rewards = data["rewards"]
        values = data["values"]
        terminated = data["terminated"]
        done = torch.clamp(data["terminated"] + data["truncated"], max=1.0)

        advantages = torch.zeros_like(rewards)
        last_gae = torch.tensor(0.0, device=device)

        for step in reversed(range(len(rewards))):
            next_value = last_value if step == len(rewards) - 1 else values[step + 1]
            next_non_terminal = 1.0 - terminated[step]
            delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
            last_gae = delta + gamma * gae_lambda * (1.0 - done[step]) * last_gae
            advantages[step] = last_gae

        data["advantages"] = advantages
        data["returns"] = advantages + values
        return data


class PPOAgent:
    """最小 PPO Agent，可复用于所有离散动作环境。"""

    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = self._resolve_device(config.device)

        self.model = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=config.hidden_sizes,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """把 auto 转换成实际设备，其余情况保持用户指定值。"""

        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def act(self, obs, deterministic: bool = False) -> tuple[int, float, float]:
        """根据单个观测选择动作，返回 action、log_prob 和 state value。"""

        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.model(obs_tensor)
            dist = Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def state_value(self, obs) -> torch.Tensor:
        """计算单个状态的价值，供 GAE 末端 bootstrap 使用。"""

        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            _, value = self.model(obs_tensor)

        return value.squeeze(0)

    def update(self, buffer: RolloutBuffer, last_value: torch.Tensor) -> dict[str, float]:
        """执行若干个 epoch 的 PPO 更新。"""

        data = buffer.compute_gae(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            device=self.device,
        )

        observations = data["observations"]
        actions = data["actions"]
        old_log_probs = data["log_probs"]
        returns = data["returns"]
        advantages = data["advantages"]

        # 标准化 advantage 可减少不同 rollout 之间的尺度差异。
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = observations.shape[0]
        minibatch_size = min(self.config.minibatch_size, batch_size)

        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        update_count = 0

        for _ in range(self.config.update_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                batch_indices = indices[start : start + minibatch_size]

                new_log_probs, entropy, values = self.model.evaluate_actions(
                    observations[batch_indices],
                    actions[batch_indices],
                )

                ratio = torch.exp(new_log_probs - old_log_probs[batch_indices])
                unclipped = ratio * advantages[batch_indices]
                clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_ratio,
                        1.0 + self.config.clip_ratio,
                    )
                    * advantages[batch_indices]
                )

                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, returns[batch_indices])
                entropy_mean = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_mean
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    stats["policy_loss"] += float(policy_loss.item())
                    stats["value_loss"] += float(value_loss.item())
                    stats["entropy"] += float(entropy_mean.item())
                    stats["approx_kl"] += float(
                        (old_log_probs[batch_indices] - new_log_probs).mean().item()
                    )
                    stats["clip_fraction"] += float(
                        ((ratio - 1.0).abs() > self.config.clip_ratio)
                        .float()
                        .mean()
                        .item()
                    )
                update_count += 1

        # 返回平均值，便于 main.py 保存日志和绘制曲线。
        return {name: value / max(update_count, 1) for name, value in stats.items()}

    def save(self, path: str | Path) -> None:
        """保存模型和恢复所需的最小信息。"""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": asdict(self.config),
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "PPOAgent":
        """加载模型；device 可以覆盖 checkpoint 中的设备配置。"""

        checkpoint = torch.load(path, map_location="cpu")
        config_data = checkpoint["config"]
        config = PPOConfig(**config_data)

        if device is not None:
            config.device = device

        agent = cls(
            obs_dim=int(checkpoint["obs_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            config=config,
        )
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        agent.model.eval()
        return agent
