"""最小可复用 PPO 模板的烟雾测试。

运行：
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from ppo_gym.model import ActorCritic
from ppo_gym.ppo import RolloutBuffer


class TestModel(unittest.TestCase):
    def test_actor_critic_output_shapes(self) -> None:
        """验证网络输出维度，避免环境维度变化后出现张量形状错误。"""

        model = ActorCritic(obs_dim=4, action_dim=2)
        obs = torch.zeros(3, 4)
        logits, values = model(obs)

        self.assertEqual(tuple(logits.shape), (3, 2))
        self.assertEqual(tuple(values.shape), (3,))


class TestRolloutBuffer(unittest.TestCase):
    def test_compute_gae_for_terminal_steps(self) -> None:
        """验证最简单的终止轨迹 GAE 结果。"""

        buffer = RolloutBuffer()
        for reward in (1.0, 2.0):
            buffer.add(
                obs=np.array([0.0], dtype=np.float32),
                action=0,
                log_prob=0.0,
                reward=reward,
                value=0.0,
                terminated=True,
                truncated=False,
            )

        data = buffer.compute_gae(
            last_value=torch.tensor(0.0),
            gamma=1.0,
            gae_lambda=1.0,
            device=torch.device("cpu"),
        )

        expected = torch.tensor([1.0, 2.0])
        self.assertTrue(torch.allclose(data["advantages"], expected))


if __name__ == "__main__":
    unittest.main()
