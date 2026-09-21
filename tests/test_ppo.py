"""最小可复用 PPO 模板的测试。

按照 Plan.md 的模块划分,优先覆盖最容易出错的部分:
    1. 网络输出维度(Agent);
    2. GAE / Return 计算(RolloutBuffer),含 terminated 与 truncated 的区分;
    3. 配置对象(config)。

运行:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from ppo_gym.config import PPOConfig
from ppo_gym.ppo import Agent, RolloutBuffer


class TestNetworks(unittest.TestCase):
    def test_actor_critic_output_shapes(self) -> None:
        """验证 Actor / Critic 输出维度,避免环境维度变化后出现张量形状错误。"""

        agent = Agent(state_dim=4, action_dim=2)
        obs = torch.zeros(3, 4)

        logits = agent.actor(obs)
        values = agent.critic(obs)

        self.assertEqual(tuple(logits.shape), (3, 2))
        self.assertEqual(tuple(values.shape), (3, 1))

    def test_actor_initial_logits_near_uniform(self) -> None:
        """正交初始化(std=0.01)后初始 logits 应接近全 0,即策略接近均匀。"""

        agent = Agent(state_dim=8, action_dim=4)
        logits = agent.actor(torch.zeros(1, 8))
        self.assertTrue(torch.all(logits.abs() < 0.5))


class TestRolloutBuffer(unittest.TestCase):
    @staticmethod
    def _make_buffer(
        rewards: tuple[float, ...],
        values: tuple[float, ...],
        terminated_flags: tuple[bool, ...],
        truncated_flags: tuple[bool, ...] | None = None,
        bootstrap_values: tuple[float, ...] | None = None,
    ) -> RolloutBuffer:
        """构造一个小型 buffer,减少各测试的重复代码。"""

        buffer = RolloutBuffer()
        truncated_flags = truncated_flags or (False,) * len(rewards)
        bootstrap_values = bootstrap_values or (0.0,) * len(rewards)

        for reward, value, terminated, truncated, bootstrap in zip(
            rewards, values, terminated_flags, truncated_flags, bootstrap_values
        ):
            buffer.add(
                state=np.array([0.0], dtype=np.float32),
                action=0,
                reward=reward,
                log_prob=0.0,
                value=value,
                terminated=terminated,
                truncated=truncated,
                bootstrap_value=bootstrap,
            )
        return buffer

    def test_compute_gae_for_terminal_steps(self) -> None:
        """终止轨迹:γ=λ=1、V=0 时,A_t 就等于从 t 开始的累计奖励。"""

        buffer = self._make_buffer(
            rewards=(1.0, 2.0),
            values=(0.0, 0.0),
            terminated_flags=(True, True),
        )
        buffer.compute_returns_and_advantages(
            last_value=torch.tensor(0.0),
            gamma=1.0,
            gae_lambda=1.0,
        )

        self.assertTrue(np.allclose(buffer.advantages, [1.0, 2.0]))
        self.assertTrue(np.allclose(buffer.returns, [1.0, 2.0]))

    def test_truncated_step_bootstraps_from_final_observation(self) -> None:
        """截断轨迹:必须用截断观测的价值 bootstrap,而不是当作价值为 0。

        场景:r=1, V(s)=2, 截断观测价值=5, γ=λ=1:
            δ = 1 + 5 - 2 = 4 → A=[4],Return=[6]。
        """

        buffer = self._make_buffer(
            rewards=(1.0,),
            values=(2.0,),
            terminated_flags=(False,),
            truncated_flags=(True,),
            bootstrap_values=(5.0,),
        )
        buffer.compute_returns_and_advantages(
            last_value=torch.tensor(0.0),
            gamma=1.0,
            gae_lambda=1.0,
        )

        self.assertTrue(np.allclose(buffer.advantages, [4.0]))
        self.assertTrue(np.allclose(buffer.returns, [6.0]))

    def test_advantage_does_not_leak_across_episodes(self) -> None:
        """λ 递推不能跨回合传递:前一步 done 后,A 不吸收后一回合的 δ。"""

        buffer = self._make_buffer(
            rewards=(1.0, 10.0),
            values=(0.0, 0.0),
            terminated_flags=(True, True),
        )
        buffer.compute_returns_and_advantages(
            last_value=torch.tensor(0.0),
            gamma=1.0,
            gae_lambda=1.0,
        )
        # 第一步的 advantage 仍是 1,没有被第二步的 10 污染。
        self.assertTrue(np.allclose(buffer.advantages, [1.0, 10.0]))

    def test_get_yields_correct_minibatch_shapes(self) -> None:
        """get() 应产出打乱后的 mini-batch,且批次总数覆盖全部数据。"""

        buffer = self._make_buffer(
            rewards=tuple(float(i) for i in range(10)),
            values=(0.0,) * 10,
            terminated_flags=(False,) * 10,
        )
        buffer.compute_returns_and_advantages(
            last_value=torch.tensor(0.0),
            gamma=0.99,
            gae_lambda=0.95,
        )

        batches = list(buffer.get(batch_size=4, device=torch.device("cpu")))
        total = sum(batch["states"].shape[0] for batch in batches)

        self.assertEqual(total, 10)
        self.assertEqual(batches[0]["states"].shape[1], 1)
        self.assertEqual(batches[0]["actions"].dtype, torch.long)


class TestConfig(unittest.TestCase):
    def test_default_hyperparameters(self) -> None:
        """验证默认超参数与 Plan.md §18 的推荐基础参数一致。"""

        config = PPOConfig()
        self.assertEqual(config.learning_rate, 3e-4)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.gae_lambda, 0.95)
        self.assertEqual(config.clip_epsilon, 0.2)
        self.assertEqual(config.update_epochs, 10)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.rollout_steps, 2048)
        self.assertEqual(config.value_coef, 0.5)
        self.assertEqual(config.entropy_coef, 0.01)
        self.assertEqual(config.max_grad_norm, 0.5)
        # CleanRL 经验值(Plan 之外的补充项)。
        self.assertEqual(config.adam_eps, 1e-5)


if __name__ == "__main__":
    unittest.main()
