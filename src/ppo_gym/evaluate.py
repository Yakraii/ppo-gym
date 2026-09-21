"""评估模块。

评估与训练严格分开:
- 不更新 Actor;
- 不更新 Critic;
- 不进行反向传播;
- 使用确定性动作(argmax)而非采样。

训练时每隔 eval_interval 次 PPO update 调用一次本模块,
评估结果用于 best model 判定和评估曲线。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    from .ppo import PPO


def evaluate(agent: "PPO", env: gym.Env, episodes: int) -> dict:
    """运行若干个完整 episode 并统计评估指标。
    评估时直接选择 logits 最大的动作，不进行随机采样
    流程:
        State → Actor → Action(确定性)→ Environment → Reward(不更新任何参数)

    返回:
        {
            "mean_reward":     平均回合奖励,
            "std_reward":      回合奖励标准差,
            "episode_rewards": 每个回合的奖励列表,
        }
    """

    episode_rewards: list[float] = []

    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

        episode_rewards.append(total_reward)

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "episode_rewards": episode_rewards,
    }
