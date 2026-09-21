"""PPO 超参数的集中配置。
所有可调超参数都集中在这个 dataclass 中,
训练、评估、测试和实验脚本统一从这里读取配置。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PPOConfig:
    """PPO 的全部超参数模板。

    字段含义:
        env_name        环境名称,例如 "CartPole-v1" 或 "LunarLander-v3"。
        total_steps     训练总环境步数,训练循环的主结束条件。
        seed            随机种子,保证实验可复现。
        device          计算设备,"auto" / "cpu" / "cuda"。

        rollout_steps   一次 rollout 收集多少个时间步。
        update_epochs   同一批 rollout 数据重复训练多少轮。
        batch_size      mini-batch 大小,rollout 数据被切分训练。

        learning_rate   学习率,推荐基础值 3e-4。
        adam_eps        Adam 优化器的 eps(CleanRL 经验:1e-5 比
                        PyTorch 默认的 1e-8 在小网络上更稳定)。
        gamma           折扣因子,推荐 0.99。
        gae_lambda      GAE 的 lambda,控制偏差-方差权衡。
        clip_epsilon    PPO 裁剪系数 epsilon,推荐 0.2。

        value_coef      Critic loss 在总 loss 中的权重。
        entropy_coef    熵奖励系数,控制探索强度。
        max_grad_norm   梯度裁剪上限,提高训练稳定性。

        eval_interval   每隔多少次 PPO update 做一次周期评估。
        eval_episodes   每次周期评估运行的回合数。
    """

    env_name: str = "CartPole-v1"
    total_steps: int = 1_000_000
    seed: int = 42
    device: str = "auto"

    rollout_steps: int = 2048
    update_epochs: int = 10
    batch_size: int = 64

    learning_rate: float = 3e-4
    adam_eps: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2

    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    eval_interval: int = 10
    eval_episodes: int = 5
