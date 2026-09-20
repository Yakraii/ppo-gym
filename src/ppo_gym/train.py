"""训练入口(Plan §21)。

只负责"启动训练"这一件事:创建环境 → 创建 PPO → 开始训练。
关键要求:

    > 除了环境名称和实验配置之外,不修改 PPO 算法。

运行方式:
    from ppo_gym.train import train
    train("CartPole-v1", PPOConfig())

命令行:
    uv run ppo-gym --env CartPole-v1
    uv run ppo-gym --env LunarLander-v3 --total-steps 500000
"""
from __future__ import annotations

import argparse

import gymnasium as gym

from .config import PPOConfig
from .ppo import PPO
from .utils import set_seed


def train(env_name: str, config: PPOConfig) -> PPO:
    """训练一个 PPO Agent。

    流程(Plan §36):
        env = gym.make(env_name)
        agent = PPO(env, config)
        agent.train()

    config.env_name 会被强制同步为 env_name,保证 checkpoint
    中记录的环境信息可用于后续 test / visualize。
    """

    config.env_name = env_name
    set_seed(config.seed)

    env = gym.make(env_name)
    env.reset(seed=config.seed)
    env.action_space.seed(config.seed)

    agent = PPO(env, config)
    agent.train()
    return agent


def build_parser() -> argparse.ArgumentParser:
    """定义命令行接口。

    暴露 PPOConfig 中的主要超参数,方便超参数实验时
    在命令行覆盖默认值(Plan §29~§30 的控制变量实验)。
    未暴露的字段(如 adam_eps)直接改 config.py 默认值。
    """

    parser = argparse.ArgumentParser(description="Minimal modular PPO (Plan.md 结构)")
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium 环境 ID")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu 或 cuda")

    # 训练预算与批次
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)

    # PPO 核心超参数
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64])

    # 周期评估
    parser.add_argument("--eval-interval", type=int, default=10, help="每隔多少次 update 评估一次")
    parser.add_argument("--eval-episodes", type=int, default=5, help="每次评估的回合数")

    return parser


def config_from_args(args: argparse.Namespace) -> PPOConfig:
    """把命令行参数转换成 PPOConfig;显式列出字段,避免静默遗漏。"""

    return PPOConfig(
        env_name=args.env,
        total_steps=args.total_steps,
        seed=args.seed,
        device=args.device,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_sizes=tuple(args.hidden_sizes),
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
    )


def main() -> None:
    """命令行入口:解析参数 → 构造 PPOConfig → 调用 train()。"""

    args = build_parser().parse_args()
    train(args.env, config_from_args(args))


if __name__ == "__main__":
    main()
