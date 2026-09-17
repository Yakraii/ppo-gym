"""项目命令行入口。

职责：连接环境、PPO Agent 和输出目录，不在这里定义网络结构或 PPO Loss。
运行示例：
    uv run ppo-gym train --env CartPole-v1
    uv run ppo-gym eval --env CartPole-v1 --model outputs/CartPole-v1/latest.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np

from .ppo import PPOAgent, PPOConfig, RolloutBuffer
from .utils import plot_training_curve, resolve_device, save_json, set_seed


def make_env(env_id: str, seed: int):
    """创建并初始化 Gymnasium 环境。

    CartPole 和 LunarLander 都是离散动作环境，因此可以直接复用同一 Agent。
    如果以后接入连续动作环境，需要在这里增加动作分布类型。
    """

    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def build_config(args: argparse.Namespace) -> PPOConfig:
    """把命令行参数转换成 PPO 配置对象。"""

    return PPOConfig(
        env_id=args.env,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        learning_rate=args.lr,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_sizes=tuple(args.hidden_sizes),
        seed=args.seed,
        device=resolve_device(args.device),
    )


def train(args: argparse.Namespace) -> None:
    """训练一个 PPO Agent 并保存模型、配置和训练记录。"""

    set_seed(args.seed)
    env = make_env(args.env, args.seed)

    # 环境维度从 Gymnasium 自动读取，避免为每个环境写死数字。
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    config = build_config(args)
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config)

    run_dir = Path(args.output_dir) / args.env
    run_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0
    episode_index = 0
    global_step = 0
    metrics: list[dict[str, float | int]] = []

    print(f"环境: {args.env}")
    print(f"观测维度: {obs_dim}, 动作维度: {action_dim}")
    print(f"设备: {agent.device}")

    while global_step < args.total_steps:
        # 每个 rollout 使用独立 buffer，更新完成后自然丢弃旧数据。
        buffer = RolloutBuffer()

        for _ in range(args.rollout_steps):
            action, log_prob, value = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            buffer.add(
                obs=obs,
                action=action,
                log_prob=log_prob,
                reward=float(reward),
                value=value,
                terminated=terminated,
                truncated=truncated,
            )

            global_step += 1
            episode_return += float(reward)
            episode_length += 1
            obs = next_obs

            if terminated or truncated:
                episode_index += 1
                metrics.append(
                    {
                        "episode": episode_index,
                        "step": global_step,
                        "return": episode_return,
                        "length": episode_length,
                    }
                )
                print(
                    f"episode={episode_index:4d} "
                    f"step={global_step:7d} "
                    f"return={episode_return:8.2f}"
                )

                obs, _ = env.reset()
                episode_return = 0.0
                episode_length = 0

            if global_step >= args.total_steps:
                break

        # epoch 结束时用当前策略的最后状态 value 完成 GAE bootstrap。
        last_value = agent.state_value(obs)
        update_stats = agent.update(buffer, last_value)

        print(
            f"update step={global_step} "
            f"policy_loss={update_stats['policy_loss']:.4f} "
            f"value_loss={update_stats['value_loss']:.4f} "
            f"entropy={update_stats['entropy']:.4f}"
        )

    checkpoint_path = run_dir / "latest.pt"
    agent.save(checkpoint_path)
    save_json(run_dir / "config.json", asdict(config))
    save_json(run_dir / "metrics.json", metrics)

    # matplotlib 未安装时只给出提示，不阻断训练结果保存。
    try:
        plot_training_curve(metrics, run_dir / "training_curve.png")
    except RuntimeError as exc:
        print(f"[提示] {exc}")

    env.close()
    print(f"模型已保存到: {checkpoint_path.resolve()}")


def evaluate(args: argparse.Namespace) -> None:
    """加载模型并使用确定性动作进行测试。"""

    set_seed(args.seed)
    env = make_env(args.env, args.seed)
    agent = PPOAgent.load(args.model, device=resolve_device(args.device))

    returns: list[float] = []
    lengths: list[int] = []

    for episode in range(args.eval_episodes):
        # 每次评估使用不同但可复现的 seed。
        obs, _ = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
        episode_return = 0.0
        episode_length = 0

        while not (terminated or truncated):
            action, _, _ = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            episode_length += 1

        returns.append(episode_return)
        lengths.append(episode_length)

    result = {
        "env": args.env,
        "episodes": args.eval_episodes,
        "mean_return": sum(returns) / len(returns),
        "std_return": float(np.std(returns)),
        "mean_length": sum(lengths) / len(lengths),
        "returns": returns,
    }
    save_json(Path(args.model).with_name("evaluation.json"), result)

    env.close()
    print(result)


def build_parser() -> argparse.ArgumentParser:
    """定义最小命令行接口。"""

    parser = argparse.ArgumentParser(description="Minimal PyTorch PPO")
    parser.add_argument("command", choices=["train", "eval"], help="train 或 eval")
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium 环境 ID")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu 或 cuda")

    # 训练参数
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--output-dir", default="outputs")

    # 评估参数
    parser.add_argument("--model", default="", help="eval 时的 checkpoint 路径")
    parser.add_argument("--eval-episodes", type=int, default=20)

    return parser


def main() -> None:
    """程序入口。"""

    args = build_parser().parse_args()

    if args.command == "train":
        train(args)
    else:
        if not args.model:
            raise SystemExit("eval 模式必须提供 --model")
        evaluate(args)


if __name__ == "__main__":
    main()

