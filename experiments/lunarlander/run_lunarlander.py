"""LunarLander-v3 基线实验:一次跑完"训练 + 测试"。

用法(在项目根目录执行):
    uv run python experiments/lunarlander/run_lunarlander.py              # 完整训练(约 50 万步)+ 测试
    uv run python experiments/lunarlander/run_lunarlander.py --smoke     # 冒烟验证(1024 步)
    uv run python experiments/lunarlander/run_lunarlander.py --skip-train  # 不训练,只测试已有模型

复用原则(Plan §23/§39):
    不重写 PPO。CartPole 上验证过的同一套算法代码,
    这里只更换环境名和训练预算,其余超参数保持 Plan §18 的基础配置,
    以此检验 PPO 实现的迁移能力。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ppo_gym.config import PPOConfig
from ppo_gym.test import load_model, test
from ppo_gym.train import train
from ppo_gym.utils import env_dir_name, save_json

ENV_NAME = "LunarLander-v3"
BEST_CHECKPOINT = Path("checkpoints") / env_dir_name(ENV_NAME) / "best.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description="LunarLander-v3 训练 + 测试")
    parser.add_argument("--smoke", action="store_true", help="只跑 1024 步,验证流程能走通")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练,直接测试已有 best.pth")
    parser.add_argument("--test-episodes", type=int, default=20, help="测试回合数")
    args = parser.parse_args()

    config = PPOConfig(
        env_name=ENV_NAME,
        # LunarLander 比 CartPole 难:状态 8 维、奖励结构复杂、训练更长(Plan §四),
        # 因此训练预算从 CartPole 的 10 万步提高到 50 万步。
        # 其余超参数刻意不动——这正是"同一套 PPO,只换环境"的验证点。
        total_steps=500_000,
        seed=42,
    )
    if args.smoke:
        config.total_steps = 1024
        config.rollout_steps = 512
        config.eval_interval = 2
        config.eval_episodes = 2

    # ---------------- 第 1 步:训练 ----------------
    if not args.skip_train:
        train(ENV_NAME, config)

    # ---------------- 第 2 步:测试 ----------------
    # 加载评估最优的 best.pth,用确定性动作跑 N 个回合。
    agent = load_model(str(BEST_CHECKPOINT), ENV_NAME)
    result = test(agent, agent.env, args.test_episodes)
    agent.env.close()

    print(f"\n{ENV_NAME} 测试结果 ({args.test_episodes} 回合):")
    print(
        f"  mean_reward = {result['mean_reward']:.2f}  "
        f"std = {result['std_reward']:.2f}  "
        f"min = {result['min_reward']:.2f}  "
        f"max = {result['max_reward']:.2f}"
    )
    print("  参考:平均 200 以上视为过关(LunarLander 官方 solved 标准)")

    report_path = Path("results") / env_dir_name(ENV_NAME) / "test_report.json"
    save_json(report_path, {"checkpoint": str(BEST_CHECKPOINT), **result})
    print(f"  测试报告: {report_path.resolve()}")


if __name__ == "__main__":
    main()
