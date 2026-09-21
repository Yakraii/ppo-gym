"""测试过程的可视化。

支持两种观察方式:
1. render_mode="human":弹出窗口直接观看智能体行为;
2. render_mode="rgb_array":配合 Gymnasium 的 RecordVideo 包装器
   把运行过程保存成视频文件,写入 videos/<env>/。

最终展示推荐"训练曲线 + 测试运行视频"(定量 + 定性)。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym

from .utils import env_dir_name

if TYPE_CHECKING:
    from .ppo import PPO


def run_visualization(
    agent: "PPO",
    env_name: str,
    episodes: int = 3,
    render_mode: str = "human",
    save_video: bool = False,
    video_dir: str = "videos",
) -> None:
    """加载模型并可视化运行若干个 episode。

    参数:
        agent       已加载权重的 PPO Agent(来自 test.load_model)。
        env_name    环境名称,例如 "CartPole-v1"。
        episodes    可视化运行的回合数。
        render_mode "human" 弹窗实时观看;"rgb_array" 用于录制视频。
        save_video  True 时用 gym.wrappers.RecordVideo 录制,
                    视频保存到 videos/<env_name>/ 下。
        video_dir   视频根目录。

    可视化运行使用确定性动作,与 evaluate/test 保持一致;
    运行结束后 env.close() 让渲染窗口和视频文件正常收尾。
    """

    # 录视频必须用 rgb_array 渲染,human 弹窗模式无法同时录制。
    if save_video:
        render_mode = "rgb_array"

    env = gym.make(env_name, render_mode=render_mode)

    video_folder = None
    if save_video:
        video_folder = Path(video_dir) / env_dir_name(env_name)
        video_folder.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env, str(video_folder), episode_trigger=lambda idx: True
        )

    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

        print(f"episode reward: {total_reward:.2f}")

    env.close()
    if video_folder is not None:
        print(f"视频已保存到: {video_folder.resolve()}")


def main() -> None:
    """命令行入口:--checkpoint + --env + --render → run_visualization。"""

    parser = argparse.ArgumentParser(description="加载模型并可视化运行")
    parser.add_argument("--checkpoint", required=True, help="checkpoint 路径")
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium 环境 ID")
    parser.add_argument("--episodes", type=int, default=3, help="运行回合数")
    parser.add_argument("--render", default="human", choices=["human", "rgb_array"])
    parser.add_argument("--save-video", action="store_true", help="录制视频而不是弹窗")
    parser.add_argument("--device", default="auto", help="auto, cpu 或 cuda")
    args = parser.parse_args()

    # 复用 test.load_model 完成环境和模型构建。
    from .test import load_model

    agent = load_model(args.checkpoint, args.env, device=args.device)
    run_visualization(
        agent,
        env_name=args.env,
        episodes=args.episodes,
        render_mode=args.render,
        save_video=args.save_video,
    )


if __name__ == "__main__":
    main()
