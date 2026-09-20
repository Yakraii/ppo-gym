"""PPO 项目的包入口。

按照 Plan.md §37 的模块划分,各文件职责:

    config.py      PPOConfig,集中管理超参数
    networks.py    Actor / Critic 网络
    buffer.py      RolloutBuffer,rollout 缓存与 GAE
    ppo.py         PPO 主类:采样、优势估计、更新、训练、保存加载
    train.py       训练入口
    evaluate.py    周期评估
    test.py        训练结束后的最终测试
    visualize.py   可视化与视频录制
    utils.py       种子、checkpoint、metrics、绘图等通用工具
"""

__version__ = "0.2.0"
