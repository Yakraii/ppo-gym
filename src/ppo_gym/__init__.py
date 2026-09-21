"""PPO 项目的包入口。

各文件职责:

    config.py      PPOConfig,集中管理超参数
    ppo.py         Agent 网络 + RolloutBuffer + PPO 主类:采样、GAE、更新、训练、保存加载
    train.py       训练入口
    evaluate.py    周期评估
    test.py        训练结束后的最终测试
    visualize.py   可视化与视频录制
    utils.py       种子、checkpoint、metrics、绘图等通用工具
"""

__version__ = "0.2.0"
