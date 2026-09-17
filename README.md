# PPO Gym

使用 PyTorch 自己实现的最小 PPO 模板，目标是复用同一套代码完成：

- `CartPole-v1`
- `LunarLander-v3`

算法代码位于 `src/ppo_gym/`，训练入口为 `src/ppo_gym/main.py`。正式交付不使用 Stable-Baselines3 代替自研 PPO。

## 目录结构

```text
ppo-gym/
├── pyproject.toml
├── README.md
├── Task.md
├── Plan.md
├── .gitignore
├── tests/
│   └── test_ppo.py
├── outputs/                    # 训练产生，不提交 Git
└── src/
    └── ppo_gym/
        ├── __init__.py
        ├── main.py             # 命令行、环境创建、训练/评估调度
        ├── model.py            # Actor-Critic 网络
        ├── ppo.py              # Rollout、GAE、PPO 更新、保存加载
        └── utils.py            # seed、JSON、绘图等工具
```

## 快速开始

```powershell
# 按 pyproject.toml 安装依赖
uv sync

# 训练 CartPole
uv run ppo-gym train --env CartPole-v1 --total-steps 100000

# 评估模型
uv run ppo-gym eval --env CartPole-v1 --model outputs/CartPole-v1/latest.pt

# 运行最简测试
uv run python -m unittest discover -s tests -v
```

训练结果默认写入 `outputs/<env_id>/`：

```text
latest.pt
config.json
metrics.json
training_curve.png       # 需要 matplotlib 时才会生成
evaluation.json          # 运行 eval 后生成
```

## LunarLander 依赖提示

`LunarLander-v3` 依赖 Box2D。如果 `gymnasium.make("LunarLander-v3")` 报错，请安装：

```powershell
uv add "gymnasium[box2d]"
```

## 代码职责

- `main.py`：只负责连接环境、Agent 和输出目录，不定义网络和 PPO Loss。
- `model.py`：只负责 Actor-Critic 网络。
- `ppo.py`：只负责采样缓存、GAE、PPO 更新和模型保存加载。
- `utils.py`：只负责跨模块通用工具。
- `tests/test_ppo.py`：先覆盖最容易出错的网络维度和 GAE。

## 当前模板的扩展位置

- 超参数实验：修改 `PPOConfig` 或使用 `main.py` 的命令行参数。
- 连续动作环境：在 `model.py` 中增加高斯策略。
- 并行环境：在以后确实需要时再增加 `envs.py` 和向量化环境。
- 训练曲线：运行 `uv add matplotlib` 后重新训练，模板会自动生成图片。
