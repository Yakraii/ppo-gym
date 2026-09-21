# PPO Gym

用 PyTorch 实现的 PPO，同一套算法代码复用于：

- `CartPole-v1`
- `LunarLander-v3`

核心设计：网络维度从环境自动读取（`env.observation_space` / `env.action_space`），算法与具体环境解耦。

## 目录结构

```text
src/ppo_gym/
├── config.py      # PPOConfig,超参数集中管理
├── ppo.py         # Agent(Actor-Critic MLP)+ RolloutBuffer + PPO 核心:采样、GAE、更新、训练、保存加载
├── train.py       # 训练入口(命令行 ppo-gym)
├── evaluate.py    # 训练中周期评估
├── test.py        # 训练后最终测试
├── visualize.py   # 弹窗观看 / 录制视频
└── utils.py       # seed、checkpoint、metrics、绘图

experiments/       # 实验脚本
checkpoints/       # best.pth / latest.pth
results/           # metrics CSV、训练/评估曲线、config.json、测试报告
videos/            # 测试录制的视频
```

## 快速开始

```powershell
uv sync

# 训练 CartPole
uv run ppo-gym --env CartPole-v1 --total-steps 100000

# 训练 + 测试 LunarLander(约 100 万步)
uv run python experiments/lunarlander/run_lunarlander.py
#   --skip-train  跳过训练,只测试已有 best.pth

# 测试已训练模型
uv run python -m ppo_gym.test --checkpoint checkpoints/cartpole/best.pth --env CartPole-v1

# 可视化(弹窗观看 / --save-video 录制)
uv run python -m ppo_gym.visualize --checkpoint checkpoints/cartpole/best.pth --env CartPole-v1

```

## 超参数

集中在 `PPOConfig`(默认值即 Plan §18 基础配置:lr 3e-4、γ 0.99、λ 0.95、clip 0.2、rollout 2048、epochs 10、batch 64),可通过命令行覆盖,例如:

```powershell
uv run ppo-gym --env CartPole-v1 --lr 1e-3 --clip-epsilon 0.1
```

## 结果

每次运行的完整配置写入 `results/<env>/config.json`:

```text
results/<env>/
├── metrics.csv          # 每回合 return / length
├── update_metrics.csv   # 每次 update 的 loss / KL / clip_fraction
├── eval_metrics.csv     # 周期评估结果
├── curves.png           # 训练曲线(原始 + 移动平均)
├── eval_curve.png       # 评估曲线(±1 std)
├── diagnostics.png      # 诊断面板:KL / clip_fraction / entropy / value_loss
├── test_curve.png       # 测试曲线:每回合确定性策略 return(test.py 生成)
└── test_report.json     # 测试报告:mean/std/min/max + 逐回合 reward
```

## 参考

算法实现参考 [CleanRL](https://github.com/vwxyzjn/cleanrl) 的 `ppo.py`(正交初始化、Adam eps=1e-5、k3 估计 approx_kl、mini-batch advantage 标准化),实验方案见 `Plan.md`,课题要求见 `Task.md`。
