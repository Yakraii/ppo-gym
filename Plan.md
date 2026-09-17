# PPO 经典控制任务实施计划

> 依据：`Task.md`  
> 项目现状核对日期：2026-09-16

## 1. 任务理解

- 本课题的目标不是提出新的强化学习算法，而是独立完成一次完整的强化学习开发闭环：环境交互、数据采样、PPO 训练、模型保存与加载、评估、可视化和超参数实验。
- 必须使用 PyTorch 自己实现 PPO。Stable-Baselines3 不作为最终训练实现，后续最多只用于结果对照，不能成为主流程依赖。
- 使用 Gymnasium 的 `CartPole-v1` 和 `LunarLander-v3` 两个经典控制环境，二者都属于离散动作空间。
- 核心工程要求是“同一套 PPO 代码复用于两个环境”，切换环境时只允许改变配置、网络规模、环境参数和训练预算，不应复制算法代码或为某个环境单独重写 PPO。
- `CartPole-v1` 是第一道正确性门槛：环境简单、反馈快，优先用它验证采样、GAE、PPO Loss、更新和保存加载是否可靠。
- `LunarLander-v3` 是泛化与稳定性验证：状态、奖励和训练动态更复杂，需要更长训练时间，并检验同一算法实现能否通过配置迁移到新环境。

## 2. 当前项目状态

- 当前仓库只有最小 Python 包骨架，`src/ppo_gym/__init__.py` 还是占位入口。
- `pyproject.toml` 已声明 Gymnasium、PyTorch 和 Jupyter 等依赖，且配置了 CUDA 126 的 PyTorch 源。
- 现有虚拟环境为 Python 3.12.10，PyTorch 2.14.0+cu126 可用，`torch.cuda.is_available()` 为 `True`，Gymnasium 版本为 1.3.0。
- 当前目录尚未初始化为 Git 仓库。
- `.ipynb_checkpoints/train-checkpoint.ipynb` 中使用 Stable-Baselines3 训练 CartPole，仅能作为环境连通性示例；它不满足“自行实现 PPO”的核心要求，不能作为最终交付代码。
- 因而主要工作不是修补现有算法，而是从零建立可复用、可测试、可配置的 PyTorch PPO 工程。

## 3. 总体设计原则

- **算法与环境解耦**：PPO 只能依赖观测维度、动作维度、离散动作接口和设备配置，不依赖具体环境名称。
- **配置驱动**：环境名、训练步数、网络大小、学习率、GAE 参数、批量大小、随机种子和输出目录均通过配置传入。
- **先正确、后性能**：先完成小规模可运行和单元测试，再追求 CartPole/LunarLander 的收敛成绩。
- **可复现**：固定 Python、PyTorch、Gymnasium 和依赖版本；记录 seed、完整配置、代码版本和评估方式。
- **训练与评估分离**：训练时允许随机采样，评估时使用确定性动作，不更新模型，并使用独立评估回合报告指标。
- **保留证据**：每组实验保存配置、日志、模型、评估结果和曲线，避免只在 notebook 中保留无法复现的输出。

## 4. 最终采用的简化项目结构

```text
ppo-gym/
├── pyproject.toml
├── README.md
├── Task.md
├── Plan.md
├── .gitignore
├── tests/
│   └── test_ppo.py
├── outputs/                    # 训练产物；不提交大型文件
└── src/
    └── ppo_gym/
        ├── __init__.py
        ├── main.py             # 命令行、环境创建、训练/评估调度
        ├── model.py            # Actor-Critic 网络
        ├── ppo.py              # Rollout、GAE、PPO 更新、保存加载
        └── utils.py            # seed、设备、日志和绘图工具
```

第一版只保留四个核心代码文件：

- `main.py` 负责连接组件，不定义网络和 PPO Loss；
- `model.py` 负责 Actor-Critic，输入和输出维度由环境自动传入；
- `ppo.py` 负责算法、采样缓存、GAE 和保存加载；
- `utils.py` 负责跨模块通用工具。

暂时不创建 `configs/`、`envs.py`、`rollout.py`、`checkpoint.py`、`plotting.py`、`scripts/` 和复杂的嵌套目录。只有当对应文件确实变长、职责开始混杂时再拆分。
## 5. PPO 核心实现计划

### 5.1 网络

- 实现共享底层 MLP 的 Actor-Critic，或独立的 Actor/Critic MLP；第一版优先采用共享或清晰的独立实现，保证接口简单。
- Actor 输出每个离散动作的 logits，并通过 `Categorical` 构造动作分布。
- Critic 输出单个状态价值 `V(s)`。
- 网络输入和输出维度从环境空间自动推断，禁止硬编码 CartPole 的维度。
- 支持配置隐藏层宽度、层数、激活函数和初始化方式。

### 5.2 轨迹采样

- 与环境逐步交互，保存 `observation`、`action`、`reward`、`done`、`log_prob`、`value`、`terminated`、`truncated`。
- 严格区分 `terminated` 和 `truncated`：
  - `terminated=True` 表示环境自然终止，后续价值为 0；
  - `truncated=True` 表示时间截断，需要从最终观测 bootstrap 价值。
- 每次 rollout 结束后计算最后一个观测的 `V(s)`，为 GAE 和 Return 提供 bootstrap。
- 支持观测 `float32`、动作 `int64` 和正确设备迁移。

### 5.3 Advantage 与 Return

- 使用 GAE(lambda) 计算优势：
  - `delta_t = r_t + gamma * V(s_{t+1}) * (1 - terminated_t) - V(s_t)`
  - `A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}`
- Return 使用 `return_t = advantage_t + value_t`。
- 对 minibatch 内 Advantage 做标准化，降低更新尺度对训练稳定性的影响。
- 用独立单元测试校验递归方向、终止处理、截断处理和数值结果。

### 5.4 PPO 更新目标

- 保存旧策略的 log probability，计算概率比：
  - `ratio = exp(new_log_prob - old_log_prob)`
- 策略损失使用 clipped surrogate：
  - `L_policy = -min(ratio * A, clip(ratio, 1-epsilon, 1+epsilon) * A).mean()`
- Critic 使用 value loss，第一版可采用 MSE；是否启用 value clipping 作为配置项。
- 加入 entropy bonus 鼓励探索，并使用 `value_coef`、`entropy_coef` 控制各项权重。
- 使用梯度裁剪，避免 LunarLander 训练早期的异常梯度。
- 每个 rollout 做若干 epoch，再切分 minibatch；记录 approximate KL、clip fraction、policy loss、value loss 和 entropy。

### 5.5 训练控制与模型持久化

- 训练循环使用 `global_step` 驱动，按配置的总 timesteps 停止。
- 按固定间隔执行 evaluation，并保存最佳模型和最新模型。
- Checkpoint 至少包含：
  - 模型 `state_dict`
  - optimizer `state_dict`
  - 训练步数
  - 环境名
  - 观测/动作维度
  - 完整超参数配置
  - 随机种子
  - 代码和依赖版本信息
- 加载时校验环境维度和配置，支持从 checkpoint 继续训练或单独评估。
- 保存/加载后，在固定观测和固定 seed 下比较动作或 value，确保结果一致。

## 6. 分阶段执行步骤

### 阶段 0：工程环境与基线检查

- [ ] 初始化 Git 仓库，补充 `.gitignore`，排除 `.venv/`、checkpoint、运行日志和大体积模型。
- [ ] 检查 Gymnasium 环境创建、空间类型、reset/step 返回值和渲染后端。
- [ ] 安装并验证 LunarLander 所需的 Box2D 依赖；若导入失败，优先安装 `gymnasium[box2d]` 及对应构建依赖。
- [ ] 编写随机策略基线，记录两个环境的随机平均回报和回合长度。
- [ ] 清理或明确标注现有 SB3 notebook 的用途，避免与最终自研 PPO 主流程混淆。

### 阶段 1：最小可运行 PPO

- [ ] 实现配置、环境封装、Actor-Critic、rollout buffer 和 CLI 入口。
- [ ] 在 `CartPole-v1` 上跑通一次小规模训练，检查张量形状、动作范围和 Loss 是否为有限值。
- [ ] 实现 GAE、PPO clipped loss、entropy、value loss 和梯度裁剪。
- [ ] 添加单元测试：网络输出、GAE、PPO ratio/clip、保存加载。
- [ ] 在小规模运行中检查策略是否持续优于随机策略。

### 阶段 2：CartPole-v1 收敛

- [ ] 使用同一份 PPO 代码训练至稳定收敛。
- [ ] 定期执行确定性评估，报告 20 或 100 个 episode 的平均回报、标准差和回合长度。
- [ ] 成功标准：平均回报持续接近 500；不能只以偶然单次 500 分判断成功。
- [ ] 保存训练曲线、配置、日志和最终模型。
- [ ] 从保存的 checkpoint 重新加载并复现评估结果。
- [ ] 在 README 中记录可复现实验命令。

### 阶段 3：迁移到 LunarLander-v3

- [ ] 只新增环境配置和必要参数，不复制或重写 PPO 算法。
- [ ] 首先跑短程 smoke test，确认观测维度、动作维度、终止/截断语义和渲染正常。
- [ ] 根据环境规模调整网络宽度、rollout 长度和总训练步数。
- [ ] 诊断训练问题：若回报不升，依次检查 reward 尺度、GAE、终止 bootstrap、entropy、学习率和梯度范数。
- [ ] 目标参考：确定性评估平均回报达到 200 左右，并报告 episode 数量与方差；若未达到，完整记录失败实验和原因。
- [ ] 保存最佳模型、训练曲线、评估日志和加载测试结果。

### 阶段 4：超参数实验

- [ ] 先做单变量实验，不同时改变多个核心超参数。
- [ ] 建议实验维度：学习率、clip epsilon、rollout 长度、minibatch 大小、entropy coefficient、GAE lambda、网络宽度。
- [ ] 每组实验至少使用多个随机种子，区分性能差异与随机波动。
- [ ] 汇总结果为表格，并绘制均值曲线和方差带。
- [ ] 将 Stable-Baselines3 结果作为可选外部基线；只比较，不替换自研核心实现。
- [ ] 对关键结论给出解释，例如稳定性、样本效率、最终回报和训练时间之间的权衡。

### 阶段 5：测试、可视化与复现

- [ ] 提供独立的 `evaluate` 命令，加载 checkpoint 后运行确定性和随机策略评估。
- [ ] 支持 `human` 可视化和 `rgb_array` 视频录制；视频生成失败时不影响核心测试。
- [ ] 绘制 episode return、episode length、policy loss、value loss、entropy、approx KL 和 clip fraction。
- [ ] 固化环境、依赖、随机种子、配置和命令，保证同一配置能重复得到相近结果。
- [ ] README 包含项目结构、安装方式、训练命令、评估命令、结果和已知问题。
- [ ] 提交清晰的小步 Git commit，避免把大型运行产物直接提交到版本库。

## 7. 初始建议超参数

这些值是起点而不是结论，必须通过实验调整：

| 参数 | CartPole 起点 | LunarLander 起点 |
|---|---:|---:|
| gamma | 0.99 | 0.99 |
| lambda | 0.95 | 0.95 |
| clip epsilon | 0.2 | 0.2 |
| learning rate | 3e-4 | 2.5e-4 或 3e-4 |
| rollout steps | 2048 | 2048 |
| minibatch size | 64 | 64 |
| update epochs | 10 | 10 |
| entropy coefficient | 0.01 | 0.001 至 0.01 |
| value coefficient | 0.5 | 0.5 |
| max grad norm | 0.5 | 0.5 |
| hidden size | 64 x 64 | 256 x 256 或 128 x 128 |
| total timesteps | 200k 至 500k | 1M 至 3M |

## 8. 验收标准

- [ ] 相同的 PPO 实现仅通过配置即可运行 `CartPole-v1` 和 `LunarLander-v3`。
- [ ] PPO 训练、GAE、优势估计和裁剪目标均可解释，并通过基础单元测试。
- [ ] CartPole 能稳定训练到接近满分，而不是仅出现一次高分。
- [ ] LunarLander 能完成训练并给出明确、可复现的评估结果；若未达到目标回报，必须说明问题和实验证据。
- [ ] 模型可以保存、重新加载、继续训练或评估，结果具有一致性。
- [ ] 有训练曲线、评估日志、超参数实验表和可复现命令。
- [ ] Git 历史能体现项目从环境搭建、最小算法、CartPole 到 LunarLander 的演进。
- [ ] README 能让另一位同学在新环境中按步骤复现实验。

## 9. 主要风险与应对

- **LunarLander 依赖缺失**：环境创建可能因 Box2D 安装失败而报错，先解决依赖再调试算法。
- **GAE/终止处理错误**：错误 bootstrap 或不区分 terminated/truncated 会系统性污染 Advantage，优先用单元测试验证。
- **训练不稳定**：使用 Advantage 标准化、梯度裁剪、合理 entropy、较小学习率和定期评估。
- **GPU 反而更慢**：小型 MLP 在 GPU 上的数据传输可能得不偿失，设备应可配置，并比较 CPU/GPU。
- **只保存模型不保存配置**：checkpoint 必须同时保存配置和版本信息，否则无法可靠复现。
- **评估结果偶然性过高**：使用固定评估 seed、足够 episode 数和多次随机种子实验。
- **Notebook 与正式代码脱节**：正式实现以 `src/ppo_gym` 和 CLI 为准，notebook 只用于探索或展示。

## 10. 推荐执行顺序

1. 先建立 Git、配置、环境封装和测试骨架。
2. 用随机策略确认两个 Gymnasium 环境接口和依赖都正常。
3. 实现并验证 Actor-Critic、rollout、GAE 和 PPO clipped loss。
4. 先在 CartPole 小规模跑通，再调到稳定收敛。
5. 固化保存加载、评估和绘图流程。
6. 仅通过配置迁移到 LunarLander，并进行长期训练和问题诊断。
7. 完成单变量超参数实验、结果汇总和 README。
8. 最后进行从零复现测试，确认交付物完整。

