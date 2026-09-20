# PPO Gym 实验 Plan

## 1. 实验目标

使用**同一套 PPO 算法代码**完成两个 Gymnasium 经典环境：

1.  `CartPole-v1`
2.  `LunarLander-v3`

核心要求：

-   PPO 算法与具体环境解耦。
-   不为两个环境分别实现 PPO。
-   PPO 自动从 `env` 获取状态维度和动作维度。
-   Actor 与 Critic 使用**独立网络**。
-   完成训练、评估、模型保存、训练曲线、超参数实验、模型测试和可视化。
-   保持**普通实验级别**，不引入过度工程化的设计。

------------------------------------------------------------------------

# 2. 总体设计思想

整个实验采用：

``` text
                 Gymnasium Environment
                         │
                         │ env
                         ↓
                    PPO(env)
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ↓                             ↓
       Actor                         Critic
          │                             │
          ↓                             ↓
Action Distribution                  V(s)
          │
          ↓
       Action
          │
          ↓
      Environment
          │
          ↓
       Reward
          │
          └──────────────┐
                         ↓
                 collect_rollout()
                         │
                         ↓
                  Rollout Buffer
                         │
                         ↓
              Return / Advantage
                         │
                         ↓
                    PPO Update
                         │
                 ┌───────┴───────┐
                 ↓               ↓
               Actor           Critic
                 │               │
                 └───────┬───────┘
                         ↓
                    下一轮训练
```

最重要的复用关系：

``` text
                    PPO
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
    CartPole-v1          LunarLander-v3
          │                     │
    state_dim=4            state_dim=8
    action_dim=2           action_dim=4
          │                     │
          └──────────┬──────────┘
                     ↓
              同一套 PPO 算法
```

> 这里的维度由 Gymnasium 环境实际提供的信息自动读取，不在 PPO 中写死。

------------------------------------------------------------------------

# 3. 为什么采用独立 Actor-Critic

本实验采用：

``` text
                 State
                /     \
               ↓       ↓
           Actor网络  Critic网络
               ↓       ↓
      Action Distribution  V(s)
               ↓
             Action
```

即：

``` python
actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)
```

而不是共享 Backbone。

原因：

1.  Actor 的目标是学习策略 `π(a|s)`。
2.  Critic 的目标是学习状态价值 `V(s)`。
3.  两者优化目标不同。
4.  独立网络更容易理解、调试和展示。
5.  CartPole 和 LunarLander
    的网络规模都较小，没有必要为了节省参数而强制共享 Backbone。

注意：

> 本实验并不声称独立网络在所有 PPO
> 任务中都一定比共享网络效果好。这里只是为了实验结构清晰而采用独立网络。

------------------------------------------------------------------------

# 4. 推荐项目结构

``` text
ppo-gym/
│
├── pyproject.toml
├── README.md
├── Plan.md
│
├── src/
│   └── ppo_gym/
│       ├── __init__.py
│       │
│       ├── networks.py
│       │
│       ├── buffer.py
│       │
│       ├── ppo.py
│       │
│       ├── config.py
│       │
│       ├── train.py
│       │
│       ├── evaluate.py
│       │
│       ├── test.py
│       │
│       ├── visualize.py
│       │
│       └── utils.py
│
├── experiments/
│   ├── cartpole/
│   └── lunarlander/
│
├── checkpoints/
│   ├── cartpole/
│   └── lunarlander/
│
├── results/
│   ├── cartpole/
│   │   ├── metrics.csv
│   │   └── curves.png
│   │
│   └── lunarlander/
│       ├── metrics.csv
│       └── curves.png
│
└── videos/
```

说明：

-   `src/ppo_gym/`：核心代码。
-   `experiments/`：不同实验的运行配置或实验记录。
-   `checkpoints/`：模型权重。
-   `results/`：训练数据和曲线。
-   `videos/`：测试时保存的视频。
-   `Plan.md`：实验方案。
-   `README.md`：最终使用说明。

不要求把每个功能拆成很多文件；上述结构已经足够支持完整实验，同时不会过度工程化。

------------------------------------------------------------------------

# 5. 各文件职责

## 5.1 `networks.py`

负责 Actor 和 Critic。

主要内容：

``` text
Actor
Critic
```

### Actor

输入：

``` text
State
```

输出：

``` text
Action logits
```

结构：

``` text
State
  ↓
Linear
  ↓
Activation
  ↓
Linear
  ↓
Action logits
```

例如：

``` python
Actor(state_dim, action_dim)
```

其中：

-   `state_dim`：由环境决定。
-   `action_dim`：由环境决定。

不要写死：

``` python
nn.Linear(4, ...)
nn.Linear(..., 2)
```

而应该：

``` python
nn.Linear(state_dim, ...)
nn.Linear(..., action_dim)
```

------------------------------------------------------------------------

### Critic

输入：

``` text
State
```

输出：

``` text
V(s)
```

结构：

``` text
State
  ↓
Linear
  ↓
Activation
  ↓
Linear
  ↓
1
```

接口：

``` python
Critic(state_dim)
```

输出维度永远是：

``` text
1
```

------------------------------------------------------------------------

# 6. `buffer.py`

负责保存一次 rollout 收集的数据。

建议保存：

``` text
states
actions
rewards
dones
log_probs
values
```

必要时还可以保存：

``` text
next_states
```

但如果通过下一状态或 bootstrap value 计算
GAE，也可以根据实现选择不显式保存全部 `next_states`。

主要类：

``` python
RolloutBuffer
```

主要方法：

``` python
add(...)
compute_returns_and_advantages(...)
get(...)
clear(...)
```

Buffer 的作用：

``` text
Environment
     ↓
collect_rollout()
     ↓
RolloutBuffer
     ↓
Advantage / Return
     ↓
PPO Update
```

------------------------------------------------------------------------

# 7. `ppo.py`

这是整个实验的核心。

主要类：

``` python
PPO
```

初始化：

``` python
PPO(env, config)
```

初始化时自动获取：

``` python
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
```

然后创建：

``` python
Actor(state_dim, action_dim)
Critic(state_dim)
```

因此 PPO 不需要知道自己是在 CartPole 还是 LunarLander 中运行。

------------------------------------------------------------------------

# 8. PPO 的核心方法

## 8.1 `select_action()`

作用：

> 根据当前 Actor 对一个状态选择动作。

流程：

``` text
State
  ↓
Actor
  ↓
Action logits
  ↓
Categorical Distribution
  ↓
Action
```

同时记录：

``` text
action
log_prob
value
```

因为 PPO 后面需要旧策略的 `log_prob` 和 Critic 的 `V(s)`。

------------------------------------------------------------------------

## 8.2 `collect_rollout()`

### 定义

`collect_rollout()` 是 PPO 的**数据采集阶段**。

它不是 PPO 的独立算法，而是负责：

> 使用当前策略与环境交互，收集一批训练数据。

核心流程：

``` text
当前 Actor
    ↓
读取 State
    ↓
选择 Action
    ↓
env.step(Action)
    ↓
得到：
next_state
reward
done
    ↓
保存数据
    ↓
继续交互
```

收集：

``` text
state
action
reward
done
log_prob
value
```

伪代码：

``` python
def collect_rollout(self):
    buffer.clear()

    for _ in range(rollout_steps):
        action, log_prob, value = self.select_action(state)

        next_state, reward, terminated, truncated, _ = \
            env.step(action)

        done = terminated or truncated

        buffer.add(
            state,
            action,
            reward,
            done,
            log_prob,
            value
        )

        state = next_state

        if done:
            state, _ = env.reset()
```

------------------------------------------------------------------------

# 9. 为什么 PPO 需要 Rollout

PPO 不是：

``` text
state
 ↓
action
 ↓
马上更新
```

而是：

``` text
状态
 ↓
动作
 ↓
奖励
 ↓
状态
 ↓
动作
 ↓
奖励
 ↓
...
 ↓
收集一批数据
 ↓
计算 Advantage
 ↓
PPO Update
```

所以 PPO 的训练循环可以概括为：

``` text
Collect
  ↓
Estimate
  ↓
Update
  ↓
Collect
  ↓
Estimate
  ↓
Update
  ↓
...
```

------------------------------------------------------------------------

# 10. `compute_advantage()`

PPO 使用 Advantage 判断：

> 当前采取的动作，比 Critic 原本对这个状态的预期好多少？

核心：

``` text
Advantage
=
实际表现
-
Critic预测
```

本实验推荐使用 GAE：

``` text
GAE(Generalized Advantage Estimation)
```

核心公式：

``` text
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
```

然后：

``` text
A_t = δ_t + γλδ_{t+1} + (γλ)^2δ_{t+2} + ...
```

其中：

-   `γ`：discount factor。
-   `λ`：GAE 参数。
-   `V(s)`：Critic 对状态价值的估计。

------------------------------------------------------------------------

# 11. Return

Critic 的训练目标需要 Return：

``` text
G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
```

可以理解为：

``` text
Return
=
从当前状态开始，未来累计折扣奖励
```

最终：

``` text
Critic:
V(s) → Return
```

------------------------------------------------------------------------

# 12. PPO Update

这是 PPO 真正进行参数更新的地方。

对于 Actor：

``` text
旧策略概率
      ↓
ratio
      ↓
clipped objective
      ↓
Actor loss
      ↓
反向传播
      ↓
Actor 更新
```

核心概率比：

``` text
r_t(θ)
=
π_θ(a_t|s_t)
/
π_old(a_t|s_t)
```

PPO 的核心思想：

``` text
不要让新策略相对于旧策略变化太大
```

因此使用：

``` text
clip(r_t, 1-ε, 1+ε)
```

Actor 的 PPO clipped objective：

``` text
L^CLIP
=
E[
min(
r_t A_t,
clip(r_t, 1-ε, 1+ε) A_t
)
]
```

实现时通常使用最小化 loss，因此：

``` text
actor_loss = -mean(
    min(
        ratio * advantage,
        clipped_ratio * advantage
    )
)
```

------------------------------------------------------------------------

# 13. Critic Update

Critic：

``` text
V(s)
```

目标：

``` text
Return
```

通常：

``` text
critic_loss
=
MSE(V(s), Return)
```

所以：

``` text
Actor:
State → Action Distribution
              ↓
         PPO Objective

Critic:
State → V(s)
          ↓
       Return
```

------------------------------------------------------------------------

# 14. Entropy

Actor 可以加入 entropy bonus：

``` text
Entropy
```

作用：

> 鼓励策略保持一定探索能力，避免过早变得过于确定。

总 loss 可以写成：

``` text
loss =
actor_loss
+
value_coef * critic_loss
-
entropy_coef * entropy
```

------------------------------------------------------------------------

# 15. `train()`

训练流程：

``` text
reset environment
       ↓
collect_rollout()
       ↓
compute advantages / returns
       ↓
PPO update
       ↓
记录 metrics
       ↓
evaluate
       ↓
保存 best model
       ↓
重复
```

建议训练以：

``` text
rollout
```

或者：

``` text
update
```

作为主要训练单位，而不是强制以单 episode 作为 PPO 更新单位。

------------------------------------------------------------------------

# 16. `config.py`

集中保存超参数。

推荐：

``` python
@dataclass
class PPOConfig:
    learning_rate: float = 3e-4

    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_epsilon: float = 0.2

    update_epochs: int = 10
    batch_size: int = 64
    rollout_steps: int = 2048

    value_coef: float = 0.5
    entropy_coef: float = 0.01

    max_grad_norm: float = 0.5
```

------------------------------------------------------------------------

# 17. 超参数说明

## 17.1 Learning Rate

``` text
learning_rate
```

控制网络参数每次更新的幅度。

推荐基础值：

``` text
3e-4
```

实验可以比较：

``` text
1e-4
3e-4
1e-3
```

------------------------------------------------------------------------

## 17.2 Gamma

``` text
gamma = 0.99
```

决定未来奖励的重要程度。

越接近 1：

``` text
更重视长期奖励
```

越小：

``` text
更重视近期奖励
```

建议基础实验：

``` text
0.99
```

如果做参数实验：

``` text
0.95
0.99
```

------------------------------------------------------------------------

## 17.3 GAE Lambda

``` text
gae_lambda = 0.95
```

控制 Advantage 估计中的偏差-方差权衡。

建议：

``` text
0.90
0.95
0.99
```

------------------------------------------------------------------------

## 17.4 Clip Epsilon

``` text
clip_epsilon = 0.2
```

决定 PPO 对策略更新幅度的限制。

常用基础值：

``` text
0.2
```

实验：

``` text
0.1
0.2
0.3
```

------------------------------------------------------------------------

## 17.5 Rollout Steps

``` text
rollout_steps = 2048
```

表示一次收集多少个时间步的数据。

较大：

``` text
数据更多
更新更稳定
但单次更新等待更久
```

较小：

``` text
更新更频繁
但数据可能更加噪声化
```

对于 CartPole 可以使用较小值进行快速实验；最终统一实验时再设置固定值。

------------------------------------------------------------------------

## 17.6 Update Epochs

``` text
update_epochs = 10
```

表示同一批 rollout 数据被重复训练多少轮。

建议基础值：

``` text
10
```

------------------------------------------------------------------------

## 17.7 Batch Size

``` text
batch_size = 64
```

将 rollout 数据划分成 mini-batch。

------------------------------------------------------------------------

## 17.8 Value Coefficient

``` text
value_coef = 0.5
```

控制 Critic loss 对总 loss 的影响。

------------------------------------------------------------------------

## 17.9 Entropy Coefficient

``` text
entropy_coef = 0.01
```

控制探索奖励。

如果过高：

``` text
策略可能长期保持随机
```

如果过低：

``` text
可能过早失去探索
```

------------------------------------------------------------------------

## 17.10 Max Grad Norm

``` text
max_grad_norm = 0.5
```

用于梯度裁剪，提高训练稳定性。

------------------------------------------------------------------------

# 18. 推荐的基础参数

第一阶段不要调参，先使用一组固定参数验证 PPO：

``` text
learning_rate = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
update_epochs = 10
batch_size = 64
rollout_steps = 2048
value_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5
```

注意：

> 这是一套实验起始配置，不应在没有实际运行结果的情况下声称它对两个环境都是最优配置。

------------------------------------------------------------------------

# 19. `evaluate.py`

评估与训练分开。

核心方法：

``` python
evaluate(agent, env, episodes)
```

评估阶段：

``` text
不更新 Actor
不更新 Critic
不进行反向传播
```

只执行：

``` text
State
 ↓
Actor
 ↓
Action
 ↓
Environment
 ↓
Reward
```

最终计算：

``` text
mean reward
std reward
episode rewards
```

建议每隔若干次 PPO update 进行一次评估。

------------------------------------------------------------------------

# 20. 模型保存

保存：

``` text
checkpoints/
├── cartpole/
│   └── best.pth
│
└── lunarlander/
    └── best.pth
```

至少保存：

``` python
{
    "actor": actor.state_dict(),
    "critic": critic.state_dict()
}
```

如果需要断点继续训练，可以额外保存：

``` text
actor optimizer
critic optimizer
training step
config
```

实验阶段建议保存：

``` text
best model
latest model
```

其中：

-   `best model`：评估 reward 最好的模型。
-   `latest model`：最近一次训练状态。

------------------------------------------------------------------------

# 21. `train.py`

负责启动训练。

推荐接口：

``` python
def train(env_name, config):
    env = gym.make(env_name)

    agent = PPO(env, config)

    ...
```

运行：

``` python
train("CartPole-v1", config)
```

或者：

``` python
train("LunarLander-v3", config)
```

关键要求：

> 除了环境名称和实验配置之外，不修改 PPO 算法。

------------------------------------------------------------------------

# 22. CartPole 实验

## 目标

验证：

> PPO 的基本实现是否正确。

环境：

``` text
CartPole-v1
```

状态：

``` text
4-dimensional
```

动作：

``` text
2 discrete actions
```

实验流程：

``` text
CartPole-v1
    ↓
PPO(env)
    ↓
自动读取 state_dim/action_dim
    ↓
训练
    ↓
记录 reward
    ↓
绘制 reward curve
    ↓
evaluate
    ↓
保存 best model
```

CartPole 的意义：

> 如果一个基本 PPO 实现连 CartPole
> 都无法稳定学习，应优先检查算法实现、数据处理、GAE、ratio、clip、done
> 处理等，而不是直接增加复杂技巧。

------------------------------------------------------------------------

# 23. LunarLander 实验

环境：

``` text
LunarLander-v3
```

不重新实现 PPO。

只更换：

``` python
env = gym.make("LunarLander-v3")
```

然后：

``` python
agent = PPO(env, config)
```

PPO 自动获取：

``` text
state_dim
action_dim
```

网络自动适配。

实验目标：

> 验证 PPO
> 从简单环境迁移到状态和奖励结构更加复杂的环境后，代码是否仍然能够工作。

------------------------------------------------------------------------

# 24. 两个环境的代码复用原则

禁止：

``` python
if env_name == "CartPole-v1":
    # 一套 PPO

elif env_name == "LunarLander-v3":
    # 另一套 PPO
```

推荐：

``` python
env = gym.make(env_name)
agent = PPO(env, config)
agent.train()
```

PPO 内部：

``` python
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
```

这样：

``` text
Environment
     ↓
提供 observation_space / action_space
     ↓
PPO 自动构建网络
```

------------------------------------------------------------------------

# 25. `utils.py`

用于放一些不属于 PPO 核心算法的通用功能。

例如：

``` python
set_seed(...)
save_checkpoint(...)
load_checkpoint(...)
moving_average(...)
save_metrics(...)
```

可以包含：

``` text
随机种子
日志保存
模型保存辅助函数
数据处理
曲线数据处理
```

不要把 PPO 核心算法塞进 `utils.py`。

------------------------------------------------------------------------

# 26. 训练指标

训练过程中建议记录：

``` text
episode_reward
episode_length
evaluation_reward
actor_loss
critic_loss
entropy
approx_kl
```

最核心：

``` text
episode_reward
evaluation_reward
```

辅助分析：

``` text
actor_loss
critic_loss
entropy
```

------------------------------------------------------------------------

# 27. 训练曲线

至少绘制：

``` text
Reward vs Episode
```

推荐：

``` text
原始 reward
+
移动平均 reward
```

例如：

``` text
Reward
  │
  │                 ╭──────
  │            ╭────╯
  │       ╭────╯
  │  ╭────╯
  │──╯
  └────────────────────── Episode
```

这样更容易判断训练趋势。

------------------------------------------------------------------------

# 28. 评估曲线

如果训练过程中周期性评估：

``` text
Evaluation Reward vs Update
```

可以得到：

``` text
Update
  │
  │              ╭─────
  │         ╭────╯
  │    ╭────╯
  │────╯
  └────────────────────
```

相比单个 episode reward，更适合观察策略性能。

------------------------------------------------------------------------

# 29. 超参数实验设计

不要一开始同时改变所有参数。

推荐：

## 实验 0：Baseline

``` text
固定基础参数
```

目标：

``` text
CartPole 能训练
LunarLander 能训练
```

------------------------------------------------------------------------

## 实验 1：Learning Rate

``` text
1e-4
3e-4
1e-3
```

保持其他参数完全一致。

比较：

``` text
训练曲线
最终评估 reward
训练稳定性
```

------------------------------------------------------------------------

## 实验 2：Clip Epsilon

``` text
0.1
0.2
0.3
```

保持其他参数一致。

------------------------------------------------------------------------

## 实验 3：GAE Lambda

``` text
0.90
0.95
0.99
```

保持其他参数一致。

------------------------------------------------------------------------

# 30. 超参数实验控制变量原则

每次实验：

``` text
只改变一个主要超参数
```

例如：

``` text
实验 A
lr = 1e-4
gamma = 0.99
lambda = 0.95
epsilon = 0.2

实验 B
lr = 3e-4
gamma = 0.99
lambda = 0.95
epsilon = 0.2

实验 C
lr = 1e-3
gamma = 0.99
lambda = 0.95
epsilon = 0.2
```

这样才能知道：

> 性能变化主要来自哪个参数。

------------------------------------------------------------------------

# 31. 随机种子

强化学习存在随机性。

因此正式比较超参数时，不应该只跑一次。

推荐：

``` text
seed = 0
seed = 1
seed = 2
```

如果计算资源有限，普通课程实验至少可以：

``` text
2~3 seeds
```

然后报告：

``` text
mean reward
std reward
```

不要只比较某一次运行的最高 reward。

------------------------------------------------------------------------

# 32. 模型测试

训练结束后：

``` text
加载 best checkpoint
       ↓
创建测试环境
       ↓
Actor 推理
       ↓
执行 Action
       ↓
记录 Reward
```

测试环境可以：

``` python
env = gym.make(
    "CartPole-v1",
    render_mode="human"
)
```

或者：

``` python
env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)
```

------------------------------------------------------------------------

# 33. 可视化

测试阶段使用：

``` text
render_mode="human"
```

直接观察智能体行为。

也可以使用：

``` text
render_mode="rgb_array"
```

然后保存视频。

推荐最终展示：

``` text
训练曲线
+
测试运行视频
```

这样实验结果同时包含：

``` text
定量结果
```

和：

``` text
定性结果
```

------------------------------------------------------------------------

# 34. `test.py`

主要负责：

``` python
load_model(...)
test(...)
```

典型流程：

``` text
读取 checkpoint
      ↓
创建环境
      ↓
创建 PPO
      ↓
加载 Actor / Critic
      ↓
运行 N 个 episode
      ↓
记录 reward
      ↓
输出：
mean
std
min
max
```

测试时不要更新模型。

------------------------------------------------------------------------

# 35. `visualize.py`

负责测试过程的可视化。

可以提供：

``` python
run_visualization(...)
```

功能：

``` text
加载模型
创建 render 环境
运行 episode
展示智能体行为
```

如果保存视频，则使用 Gymnasium 的录制工具完成。

------------------------------------------------------------------------

# 36. 最终完整实验流程

``` text
                开始
                  │
                  ↓
        创建 Gymnasium Environment
                  │
                  ↓
              PPO(env)
                  │
                  ↓
        自动获取 state_dim
        自动获取 action_dim
                  │
                  ↓
       创建 Actor + Critic
                  │
                  ↓
          ┌───────────────┐
          │   Training    │
          └───────┬───────┘
                  │
                  ↓
         collect_rollout()
                  │
                  ↓
          Rollout Buffer
                  │
                  ↓
       Return + GAE Advantage
                  │
                  ↓
             PPO Update
                  │
          ┌───────┴───────┐
          ↓               ↓
        Actor           Critic
          │               │
          └───────┬───────┘
                  ↓
             记录 Metrics
                  │
                  ↓
             Periodic Eval
                  │
                  ↓
            是否 Best Model?
             /          \
           Yes           No
            │             │
            ↓             │
       Save Checkpoint    │
            │             │
            └──────┬──────┘
                   ↓
              继续训练
                   │
                   ↓
              训练完成
                   │
                   ↓
             最终评估
                   │
                   ↓
              加载 Best
                   │
                   ↓
             模型测试
                   │
                   ↓
             可视化运行
                   │
                   ↓
              实验结果
```

------------------------------------------------------------------------

# 37. 最终项目结构图

``` text
ppo-gym/
│
├── Plan.md
├── README.md
├── pyproject.toml
│
├── src/
│   └── ppo_gym/
│       │
│       ├── __init__.py
│       │
│       ├── networks.py
│       │      ├── Actor
│       │      └── Critic
│       │
│       ├── buffer.py
│       │      └── RolloutBuffer
│       │
│       ├── config.py
│       │      └── PPOConfig
│       │
│       ├── ppo.py
│       │      └── PPO
│       │          ├── select_action()
│       │          ├── collect_rollout()
│       │          ├── compute_advantage()
│       │          ├── update()
│       │          └── train()
│       │
│       ├── train.py
│       │      └── 训练入口
│       │
│       ├── evaluate.py
│       │      └── evaluate()
│       │
│       ├── test.py
│       │      └── test()
│       │
│       ├── visualize.py
│       │      └── run_visualization()
│       │
│       └── utils.py
│              ├── seed
│              ├── checkpoint
│              └── metrics
│
├── experiments/
│   ├── cartpole/
│   └── lunarlander/
│
├── checkpoints/
│   ├── cartpole/
│   │   ├── best.pth
│   │   └── latest.pth
│   │
│   └── lunarlander/
│       ├── best.pth
│       └── latest.pth
│
├── results/
│   ├── cartpole/
│   │   ├── metrics.csv
│   │   └── curves.png
│   │
│   └── lunarlander/
│       ├── metrics.csv
│       └── curves.png
│
└── videos/
    ├── cartpole/
    └── lunarlander/
```

------------------------------------------------------------------------

# 38. 文件依赖关系

``` text
                     train.py
                         │
                         ↓
                    PPO(env)
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
      networks.py    buffer.py      config.py
          │              │
      ┌───┴───┐          │
      ↓       ↓          │
    Actor   Critic        │
      │       │           │
      └───┬───┘           │
          ↓               ↓
             ppo.py
                │
       ┌────────┼────────┐
       ↓        ↓        ↓
    rollout   GAE     update
                │
                ↓
            metrics
                │
        ┌───────┴────────┐
        ↓                ↓
 evaluate.py          utils.py
        │                │
        ↓                ↓
 test.py           checkpoints/results
        │
        ↓
 visualize.py
```

------------------------------------------------------------------------

# 39. 最重要的代码复用验证

最终应该能够做到：

## CartPole

``` python
env = gym.make("CartPole-v1")

agent = PPO(env, config)

agent.train()
```

## LunarLander

``` python
env = gym.make("LunarLander-v3")

agent = PPO(env, config)

agent.train()
```

两者之间：

``` text
PPO代码：不变
Actor代码：不变
Critic代码：不变
Buffer代码：不变
GAE代码：不变
Update代码：不变
Evaluate代码：不变
Test代码：不变
```

变化的主要是：

``` text
环境名称
实验配置
模型保存路径
结果保存路径
```

------------------------------------------------------------------------

# 40. 推荐实现顺序

不要一次把所有功能全部写完。

按照下面顺序实现：

``` text
Step 1
Gymnasium 环境创建
        ↓
Step 2
Actor
        ↓
Step 3
Critic
        ↓
Step 4
PPO select_action()
        ↓
Step 5
RolloutBuffer
        ↓
Step 6
collect_rollout()
        ↓
Step 7
Return + GAE
        ↓
Step 8
PPO clipped update
        ↓
Step 9
CartPole 训练
        ↓
Step 10
训练曲线
        ↓
Step 11
Evaluate
        ↓
Step 12
Checkpoint
        ↓
Step 13
LunarLander 迁移
        ↓
Step 14
模型测试 + 可视化
        ↓
Step 15
超参数实验
```

------------------------------------------------------------------------

# 41. 实验完成后的最终成果

最终应该能够得到：

## 算法

``` text
PPO
Actor
Critic
GAE
Clipping
Entropy
```

## 环境

``` text
CartPole-v1
LunarLander-v3
```

## 训练

``` text
Training Reward
Evaluation Reward
Loss
Entropy
```

## 模型

``` text
best.pth
latest.pth
```

## 实验

``` text
Baseline
Learning Rate
Clip Epsilon
GAE Lambda
```

## 测试

``` text
平均 Reward
Reward 标准差
测试 Episode
```

## 可视化

``` text
训练曲线
评估曲线
智能体运行效果
```

------------------------------------------------------------------------

# 42. 最终核心脉络

整个项目最终可以压缩成这一条线：

``` text
Gymnasium Environment
        ↓
      State
        ↓
      Actor ─────────→ Action
        │                 ↓
        │             Environment
        │                 ↓
        │              Reward
        │                 ↓
        └──────→ Rollout Buffer
                         ↓
                    Critic V(s)
                         ↓
                  GAE Advantage
                         ↓
                  PPO Clipping
                         ↓
                Actor + Critic Update
                         ↓
                    新策略
                         ↓
                    下一轮 Rollout
```

而整个实验的核心验证是：

``` text
        同一个 PPO
             │
       ┌─────┴─────┐
       ↓           ↓
   CartPole    LunarLander
       │           │
       ↓           ↓
   验证正确性   验证迁移能力
```

这就是本项目最核心的设计原则：

> **环境变化，但 PPO 算法不变。**
