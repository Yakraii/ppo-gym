### 一、课题名称

基于 PPO 的经典控制任务训练与实验分析
> 使用pytorch自己实现一下ppo算法，环境用openai的Gymnasium就行

[[PPO计划]]
### 二、课题目标

本课题不要求提出新算法，主要目标是让新同学熟悉完整的**强化学习开发流程**，包括：

- [ ] 环境搭建与交互；
- [ ] 状态、动作、奖励的定义；
- [ ] Agent 与 Environment 的交互循环；
- [ ] Policy Network / Value Network 的基本结构；
- [ ] 轨迹采样；
- [ ] Return 与 Advantage 的计算；
- [ ] PPO 算法训练；
- [ ] 模型保存与加载；
- [ ] 训练曲线绘制；
- [ ] 超参数实验；
- [ ] 模型测试与可视化；
- [ ] Git 项目管理与实验复现。

最终要求能够独立完成一个**强化学习算法**从**训练**到**测试**的完整流程。
### 三、基础环境

使用 **Gymnasium** 中的两个经典环境：

#### 任务一：CartPole-v1

目标：控制小车左右移动，使杆保持竖直。

##### 状态空间：
x
x_dot
theta
theta_dot

即：
State Dimension = 4

##### 动作空间：

0 -> 向左
1 -> 向右

即：

Action Dimension = 2

这个任务主要用于：

**验证强化学习代码是否正确**。
如果 PPO 连 CartPole 都训练不起来，说明代码实现大概率存在问题。

### 四、第二个环境：LunarLander-v3

在完成 CartPole 后，将同样的 PPO 代码**迁移**到：
**LunarLander-v3**

任务**目标**：

控制登月器安全着陆。

相比 CartPole：

状态维度更高
奖励更加复杂
训练时间更长
策略更加明显

这个任务主要**用于**：

检验强化学习代码的**泛化能力**。
要求**不能**为不同环境**重新写**一套 PPO。
算法代码必须能够**复用**。

### 五、算法要求

实现：

**PPO**
Proximal Policy Optimization

**不建议**一开始直接调用 Stable-Baselines3。

基础任务要求：

**自己使用 PyTorch 实现 PPO**。

可以**参考论文**或**公开实现**，但必须能够**解释代码**。

##### 网络
网络至少包括：

State
  ↓
MLP
  ↓
Actor
  ↓
Action Distribution

以及：

State
  ↓
MLP
  ↓
Critic
  ↓
State Value

可以：

Actor-Critic 共用 Backbone

或者：

Actor 和 Critic 使用独立网络

二者均可。