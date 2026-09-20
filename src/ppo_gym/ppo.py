"""PPO 算法核心:整个实验最重要的文件(Plan §7~§15)。

设计约束:
- PPO 在初始化时自动从环境读取 state_dim / action_dim,
  因此它不需要知道自己运行在 CartPole 还是 LunarLander 中;
- 除了环境名称和实验配置,迁移环境时不允许修改本文件(Plan §24);
- 禁止出现 if env_name == "CartPole-v1" 之类的分支。

训练主循环(Plan §9):
    Collect → Estimate → Update → Collect → Estimate → Update → ...

从 CleanRL(cleanrl_ppo.py)吸收的实现细节:
- Adam 优化器 eps=1e-5,小网络下比默认 1e-8 更稳定;
- approx_kl 使用 k3 估计 ((ratio-1) - logratio).mean(),数值上比
  (-logratio).mean() 更稳定;
- clip_fraction 诊断统计(被裁剪的样本比例);
- 每个 mini-batch 内独立做 advantage 标准化;
- 每轮 epoch 重新打乱数据。

刻意不采用的 CleanRL 内容(与项目要求冲突,见对照分析):
tyro / wandb / TensorBoard / 向量化环境 / 学习率退火 /
裁剪价值损失 clip_vloss / target_kl 提前停止。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical

from .buffer import RolloutBuffer
from .config import PPOConfig
from .evaluate import evaluate
from .networks import Actor, Critic
from .utils import (
    env_dir_name,
    plot_evaluation_curve,
    plot_training_curve,
    resolve_device,
    save_json,
    save_metrics,
)


class PPO:
    """最小 PPO Agent,可复用于所有离散动作环境。"""

    def __init__(self, env: gym.Env, config: PPOConfig) -> None:
        """初始化 PPO。

        按照Plan §7,自动从环境读取维度并创建网络:

            state_dim  = env.observation_space.shape[0]
            action_dim = env.action_space.n
            actor      = Actor(state_dim, action_dim)
            critic     = Critic(state_dim)

        同时创建:
        - 单个 Adam 优化器(eps=1e-5)统一管理两个网络的参数;
        - 复用的 RolloutBuffer 与回合统计字段。
        """

        self.env = env
        self.config = config
        self.device = resolve_device(config.device)

        # 自动从环境读取维度,创建 Actor / Critic / Optimizer / Buffer
        state_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.n)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim, config.hidden_sizes).to(self.device)
        self.critic = Critic(state_dim, config.hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.learning_rate,
            eps=config.adam_eps,
        )

        self.buffer = RolloutBuffer()

        # 训练进度与回合统计(train() 会持续更新它们)。
        self.global_step = 0
        self.iteration = 0
        self.episode_records: list[dict[str, float | int]] = []
        self._episode_index = 0
        self._episode_return = 0.0
        self._episode_length = 0
        self._obs, _ = self.env.reset(seed=config.seed)

    # ------------------------------------------------------------------
    # 采样阶段
    # ------------------------------------------------------------------

    def select_action(self, state, deterministic: bool = False):
        """根据当前 Actor 对单个状态选择动作(Plan §8.1)。

        流程:State → Actor → logits → Categorical 分布 → Action。

        返回:
            action     采样得到的动作;deterministic=True 时取 argmax;
            log_prob   该动作的 log π(a|s),PPO ratio 的分母;
            value      Critic 的 V(s)。

        deterministic=True 用于评估和测试(Plan §19:评估不采样)。
        """

        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            logits = self.actor(state_tensor)
            dist = Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state_tensor).squeeze(-1)
        
        return int(action.item()), float(log_prob.item()), float(value.item())

    def _state_value(self, obs) -> float:
        """计算单个观测的价值(不参与梯度),达到时间上限供截断 bootstrap 使用。"""

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(obs_tensor)
        return float(value.item())

    def collect_rollout(self) -> None:
        """使用当前策略与环境交互,收集一批训练数据(Plan §8.2)。

        rollout 结束后 self._obs 停在"当前观测"上:
        - 若回合还在进行中,它就是 GAE 末端 bootstrap 用的 s_T;
        - 若最后一步恰好 done,已 env.reset(),GAE 那一侧会通过
          terminated/truncated 屏蔽掉末端价值,不会误用新回合的观测。
        """

        self.buffer.clear()

        for _ in range(self.config.rollout_steps):
            if self.global_step >= self.config.total_steps:
                break

            action, log_prob, value = self.select_action(self._obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # 截断时必须在 reset 之前算好最终观测的价值,否则观测就丢了。
            bootstrap_value = self._state_value(next_obs) if truncated else 0.0

            # value是当前的 bootstrap_value是下一步的价值,用于截断时的GAE计算
            self.buffer.add(
                state=self._obs,
                action=action,
                reward=float(reward),
                done=done,
                log_prob=log_prob,
                value=value,
                terminated=terminated,
                truncated=truncated,
                bootstrap_value=bootstrap_value,
            )
            
            self.global_step += 1
            self._episode_return += float(reward)
            self._episode_length += 1
            self._obs = next_obs

            if done:
                # 当前回合结束,记录回合统计信息。
                self._episode_index += 1
                self.episode_records.append(
                    {
                        "episode": self._episode_index,
                        "step": self.global_step,
                        "return": round(self._episode_return, 2),
                        "length": self._episode_length,
                    }
                )
                print(
                    f"episode={self._episode_index:5d} "
                    f"step={self.global_step:8d} "
                    f"return={self._episode_return:8.2f}"
                )
                # 开始新回合,清零本回合的累计数据。
                self._obs, _ = self.env.reset()
                self._episode_return = 0.0
                self._episode_length = 0

    # ------------------------------------------------------------------
    # 估计阶段
    # ------------------------------------------------------------------

    def compute_advantage(self) -> None:
        """计算 Advantage 和 Return(Plan §10/§11)。

        实际计算委托给 buffer.compute_returns_and_advantages(),
        这里只负责提供 bootstrap 用的 last_value(当前观测的 V(s))。
        """

        last_value = self._state_value(self._obs)
        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

    # ------------------------------------------------------------------
    # 更新阶段
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        """执行若干个 epoch 的 PPO 更新,返回平均后的训练指标(Plan §12~§14)。

        对每个 mini-batch:
        1. 重新计算 log π_θ(a_t|s_t),ratio = exp(log_new - log_old);
        2. Actor 的 clipped objective:
               policy_loss = mean(max(-ratio·A, -clip(ratio)·A))
        3. Critic 的 MSE(乘 0.5,与 CleanRL 的梯度尺度约定一致):
               value_loss = 0.5 · mean((V(s) - Return)²)
        4. 总 loss:
               loss = policy_loss + value_coef·value_loss - entropy_coef·entropy
        5. 反向传播 → 梯度裁剪 → optimizer.step()。

        返回指标(Plan §26):policy_loss / value_loss / entropy /
        approx_kl(k3 估计)/ old_approx_kl / clip_fraction。
        """

        # 累计每个 mini-batch 的训练指标,最后求平均后返回。
        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "old_approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        update_count = 0  # 实际完成的 mini-batch 更新次数。

        for _ in range(self.config.update_epochs):
            # 同一批 rollout 数据重复训练多轮;每轮 get() 都会重新打乱样本。
            for batch in self.buffer.get(self.config.batch_size, self.device):
                # print(f"batch: {batch['states'].shape}, {batch['actions'].shape}, {batch['returns'].shape}")
                states = batch["states"]  # [batch_size, state_dim],当前状态。
                actions = batch["actions"]  # [batch_size],实际执行的离散动作。
                old_log_probs = batch["old_log_probs"]  # 采样时旧策略的动作 log 概率。
                returns = batch["returns"]  # Critic 的目标价值。
                advantages = batch["advantages"]  # Actor 判断动作好坏的信号。

                logits = self.actor(states)  # 新 Actor 对每个动作输出原始分数。
                dist = Categorical(logits=logits)  # 根据 logits 创建离散动作分布。
                new_log_probs = dist.log_prob(actions)  # 新策略对旧动作的 log 概率。
                entropy = dist.entropy().mean()  # 平均熵,鼓励 Actor 保持探索。
                values = self.critic(states).squeeze(-1)  # Critic 对当前状态的价值预测。

                # 比较新旧策略: ratio > 1 表示新策略更偏好该动作。
                log_ratio = new_log_probs - old_log_probs
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # 以下三个值只用于监控策略变化,不参与反向传播。
                    old_approx_kl = (-log_ratio).mean()  # 旧策略与新策略差异的近似指标。
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()  # 更稳定的 KL 近似。
                    clip_fraction = (
                        ((ratio - 1.0).abs() > self.config.clip_epsilon)
                        .float()
                        .mean()
                    )  # 超出裁剪范围的样本比例。

                # 每个 mini-batch 独立标准化 Advantage,避免数值尺度影响策略更新。
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # PPO 策略损失:同时计算未裁剪和裁剪后的两种目标。
                policy_loss1 = -advantages * ratio
                policy_loss2 = -advantages * torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                )
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()  # 训练 Actor。

                # Critic 回归损失:让预测的 values 接近 GAE 得到的 returns。预估的和"实际的"
                value_loss = 0.5 * ((values - returns) ** 2).mean()

                # 总损失:策略损失 + 价值损失权重 - 熵奖励权重。
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()  # 清除 Actor 和 Critic 上一轮的梯度。
                loss.backward()  # 根据总损失分别计算两套网络的梯度。
                # PyTorch记录了整条计算路径，所以能从loss反推参数应该怎么调整。
                
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )  # 限制梯度大小,提高训练稳定性。
                self.optimizer.step()  # 用 Adam 同时更新 Actor 和 Critic。

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.item())
                stats["approx_kl"] += float(approx_kl.item())
                stats["old_approx_kl"] += float(old_approx_kl.item())
                stats["clip_fraction"] += float(clip_fraction.item())
                update_count += 1

        return {name: value / max(update_count, 1) for name, value in stats.items()}

    # ------------------------------------------------------------------
    # 训练主循环
    # ------------------------------------------------------------------

    def train(self) -> None:
        """完整训练流程(Plan §15/§36):

            collect_rollout → compute_advantage → update
                → 记录 metrics → 周期评估 → best/latest 保存 → 循环

        产物(按 Plan §20/§26/§27/§28):
            checkpoints/<env>/best.pth     评估 reward 最优的模型
            checkpoints/<env>/latest.pth   最新模型
            results/<env>/metrics.csv      每回合 return / length
            results/<env>/update_metrics.csv   每次 update 的 loss 指标
            results/<env>/eval_metrics.csv     周期评估结果
            results/<env>/curves.png       训练曲线(原始 + 移动平均)
            results/<env>/eval_curve.png   评估曲线
            results/<env>/config.json      本次运行的完整配置
        """

        env_dir = env_dir_name(self.config.env_name)
        checkpoint_dir = Path("checkpoints") / env_dir
        result_dir = Path("results") / env_dir

        # 评估用独立环境:不污染训练环境里正在进行的回合状态。
        eval_env = gym.make(self.config.env_name)

        best_reward = float("-inf")
        update_records: list[dict[str, float | int]] = []
        eval_records: list[dict[str, float | int]] = []

        # 当 global_step >= total_steps 时训练结束,即使 rollout 还没收集完。
        while self.global_step < self.config.total_steps:
            self.iteration += 1

            self.collect_rollout()
            self.compute_advantage()
            stats = self.update()

            update_records.append(
                {"iteration": self.iteration, "step": self.global_step, **stats}
            )
            print(
                f"iteration={self.iteration:5d} step={self.global_step:8d} "
                f"policy_loss={stats['policy_loss']:.4f} "
                f"value_loss={stats['value_loss']:.4f} "
                f"entropy={stats['entropy']:.4f} "
                f"approx_kl={stats['approx_kl']:.4f} "
                f"clip_frac={stats['clip_fraction']:.3f}"
            )

            training_done = self.global_step >= self.config.total_steps

            # 如果训练结束或达到周期评估间隔,就做一次评估。
            if self.iteration % self.config.eval_interval == 0 or training_done:
                result = evaluate(self, eval_env, self.config.eval_episodes)
                eval_records.append(
                    {
                        "iteration": self.iteration,
                        "step": self.global_step,
                        "mean_reward": result["mean_reward"],
                        "std_reward": result["std_reward"],
                    }
                )
                print(
                    f"[eval] iteration={self.iteration} "
                    f"mean_reward={result['mean_reward']:.2f} "
                    f"std_reward={result['std_reward']:.2f}"
                )

                if result["mean_reward"] > best_reward:
                    best_reward = result["mean_reward"]
                    self.save(checkpoint_dir / "best.pth")
                    print(f"[save] 新的最优模型 (mean={best_reward:.2f})")

                self.save(checkpoint_dir / "latest.pth")

        eval_env.close()

        save_metrics(result_dir / "metrics.csv", self.episode_records)
        save_metrics(result_dir / "update_metrics.csv", update_records)
        save_metrics(result_dir / "eval_metrics.csv", eval_records)
        save_json(result_dir / "config.json", asdict(self.config))
        plot_training_curve(self.episode_records, result_dir / "curves.png")
        plot_evaluation_curve(eval_records, result_dir / "eval_curve.png")

        print(f"训练完成: {self.global_step} 步,最优评估 reward={best_reward:.2f}")
        print(f"checkpoints: {checkpoint_dir.resolve()}")
        print(f"results:     {result_dir.resolve()}")

    # ------------------------------------------------------------------
    # 模型保存与加载(Plan §20)
    # ------------------------------------------------------------------

    def save(self, path, include_optimizer: bool = False) -> None:
        """保存 checkpoint。

        至少保存:
            {"actor": actor.state_dict(), "critic": critic.state_dict()}

        include_optimizer=True 时额外保存 optimizer / global_step,
        用于断点继续训练。config 始终保存,test/visualize 加载时需要。
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "config": asdict(self.config),
            "global_step": self.global_step,
        }
        if include_optimizer:
            payload["optimizer"] = self.optimizer.state_dict()

        torch.save(payload, path)

    @classmethod
    def load(cls, path, env: gym.Env, device: str = "auto") -> "PPO":
        """从 checkpoint 恢复 PPO Agent。

        步骤:读取 checkpoint → 按 env 和存档 config 重建 PPO
        → 加载 actor/critic 权重 → 切到 eval 模式,供 test / visualize 使用。
        """

        checkpoint = torch.load(path, map_location="cpu")
        config = PPOConfig(**checkpoint["config"])
        config.device = device

        agent = cls(env, config)
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.actor.eval()
        agent.critic.eval()
        return agent
