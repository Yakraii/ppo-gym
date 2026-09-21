"""PPO 算法核心

训练主循环:
    Collect → Estimate → Update → Collect → Estimate → Update → ...

 CleanRL:
- Adam 优化器 eps=1e-5,小网络下比默认 1e-8 更稳定;
- approx_kl 使用 k3 估计 ((ratio-1) - logratio).mean(),数值上比
  (-logratio).mean() 更稳定;
- clip_fraction 诊断统计(被裁剪的样本比例);
- 每个 mini-batch 内独立做 advantage 标准化;
- 每轮 epoch 重新打乱数据。

"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .config import PPOConfig
from .evaluate import evaluate
from .utils import (
    env_dir_name,
    plot_diagnostics,
    plot_evaluation_curve,
    plot_training_curve,
    resolve_device,
    save_json,
    save_metrics,
)


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """对单个 Linear 层做正交初始化(CleanRL 经验)。

    隐藏层用 std=√2(Tanh 激活的标准选择);
    输出层的 std 由调用方显式指定(Actor 0.01 / Critic 1.0)。
    """

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Actor-Critic 网络:独立的 Actor 和 Critic 两个 MLP(CleanRL 风格)。

    Critic 输出层 std=1.0(价值量级不确定,不做收缩);
    Actor 输出层 std=0.01(初始 logits 几乎全 0,策略接近均匀随机)。
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        """state_dim / action_dim 由 PPO 从环境自动读入,不写死具体环境的维度。"""

        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )


class RolloutBuffer:
    """保存一次 rollout 的数据,计算 GAE Advantage / Return,按 mini-batch 随机产出。

    不显式保存 next_states:GAE 通过"下一时刻的 value + 末端 bootstrap
    value"计算(与 CleanRL 一致)。严格区分 terminated 和 truncated:
    terminated 的下一状态价值为 0;truncated 必须用截断观测的
    bootstrap_values(采样现场算好,reset 之后观测就丢了)。
    """

    def __init__(self) -> None:
        self.states: list = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.terminateds: list[bool] = []
        self.truncateds: list[bool] = []
        self.bootstrap_values: list[float] = []
        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None

    def add(
        self,
        state,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        terminated: bool,
        truncated: bool,
        bootstrap_value: float = 0.0, # 不必要的时候 0 就行
    ) -> None:
        """追加一个时间步;bootstrap_value 仅 truncated=True 时需要传入。"""

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.terminateds.append(terminated)
        self.truncateds.append(truncated)
        self.bootstrap_values.append(bootstrap_value)

    def clear(self) -> None:
        """清空所有字段,每个新 rollout 开始时调用。"""

        for name in (
            "states", "actions", "rewards", "log_probs", "values",
            "terminateds", "truncateds", "bootstrap_values",
        ):
            getattr(self, name).clear()
        self.advantages = None
        self.returns = None

    def compute_returns_and_advantages(
        self, last_value, gamma: float, gae_lambda: float
    ) -> None:
        """逆序递推 GAE,结果存入 self.advantages / self.returns。

        next_value 的取法:truncated → 截断观测的 bootstrap 价值;
        最后一步 → last_value;其余 → values[t+1]。
        terminated 屏蔽下一状态价值;λ 递推用 (1 - done),
        无论终止还是截断,优势都不跨回合传递。
        """

        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        terminateds = np.asarray(self.terminateds, dtype=np.float32)
        truncateds = np.asarray(self.truncateds, dtype=np.float32)
        dones = np.maximum(terminateds, truncateds)
        last_value = float(last_value)

        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for step in reversed(range(len(rewards))):
            if truncateds[step] > 0:
            # 符合截断的情况的话
                next_value = self.bootstrap_values[step]
            elif step == len(rewards) - 1:
            # rollout最后一个但是后面本来还有value
                next_value = last_value
            else:
            # 中间步
                next_value = values[step + 1]

            delta = (
                rewards[step]
                + gamma * next_value * (1.0 - terminateds[step])
                - values[step]
            )
            # 如果是最后一个就是自己的delta
            last_gae = delta + gamma * gae_lambda * (1.0 - dones[step]) * last_gae
            advantages[step] = last_gae
        # 核心
        self.advantages = advantages
        self.returns = advantages + values

    def get(
        self, batch_size: int, device: torch.device
    ) -> Iterator[dict[str, torch.Tensor]]:
        """把 rollout 数据转成张量并按 mini-batch 随机产出;每次调用重新打乱。"""

        if self.advantages is None:
            raise RuntimeError("必须先调用 compute_returns_and_advantages() 再调用 get()")

        states = torch.as_tensor(np.asarray(self.states), dtype=torch.float32, device=device)
        actions = torch.as_tensor(self.actions, dtype=torch.long, device=device)
        log_probs = torch.as_tensor(self.log_probs, dtype=torch.float32, device=device)
        returns = torch.as_tensor(self.returns, dtype=torch.float32, device=device)
        advantages = torch.as_tensor(self.advantages, dtype=torch.float32, device=device)

        total = states.shape[0]

        # 如果总数还不够一个batch,就直接返回全部数据。
        batch_size = min(batch_size, total)
        indices = np.random.permutation(total)

        for start in range(0, total, batch_size):
            idx = torch.as_tensor(indices[start : start + batch_size], device=device)
            yield {
                "states": states[idx],
                "actions": actions[idx],
                "old_log_probs": log_probs[idx],
                "returns": returns[idx],
                "advantages": advantages[idx],
            }


class PPO:
    """最小 PPO Agent,可复用于所有离散动作环境。"""

    def __init__(self, env: gym.Env, config: PPOConfig) -> None:
        """初始化 PPO。

            state_dim  = env.observation_space.shape[0]
            action_dim = env.action_space.n
            agent      = Agent(state_dim, action_dim)   # 内含 actor 与 critic

        同时创建:
        - 单个 Adam 优化器(eps=1e-5)统一管理两个网络的参数;
        - 复用的 RolloutBuffer 与回合统计字段。
        """

        self.env = env
        self.config = config
        self.device = resolve_device(config.device)

        # 自动从环境读取维度,创建 Agent(Actor + Critic)/ Optimizer / Buffer
        state_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.n)
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 两个神经网络一起放进去
        self.ac = Agent(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.ac.parameters(),
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


    def select_action(self, state, deterministic: bool = False):
        """根据当前 Actor 对单个状态选择动作。

        流程:State → Actor → logits → Categorical 分布 → Action。

        返回:
            action     采样得到的动作;deterministic=True 时取 argmax;
            log_prob   该动作的 log π(a|s),PPO ratio 的分母;
            value      Critic 的 V(s)。

        deterministic=True 用于评估和测试,不再sample。
        """

        # unsqueeze加一维
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            logits = self.ac.actor(state_tensor)
            dist = Categorical(logits=logits)
            # 测试直接选概率最高的
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.ac.critic(state_tensor).squeeze(-1)
        
        return int(action.item()), float(log_prob.item()), float(value.item())

    def _state_value(self, obs) -> float:
        """计算单个观测的价值(不参与梯度),达到时间上限供截断 bootstrap 使用。"""

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value = self.ac.critic(obs_tensor)
        return float(value.item())

    def collect_rollout(self) -> None:
        """使用当前策略与环境交互,收集一批训练数据。

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

    def update(self) -> dict[str, float]:
        """执行若干个 epoch 的 PPO 更新,返回平均后的训练指标。

        对每个 mini-batch:
        1. 重新计算 log π_θ(a_t|s_t),ratio = exp(log_new - log_old);
        2. Actor 的 clipped objective:
               policy_loss = mean(max(-ratio·A, -clip(ratio)·A))
        3. Critic 的 MSE(乘 0.5,与 CleanRL 的梯度尺度约定一致):
               value_loss = 0.5 · mean((V(s) - Return)²)
        4. 总 loss:
               loss = policy_loss + value_coef·value_loss - entropy_coef·entropy
        5. 反向传播 → 梯度裁剪 → optimizer.step()。

        返回指标:policy_loss / value_loss / entropy /
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

                logits = self.ac.actor(states)  # 新 Actor 对每个动作输出原始分数。
                dist = Categorical(logits=logits)  # 根据 logits 创建离散动作分布。
                new_log_probs = dist.log_prob(actions)  # 新策略对旧动作的 log 概率。
                entropy = dist.entropy().mean()  # 平均熵,鼓励 Actor 保持探索。
                values = self.ac.critic(states).squeeze(-1)  # Critic 对当前状态的价值预测。

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
                # 不懂... 为什么还要标准化advantage
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
                    self.ac.parameters(),
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

    def train(self) -> None:
        """完整训练流程:

            collect_rollout → compute_advantage → update
                → 记录 metrics → 周期评估 → best/latest 保存 → 循环

        产物:
            checkpoints/<env>/best.pth     评估 reward 最优的模型
            checkpoints/<env>/latest.pth   最新模型
            results/<env>/metrics.csv      每回合 return / length
            results/<env>/update_metrics.csv   每次 update 的 loss 指标
            results/<env>/eval_metrics.csv     周期评估结果
            results/<env>/curves.png       训练曲线(原始 + 移动平均)
            results/<env>/eval_curve.png   评估曲线
            results/<env>/diagnostics.png  训练诊断面板(KL/clip/熵/价值损失)
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
            last_value = self._state_value(self._obs)
            self.buffer.compute_returns_and_advantages(
                last_value=last_value,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )
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
        plot_diagnostics(update_records, result_dir / "diagnostics.png")

        print(f"训练完成: {self.global_step} 步,最优评估 reward={best_reward:.2f}")
        print(f"checkpoints: {checkpoint_dir.resolve()}")
        print(f"results:     {result_dir.resolve()}")

    # ------------------------------------------------------------------
    # 模型保存与加载
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
            "actor": self.ac.actor.state_dict(),
            "critic": self.ac.critic.state_dict(),
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
        # 只取 PPOConfig 认识的字段,兼容早期存档里的额外键。
        known_fields = PPOConfig.__dataclass_fields__
        config = PPOConfig(
            **{k: v for k, v in checkpoint["config"].items() if k in known_fields}
        )
        config.device = device

        agent = cls(env, config)
        # 兼容旧存档:旧版 Actor/Critic 内部多一层 "net" 包装,键带 "net." 前缀。
        strip = lambda sd: {k.removeprefix("net."): v for k, v in sd.items()}
        agent.ac.actor.load_state_dict(strip(checkpoint["actor"]))
        agent.ac.critic.load_state_dict(strip(checkpoint["critic"]))
        agent.ac.eval()
        return agent
