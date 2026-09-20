"""Rollout 数据缓存。

按照 Plan.md §6 的设计,RolloutBuffer 负责保存一次 rollout 收集的数据:

    Environment → collect_rollout() → RolloutBuffer
        → Advantage / Return → PPO Update

保存字段:
    states / actions / rewards / dones(含 terminated 与 truncated)
    log_probs(旧策略的 π_old)/ values(Critic 的 V(s))
    bootstrap_values(截断时刻最终观测的 V,仅 truncated 时非零)

不显式保存全部 next_states:GAE 通过"下一时刻的 value + 末端 bootstrap
value"计算。这一点与 CleanRL 一致,但与 CleanRL 不同的是,这里严格区分
terminated 和 truncated(Plan 前半 §5.2 的硬性要求):
- terminated=True:环境自然终止,下一状态价值为 0;
- truncated=True:时间上限截断,必须从截断时刻的最终观测 bootstrap,
  该价值在采样时由 PPO 计算并写入 bootstrap_values。
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch


class RolloutBuffer:
    """保存一次 rollout,并计算 Return 与 GAE Advantage。"""

    def __init__(self) -> None:
        """初始化各字段的空列表,以及计算结果的占位。"""

        self.states: list = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.terminateds: list[bool] = []
        self.truncateds: list[bool] = []
        self.bootstrap_values: list[float] = []

        # compute_returns_and_advantages 的产物,None 表示尚未计算。
        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None

    def add(
        self,
        state,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        terminated: bool,
        truncated: bool,
        bootstrap_value: float = 0.0,
    ) -> None:
        """向 buffer 追加一次环境交互记录。

        参数:
            state          本时刻的观测 s_t。
            action         实际执行的动作 a_t。
            reward         环境奖励 r_t。
            done           terminated or truncated,用于切分轨迹。
            log_prob       旧策略下该动作的 log π_old(a_t|s_t),PPO ratio 的分母。
            value          Critic 对 s_t 的估值 V(s_t),GAE 的基础。
            terminated     True 表示环境自然终止,bootstrap 价值为 0。
            truncated      True 表示时间上限截断,需要 bootstrap(见下)。
            bootstrap_value 截断时刻最终观测的 V(s)。仅 truncated=True 时
                           需要传入;因为 env.reset 之后该观测就丢了,必须在
                           采样现场计算。
        """

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.terminateds.append(terminated)
        self.truncateds.append(truncated)
        self.bootstrap_values.append(bootstrap_value)

    def __len__(self) -> int:
        """返回当前 buffer 中保存的时间步数量。"""

        return len(self.rewards)

    def clear(self) -> None:
        """清空所有字段,每个新 rollout 开始时调用"""

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.terminateds.clear()
        self.truncateds.clear()
        self.bootstrap_values.clear()
        self.advantages = None
        self.returns = None

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor | float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """计算 GAE Advantage 和 Return,结果保存在 self.advantages / self.returns。

        核心公式(Plan §10):
            δ_t = r_t + γ · V(s_{t+1}) · (1 - terminated_t) - V(s_t)
            A_t = δ_t + γλ · (1 - done_t) · A_{t+1}

        三个处理要点:
        1. 从后向前逆序递推。last_value 是 rollout 结束时"当前观测"的 V(s),
           只有 rollout 在回合中间被切断时才会真正被用到;
        2. V(s_{t+1}) 的取值优先级:
           truncated_t → bootstrap_values[t](截断观测的价值);
           t 为最后一步 → last_value;
           其他 → values[t+1](下一状态仍在同一回合内)。
           terminated_t 时 (1 - terminated)=0,下一状态价值被屏蔽;
        3. λ 递推项使用 (1 - done_t) 而不是 (1 - terminated_t):
           无论终止还是截断,时间轴上回合已经结束,优势不应跨回合传递。

        Return = A_t + V(s_t),作为 Critic 的回归目标(Plan §11)。
        """

        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        terminateds = np.asarray(self.terminateds, dtype=np.float32)
        truncateds = np.asarray(self.truncateds, dtype=np.float32)
        bootstrap_values = np.asarray(self.bootstrap_values, dtype=np.float32)
        dones = np.clip(terminateds + truncateds, 0.0, 1.0)

        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        last_value = float(last_value)

        # 倒着来算
        for step in reversed(range(len(rewards))):
            if truncateds[step] > 0:
                next_value = bootstrap_values[step]
            elif step == len(rewards) - 1:
                next_value = last_value
            else:
                # index从0开始,所以 step+1 表示自己
                next_value = values[step + 1]

            # non_terminal 正好可以用来屏蔽不存在的下一个value,即 terminated=True 时的情况
            non_terminal = 1.0 - terminateds[step]
            delta = (
                rewards[step]
                + gamma * next_value * non_terminal
                - values[step]
            )
            last_gae = delta + gamma * gae_lambda * (1.0 - dones[step]) * last_gae
            advantages[step] = last_gae

        # 重点
        self.advantages = advantages
        self.returns = advantages + values

    def get(self, batch_size: int, device: torch.device) -> Iterator[dict[str, torch.Tensor]]:
        """把 rollout 数据转换成张量并按 mini-batch 随机产出。

        - 每次调用 get() 都会重新打乱索引,因此 PPO.update() 的每个 epoch
          调用一次 get(),即可得到与 CleanRL 等价的"每轮重新洗牌"行为;
        - advantage 的标准化放在 PPO.update() 里按 mini-batch 做
          (CleanRL 的 norm_adv 约定),这里不做。

        参数:
            batch_size  mini-batch 大小,最后一个批次可能更小。
            device      训练设备。
        """

        if self.advantages is None or self.returns is None:
            raise RuntimeError("必须先调用 compute_returns_and_advantages() 再调用 get()")

        states = torch.as_tensor(np.asarray(self.states), dtype=torch.float32, device=device)
        actions = torch.as_tensor(self.actions, dtype=torch.long, device=device)
        log_probs = torch.as_tensor(self.log_probs, dtype=torch.float32, device=device)
        returns = torch.as_tensor(self.returns, dtype=torch.float32, device=device)
        advantages = torch.as_tensor(self.advantages, dtype=torch.float32, device=device)

        total = states.shape[0]
        batch_size = min(batch_size, total)
        indices = np.arange(total)
        np.random.shuffle(indices)

        for start in range(0, total, batch_size):
            idx = torch.as_tensor(indices[start : start + batch_size], device=device)
            yield {
                "states": states[idx],
                "actions": actions[idx],
                "old_log_probs": log_probs[idx],
                "returns": returns[idx],
                "advantages": advantages[idx],
            }
