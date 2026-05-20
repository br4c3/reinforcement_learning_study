#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from dogfight_sim.mappo_env import MappoDogfightEnv, MappoDogfightConfig


def curriculum_mode(update: int, total_updates: int, start_progress: float = 0.0) -> str:
    local_progress = (update - 1) / max(1, total_updates - 1)
    progress = start_progress + (1.0 - start_progress) * local_progress
    if progress <= 0.30:
        return "static"
    if progress <= 0.70:
        return "circle"
    return "selfplay"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, joint_dim: int, action_dim: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, action_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))
        self.critic = nn.Sequential(nn.Linear(joint_dim, 192), nn.Tanh(), nn.Linear(192, 192), nn.Tanh(), nn.Linear(192, 2))

    def dist(self, obs: torch.Tensor) -> Normal:
        return Normal(self.actor(obs), self.log_std.exp())

    def value(self, joint_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(joint_obs)


def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    adv = torch.zeros_like(rewards)
    last_adv = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(rewards.shape[0])):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        adv[t] = last_adv
        next_value = values[t]
    returns = adv + values
    return adv, returns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--model", type=Path, default=Path("models/mappo_6dof_dogfight.pt"))
    parser.add_argument("--load-model", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--curriculum-start", type=float, default=0.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    env = MappoDogfightEnv(MappoDogfightConfig(seed=args.seed, enemy_mode="static"))
    model = ActorCritic(env.obs_dim, env.obs_dim * 2, env.action_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    if args.load_model is not None:
        checkpoint = torch.load(args.load_model, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optim.load_state_dict(checkpoint["optimizer_state_dict"])
    args.curriculum_start = float(np.clip(args.curriculum_start, 0.0, 1.0))
    obs = env.reset(seed=args.seed)
    print(f"training MAPPO on device={device}")

    for update in range(1, args.updates + 1):
        mode = curriculum_mode(update, args.updates, args.curriculum_start)
        if env.config.enemy_mode != mode:
            env.config.enemy_mode = mode
            obs = env.reset()
        obs_buf, joint_buf, action_buf, logp_buf, reward_buf, done_buf, value_buf = [], [], [], [], [], [], []
        ep_returns = []
        ep_successes = []
        ep_ret = np.zeros(2, dtype=np.float32)
        last_info = {}

        for _ in range(args.steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            joint_t = obs_t.reshape(1, -1)
            with torch.no_grad():
                dist = model.dist(obs_t)
                raw_action = dist.sample()
                action = torch.tanh(raw_action)
                logp = dist.log_prob(raw_action).sum(-1)
                value = model.value(joint_t).squeeze(0)
            action_np = action.cpu().numpy()
            if mode != "selfplay":
                action_np[1] = 0.0
            next_obs, reward, done, info = env.step(action_np)

            obs_buf.append(obs_t)
            joint_buf.append(joint_t.squeeze(0))
            action_buf.append(raw_action)
            logp_buf.append(logp)
            reward_buf.append(torch.tensor(reward, dtype=torch.float32, device=device))
            done_buf.append(torch.tensor([float(done), float(done)], dtype=torch.float32, device=device))
            value_buf.append(value)
            ep_ret += reward
            last_info = info
            obs = next_obs
            if done:
                ep_returns.append(ep_ret.copy())
                ep_successes.append(float(info.get("success", False)))
                ep_ret[:] = 0.0
                obs = env.reset()

        obs_batch = torch.stack(obs_buf)                 # T, 2, obs
        joint_batch = torch.stack(joint_buf)             # T, joint
        raw_action_batch = torch.stack(action_buf)       # T, 2, action
        old_logp = torch.stack(logp_buf).detach()        # T, 2
        rewards = torch.stack(reward_buf)
        dones = torch.stack(done_buf)
        values = torch.stack(value_buf).detach()
        with torch.no_grad():
            last_obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            last_value = model.value(last_obs_t.reshape(1, -1)).squeeze(0)
        adv, returns = gae(rewards, values, dones, last_value, 0.985, 0.92)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(args.epochs):
            dist = model.dist(obs_batch.reshape(-1, env.obs_dim))
            new_logp = dist.log_prob(raw_action_batch.reshape(-1, env.action_dim)).sum(-1).reshape(args.steps, 2)
            entropy = dist.entropy().sum(-1).mean()
            ratio = (new_logp - old_logp).exp()
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 0.8, 1.2) * adv
            actor_loss = -torch.min(unclipped, clipped).mean()
            value_pred = model.value(joint_batch)
            value_loss = 0.5 * (returns - value_pred).pow(2).mean()
            loss = actor_loss + value_loss - 0.01 * entropy
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()

        mean_ret = np.mean(ep_returns, axis=0) if ep_returns else ep_ret
        success_rate = float(np.mean(ep_successes)) if ep_successes else 0.0
        print(
            f"update={update:03d} mode={mode:8s} own_ret={mean_ret[0]:8.2f} enemy_ret={mean_ret[1]:8.2f} "
            f"dist={last_info.get('distance', 0.0):7.1f} rear={last_info.get('rear_angle_deg', 0.0):5.1f} "
            f"hold={last_info.get('rear_hold_time', 0.0):4.1f} episodes={len(ep_returns):2d} success_rate={success_rate:4.2f}"
        )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "obs_dim": env.obs_dim,
            "action_dim": env.action_dim,
            "seed": args.seed,
        },
        args.model,
    )
    print(f"saved MAPPO model to {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
