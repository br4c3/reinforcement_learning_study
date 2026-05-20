#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from dogfight_sim.rl_env import SixDofDogfightEnv, SixDofDogfightEnvConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--model", type=Path, default=Path("models/ppo_6dof_dogfight.zip"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.model.parent.mkdir(parents=True, exist_ok=True)
    env = Monitor(SixDofDogfightEnv(SixDofDogfightEnvConfig(seed=args.seed)))
    model = PPO("MlpPolicy", env, seed=args.seed, n_steps=512, batch_size=128, gamma=0.985, gae_lambda=0.92, ent_coef=0.01, verbose=1)
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(args.model)
    print(f"saved model to {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
