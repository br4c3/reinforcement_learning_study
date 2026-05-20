#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stable_baselines3 import SAC

from dogfight_sim.rl_env import SixDofDogfightEnv, SixDofDogfightEnvConfig
from scripts.eval_6dof_rl import write_final_plot
from scripts.train_sac_6dof import apply_difficulty


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/sac_6dof_dogfight.zip"))
    parser.add_argument("--csv", type=Path, default=Path("dogfight_6dof_sac_trace.csv"))
    parser.add_argument("--final-png", type=Path, default=Path("dogfight_6dof_sac_final.png"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--enemy-mode", choices=["static", "circle", "weave", "mixed"], default="circle")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="hard")
    parser.add_argument("--hierarchical-guidance", action="store_true")
    args = parser.parse_args()

    config = SixDofDogfightEnvConfig(
        seed=args.seed,
        enemy_mode=args.enemy_mode,
        hierarchical_guidance=args.hierarchical_guidance,
    )
    apply_difficulty(config, args.difficulty)
    env = SixDofDogfightEnv(config)
    model = SAC.load(args.model)
    obs, _ = env.reset(seed=args.seed)
    rows = []
    done = False
    final_info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        final_info = info
        rows.append({
            "time": env.time,
            "own_x": env.own.state.position[0],
            "own_y": env.own.state.position[1],
            "own_z": env.own.state.position[2],
            "enemy_x": env.enemy.state.position[0],
            "enemy_y": env.enemy.state.position[1],
            "enemy_z": env.enemy.state.position[2],
            "distance": info["distance"],
            "rear_angle_deg": info["rear_angle_deg"],
            "align_angle_deg": info["align_angle_deg"],
            "rear_hold_time": info["rear_hold_time"],
            "tactical_mode": info.get("tactical_mode", ""),
        })

    with args.csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_final_plot(args.final_png, rows)
    print(
        f"success={final_info.get('success')} distance={final_info.get('distance'):.1f}m "
        f"rear_angle={final_info.get('rear_angle_deg'):.1f}deg rear_hold={final_info.get('rear_hold_time'):.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
