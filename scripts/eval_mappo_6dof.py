#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from dogfight_sim.mappo_env import MappoDogfightEnv, MappoDogfightConfig
from scripts.eval_6dof_rl import write_final_plot
from scripts.train_mappo_6dof import ActorCritic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/mappo_6dof_dogfight.pt"))
    parser.add_argument("--csv", type=Path, default=Path("dogfight_6dof_mappo_trace.csv"))
    parser.add_argument("--final-png", type=Path, default=Path("dogfight_6dof_mappo_final.png"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--enemy-mode", choices=["static", "circle", "selfplay"], default="circle")
    args = parser.parse_args()

    env = MappoDogfightEnv(MappoDogfightConfig(seed=args.seed, enemy_mode=args.enemy_mode))
    checkpoint = torch.load(args.model, map_location="cpu")
    model = ActorCritic(checkpoint["obs_dim"], checkpoint["obs_dim"] * 2, checkpoint["action_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    obs = env.reset(seed=args.seed)
    rows = []
    done = False
    info = {}
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = torch.tanh(model.actor(obs_t)).numpy()
            if args.enemy_mode != "selfplay":
                action[1] = 0.0
        obs, _, done, info = env.step(action)
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
        })

    with args.csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_final_plot(args.final_png, rows)
    print(
        f"success={info.get('success')} distance={info.get('distance'):.1f}m "
        f"rear_angle={info.get('rear_angle_deg'):.1f}deg rear_hold={info.get('rear_hold_time'):.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
