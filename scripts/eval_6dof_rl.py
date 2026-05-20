#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stable_baselines3 import PPO

from dogfight_sim.rl_env import SixDofDogfightEnv, SixDofDogfightEnvConfig


def set_axes_equal(ax, xs, ys, zs) -> None:
    max_range = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0)
    mid_x = 0.5 * (max(xs) + min(xs))
    mid_y = 0.5 * (max(ys) + min(ys))
    mid_z = 0.5 * (max(zs) + min(zs))
    radius = 0.5 * max_range
    ax.set_xlim(mid_x - radius, mid_x + radius)
    ax.set_ylim(mid_y - radius, mid_y + radius)
    ax.set_zlim(max(0.0, mid_z - radius), mid_z + radius)


def write_final_plot(path: Path, rows: list[dict[str, float]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    own_x = [r["own_x"] for r in rows]
    own_y = [r["own_y"] for r in rows]
    own_z = [r["own_z"] for r in rows]
    enemy_x = [r["enemy_x"] for r in rows]
    enemy_y = [r["enemy_y"] for r in rows]
    enemy_z = [r["enemy_z"] for r in rows]
    last = rows[-1]

    fig = plt.figure(figsize=(11, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    ax.plot(enemy_x, enemy_y, enemy_z, color="#0891b2", linewidth=2.2, label="Enemy")
    ax.plot(own_x, own_y, own_z, color="#dc2626", linewidth=2.2, label="Ownship")
    ax.scatter([enemy_x[-1]], [enemy_y[-1]], [enemy_z[-1]], color="#0891b2", s=70)
    ax.scatter([own_x[-1]], [own_y[-1]], [own_z[-1]], color="#dc2626", s=70)
    ax.plot([own_x[-1], enemy_x[-1]], [own_y[-1], enemy_y[-1]], [own_z[-1], enemy_z[-1]], color="#6b7280", linestyle="--")
    ax.set_title(f"RL 6DoF Dogfight Final | distance={last['distance']:.0f} m | rear angle={last['rear_angle_deg']:.1f} deg", pad=18)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Altitude (m)")
    ax.view_init(elev=24, azim=-55)
    ax.grid(True)
    ax.legend(loc="upper left")
    set_axes_equal(ax, own_x + enemy_x, own_y + enemy_y, own_z + enemy_z)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"final 3D plot saved to {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/ppo_6dof_dogfight.zip"))
    parser.add_argument("--csv", type=Path, default=Path("dogfight_6dof_rl_trace.csv"))
    parser.add_argument("--final-png", type=Path, default=Path("dogfight_6dof_rl_final.png"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    env = SixDofDogfightEnv(SixDofDogfightEnvConfig(seed=args.seed))
    model = PPO.load(args.model)
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
