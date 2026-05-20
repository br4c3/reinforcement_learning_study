#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import FloatSchedule, update_learning_rate

from dogfight_sim.rl_env import SixDofDogfightEnv, SixDofDogfightEnvConfig


class DogfightEvalCallback(BaseCallback):
    def __init__(
        self,
        save_path: Path,
        enemy_modes: list[str],
        difficulty: str,
        eval_freq: int,
        seeds: list[int],
        hierarchical_guidance: bool,
    ) -> None:
        super().__init__()
        self.save_path = save_path
        self.enemy_modes = enemy_modes
        self.difficulty = difficulty
        self.eval_freq = eval_freq
        self.seeds = seeds
        self.hierarchical_guidance = hierarchical_guidance
        self.best_score = float("-inf")

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        results = [self._run_episode(seed, mode) for mode in self.enemy_modes for seed in self.seeds]
        successes = [float(result["success"]) for result in results]
        rear_angles = [result["rear_angle_deg"] for result in results]
        hold_times = [result["rear_hold_time"] for result in results]
        distances = [result["distance"] for result in results]
        score = (
            100.0 * (sum(successes) / len(successes))
            + 0.25 * (sum(hold_times) / len(hold_times))
            - 0.01 * (sum(rear_angles) / len(rear_angles))
            - 0.0005 * (sum(distances) / len(distances))
        )
        print(
            f"eval steps={self.num_timesteps} score={score:.3f} "
            f"success_rate={sum(successes) / len(successes):.2f} "
            f"rear={sum(rear_angles) / len(rear_angles):.1f} "
            f"hold={sum(hold_times) / len(hold_times):.2f} "
            f"distance={sum(distances) / len(distances):.1f} "
            f"modes={','.join(self.enemy_modes)}"
        )
        if score > self.best_score:
            self.best_score = score
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.save_path)
            print(f"saved best SAC model to {self.save_path}")
        return True

    def _run_episode(self, seed: int, enemy_mode: str) -> dict[str, float | bool]:
        config = SixDofDogfightEnvConfig(
            seed=seed,
            enemy_mode=enemy_mode,
            hierarchical_guidance=self.hierarchical_guidance,
        )
        apply_difficulty(config, self.difficulty)
        env = SixDofDogfightEnv(config)
        obs, _ = env.reset(seed=seed)
        done = False
        info: dict[str, float | bool] = {}
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        return info


def apply_difficulty(config: SixDofDogfightEnvConfig, difficulty: str) -> None:
    if difficulty == "easy":
        config.rear_hold_win_time = 1.2
        config.kill_range_max = 1050.0
        config.rear_cone_deg = 45.0
        config.align_cone_deg = 55.0
        config.static_spawn_min = 520.0
        config.static_spawn_max = 820.0
        config.circle_spawn_min = 800.0
        config.circle_spawn_max = 1150.0
    elif difficulty == "medium":
        config.rear_hold_win_time = 1.6
        config.kill_range_max = 1000.0
        config.rear_cone_deg = 42.0
        config.align_cone_deg = 52.0
        config.static_spawn_min = 600.0
        config.static_spawn_max = 900.0
        config.circle_spawn_min = 850.0
        config.circle_spawn_max = 1200.0
    elif difficulty == "hard":
        config.rear_hold_win_time = 2.0
        config.kill_range_max = 1150.0
        config.rear_cone_deg = 40.0
        config.align_cone_deg = 50.0
        config.static_spawn_min = 650.0
        config.static_spawn_max = 1000.0
        config.circle_spawn_min = 950.0
        config.circle_spawn_max = 1300.0
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--model", type=Path, default=Path("models/sac_6dof_dogfight.zip"))
    parser.add_argument("--load-model", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--enemy-mode", choices=["static", "circle", "weave", "mixed"], default="circle")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="hard")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--best-model", type=Path, default=None)
    parser.add_argument("--eval-freq", type=int, default=0)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[17, 23, 31])
    parser.add_argument("--eval-enemy-mode", choices=["static", "circle", "weave", "mixed"], default=None)
    parser.add_argument("--eval-enemy-modes", choices=["static", "circle", "weave", "mixed"], nargs="+", default=None)
    parser.add_argument("--hierarchical-guidance", action="store_true")
    args = parser.parse_args()

    args.model.parent.mkdir(parents=True, exist_ok=True)
    first_mode = args.enemy_mode if args.no_curriculum else "static"
    base_config = SixDofDogfightEnvConfig(
        seed=args.seed,
        enemy_mode=first_mode,
        hierarchical_guidance=args.hierarchical_guidance,
    )
    apply_difficulty(base_config, "easy")
    env = Monitor(SixDofDogfightEnv(base_config))
    if args.load_model is not None:
        model = SAC.load(args.load_model, env=env)
        if args.learning_rate is not None:
            model.learning_rate = args.learning_rate
            model.lr_schedule = FloatSchedule(args.learning_rate)
            update_learning_rate(model.actor.optimizer, args.learning_rate)
            update_learning_rate(model.critic.optimizer, args.learning_rate)
            if model.ent_coef_optimizer is not None:
                update_learning_rate(model.ent_coef_optimizer, args.learning_rate)
    else:
        model = SAC(
            "MlpPolicy",
            env,
            seed=args.seed,
            learning_rate=args.learning_rate or 3e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=256,
            tau=0.02,
            gamma=0.985,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            verbose=1,
        )
    callback = None
    if args.best_model is not None or args.eval_freq > 0:
        best_model = args.best_model or args.model.with_name(f"{args.model.stem}_best.zip")
        callback = DogfightEvalCallback(
            best_model,
            args.eval_enemy_modes or [args.eval_enemy_mode or args.enemy_mode],
            args.difficulty,
            args.eval_freq or max(1, args.timesteps // 10),
            args.eval_seeds,
            args.hierarchical_guidance,
        )
    if args.no_curriculum:
        env.env.config.enemy_mode = args.enemy_mode
        apply_difficulty(env.env.config, args.difficulty)
        env.reset()
        model.learn(total_timesteps=args.timesteps, progress_bar=False, callback=callback)
    else:
        stages = [
            ("static", "easy", 0.20),
            ("circle", "easy", 0.25),
            ("circle", "medium", 0.25),
            ("mixed", "hard", 0.15),
            ("weave", "hard", 0.15),
        ]
        used_steps = 0
        for idx, (mode, difficulty, fraction) in enumerate(stages):
            if idx == len(stages) - 1:
                stage_steps = args.timesteps - used_steps
            else:
                stage_steps = int(args.timesteps * fraction)
            used_steps += stage_steps
            env.env.config.enemy_mode = mode
            apply_difficulty(env.env.config, difficulty)
            env.reset()
            print(f"curriculum stage={idx + 1} enemy={mode} difficulty={difficulty} timesteps={stage_steps}")
            model.learn(
                total_timesteps=stage_steps,
                reset_num_timesteps=(idx == 0),
                progress_bar=False,
                callback=callback,
            )
    model.save(args.model)
    print(f"saved SAC model to {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
