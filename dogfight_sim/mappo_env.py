from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .math3d import clamp, dot, norm, normalize, rotate_world_to_body, sub
from .sixdof import SixDofAircraft, SixDofState, chase_tail_control, level_turn_control


@dataclass
class MappoDogfightConfig:
    dt: float = 0.04
    max_time: float = 35.0
    rear_hold_win_time: float = 3.0
    enemy_mode: str = "circle"
    seed: int | None = None


class MappoDogfightEnv:
    obs_dim = 20
    action_dim = 4

    def __init__(self, config: MappoDogfightConfig | None = None) -> None:
        self.config = config or MappoDogfightConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.own = SixDofAircraft()
        self.enemy = SixDofAircraft()
        self.time = 0.0
        self.rear_hold = 0.0
        self.last_distance = 0.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        own_y = float(self.rng.uniform(-180.0, 180.0))
        own_alt = float(self.rng.uniform(1050.0, 1350.0))
        enemy_alt = own_alt + float(self.rng.uniform(-90.0, 90.0))
        enemy_x = float(self.rng.uniform(650.0, 900.0)) if self.config.enemy_mode == "static" else float(self.rng.uniform(950.0, 1450.0))
        self.own.reset(SixDofState.from_euler((0.0, own_y, own_alt), 160.0, 0.0, 0.0, float(self.rng.uniform(-0.15, 0.15))))
        self.enemy.reset(SixDofState.from_euler((enemy_x, 0.0, enemy_alt), 145.0, 0.0, 0.0, float(self.rng.uniform(-0.2, 0.2))))
        self.time = 0.0
        self.rear_hold = 0.0
        self.last_distance = self.metrics()[0]
        return self.obs()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict[str, float | bool]]:
        actions = np.clip(actions.astype(np.float64), -1.0, 1.0)
        own_base = chase_tail_control(self.own, self.enemy)
        own_control = self._residual_control(own_base, actions[0])
        enemy_control = self._enemy_control(actions[1])

        self.own.step(own_control, self.config.dt)
        self.enemy.step(enemy_control, self.config.dt)
        self.time += self.config.dt

        distance, rear_angle, align_angle = self.metrics()
        in_kill = 120.0 <= distance <= 900.0 and rear_angle < 38.0 and align_angle < 48.0
        if in_kill:
            self.rear_hold += self.config.dt
        else:
            self.rear_hold = max(0.0, self.rear_hold - 0.5 * self.config.dt)

        tail_score = math.cos(math.radians(rear_angle))
        align_score = math.cos(math.radians(align_angle))
        range_score = math.exp(-abs(distance - 450.0) / 700.0)
        closing = clamp((self.last_distance - distance) / 40.0, -0.5, 0.5)
        self.last_distance = distance

        own_reward = 1.2 * tail_score + 0.7 * align_score + 0.35 * range_score + 0.3 * closing
        own_reward += 0.9 * self.rear_hold - 0.03 * float(np.mean(actions[0] * actions[0]))
        own_reward -= 0.0012 * max(0.0, distance - 1200.0)
        if self.own.state.position[2] < 500.0:
            own_reward -= 1.5
        if self.config.enemy_mode == "selfplay":
            enemy_reward = -own_reward - 0.02 * float(np.mean(actions[1] * actions[1]))
        else:
            enemy_reward = 0.0

        success = self.rear_hold >= self.config.rear_hold_win_time
        crashed = self.own.telemetry.crashed or self.enemy.telemetry.crashed
        runaway = self.time > 8.0 and distance > 6500.0
        done = success or crashed or runaway or self.time >= self.config.max_time
        if success:
            own_reward += 30.0
            enemy_reward -= 30.0
        if crashed:
            own_reward -= 30.0
            enemy_reward -= 5.0
        if runaway:
            own_reward -= 20.0
            enemy_reward += 10.0

        info = {
            "distance": distance,
            "rear_angle_deg": rear_angle,
            "align_angle_deg": align_angle,
            "rear_hold_time": self.rear_hold,
            "success": success,
            "crashed": crashed,
            "runaway": runaway,
        }
        return self.obs(), np.array([own_reward, enemy_reward], dtype=np.float32), done, info

    def obs(self) -> np.ndarray:
        return np.stack([
            self._agent_obs(self.own, self.enemy),
            self._agent_obs(self.enemy, self.own),
        ]).astype(np.float32)

    def metrics(self) -> tuple[float, float, float]:
        return self._pair_metrics(self.own, self.enemy)

    def _pair_metrics(self, ego: SixDofAircraft, other: SixDofAircraft) -> tuple[float, float, float]:
        ego_to_other = sub(other.state.position, ego.state.position)
        other_to_ego = sub(ego.state.position, other.state.position)
        distance = norm(ego_to_other)
        tail_score = clamp(-dot(other.state.forward, normalize(other_to_ego)), -1.0, 1.0)
        align_score = clamp(dot(ego.state.forward, normalize(ego_to_other)), -1.0, 1.0)
        return distance, math.degrees(math.acos(tail_score)), math.degrees(math.acos(align_score))

    def _agent_obs(self, ego: SixDofAircraft, other: SixDofAircraft) -> np.ndarray:
        rel_pos = rotate_world_to_body(ego.state.attitude, sub(other.state.position, ego.state.position))
        rel_vel = rotate_world_to_body(ego.state.attitude, sub(other.state.velocity, ego.state.velocity))
        ego_roll, ego_pitch, ego_yaw = ego.state.euler
        other_roll, other_pitch, other_yaw = other.state.euler
        distance, rear_angle, align_angle = self._pair_metrics(ego, other)
        return np.clip(np.array([
            rel_pos[0] / 2500.0, rel_pos[1] / 2500.0, rel_pos[2] / 1200.0,
            rel_vel[0] / 350.0, rel_vel[1] / 350.0, rel_vel[2] / 350.0,
            ego.state.speed / 300.0, other.state.speed / 300.0,
            ego_roll / math.pi, ego_pitch / math.pi, ego_yaw / math.pi,
            other_roll / math.pi, other_pitch / math.pi, other_yaw / math.pi,
            ego.state.omega[0] / math.radians(220.0),
            ego.state.omega[1] / math.radians(220.0),
            ego.state.omega[2] / math.radians(220.0),
            distance / 3000.0, rear_angle / 180.0, align_angle / 180.0,
        ], dtype=np.float32), -10.0, 10.0)

    def _residual_control(self, base: tuple[float, float, float, float], action: np.ndarray) -> tuple[float, float, float, float]:
        return (
            clamp(base[0] + 0.06 * float(action[0]), -1.0, 1.0),
            clamp(base[1] + 0.08 * float(action[1]), -1.0, 1.0),
            clamp(base[2] + 0.05 * float(action[2]), -1.0, 1.0),
            clamp(base[3] + 0.05 * float(action[3]), 0.0, 1.0),
        )

    def _enemy_control(self, action: np.ndarray) -> tuple[float, float, float, float]:
        if self.config.enemy_mode == "static":
            return (0.0, 0.0, 0.0, 0.55)
        base = level_turn_control(self.enemy)
        if self.config.enemy_mode == "circle":
            return base
        return self._residual_control(base, action)
