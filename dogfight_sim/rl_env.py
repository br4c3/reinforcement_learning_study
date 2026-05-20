from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .math3d import clamp, dot, norm, normalize, rotate_world_to_body, sub
from .sixdof import SixDofAircraft, SixDofState, chase_tail_control, level_turn_control, tactical_guidance_control


@dataclass
class SixDofDogfightEnvConfig:
    dt: float = 0.04
    max_time: float = 35.0
    rear_hold_win_time: float = 1.2
    kill_range_min: float = 100.0
    kill_range_max: float = 1050.0
    rear_cone_deg: float = 45.0
    align_cone_deg: float = 55.0
    static_spawn_min: float = 520.0
    static_spawn_max: float = 820.0
    circle_spawn_min: float = 800.0
    circle_spawn_max: float = 1150.0
    enemy_mode: str = "circle"
    mixed_enemy_modes: tuple[str, ...] = ("static", "circle", "weave")
    mixed_enemy_probs: tuple[float, ...] = (0.20, 0.45, 0.35)
    situation_reward_weight: float = 0.45
    safety_reward_weight: float = 0.35
    hierarchical_guidance: bool = False
    runaway_time: float = 8.0
    runaway_distance: float = 6500.0
    seed: int | None = None


class SixDofDogfightEnv(gym.Env):
    def __init__(self, config: SixDofDogfightEnvConfig | None = None) -> None:
        super().__init__()
        self.config = config or SixDofDogfightEnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.own = SixDofAircraft()
        self.enemy = SixDofAircraft()
        self.time = 0.0
        self.rear_hold = 0.0
        self.last_distance = 0.0
        self.enemy_target_alt = 1200.0
        self.enemy_circle_center = (0.0, 0.0)
        self.enemy_circle_phase = 0.0
        self.enemy_circle_radius = 1050.0
        self.current_enemy_mode = self.config.enemy_mode
        self.tactical_mode = "tail"
        self.last_situation = 0.0
        self.rear_hold_milestone = 0
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(20,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_enemy_mode = self._sample_enemy_mode()
        self.tactical_mode = "tail"
        own_y = float(self.rng.uniform(-180.0, 180.0))
        own_alt = float(self.rng.uniform(1050.0, 1350.0))
        enemy_alt = own_alt + float(self.rng.uniform(-90.0, 90.0))
        if self.current_enemy_mode == "static":
            enemy_x = float(self.rng.uniform(self.config.static_spawn_min, self.config.static_spawn_max))
        else:
            enemy_x = float(self.rng.uniform(self.config.circle_spawn_min, self.config.circle_spawn_max))
        self.own.reset(SixDofState.from_euler((0.0, own_y, own_alt), 160.0, 0.0, 0.0, float(self.rng.uniform(-0.15, 0.15))))
        self.enemy.reset(SixDofState.from_euler((enemy_x, 0.0, enemy_alt), 145.0, 0.0, 0.0, float(self.rng.uniform(-0.2, 0.2))))
        self.enemy_target_alt = enemy_alt
        self.enemy_circle_radius = float(self.rng.uniform(850.0, 1150.0))
        self.enemy_circle_phase = math.pi / 2.0
        self.enemy_circle_center = (enemy_x, -self.enemy_circle_radius)
        self.time = 0.0
        self.rear_hold = 0.0
        self.rear_hold_milestone = 0
        self.last_distance = self._metrics()[0]
        self.last_situation = self._situation_score(*self._metrics(), self.own.state.position[2], self.own.state.speed)
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action.astype(np.float64), -1.0, 1.0)
        pre_distance, pre_rear_angle, pre_align_angle = self._metrics()
        self.tactical_mode = self._select_tactical_mode(pre_distance, pre_rear_angle, pre_align_angle)
        if self.config.hierarchical_guidance:
            base = tactical_guidance_control(self.own, self.enemy, self.tactical_mode)
        else:
            base = chase_tail_control(self.own, self.enemy)
        control = (
            clamp(base[0] + 0.06 * float(action[0]), -1.0, 1.0),
            clamp(base[1] + 0.08 * float(action[1]), -1.0, 1.0),
            clamp(base[2] + 0.05 * float(action[2]), -1.0, 1.0),
            clamp(base[3] + 0.05 * float(action[3]), 0.0, 1.0),
        )
        self.own.step(control, self.config.dt)
        if self.current_enemy_mode == "static":
            self.enemy.step(level_turn_control(self.enemy, self.enemy_target_alt, target_roll_deg=0.0), self.config.dt)
        else:
            self._step_circle_enemy()
        self.time += self.config.dt

        distance, rear_angle, align_angle = self._metrics()
        in_range = self.config.kill_range_min <= distance <= self.config.kill_range_max
        in_rear = rear_angle < self.config.rear_cone_deg
        aligned = align_angle < self.config.align_cone_deg
        if in_range and in_rear and aligned:
            self.rear_hold += self.config.dt
        else:
            self.rear_hold = max(0.0, self.rear_hold - 0.25 * self.config.dt)
        milestone = int(min(3, math.floor((self.rear_hold + 1e-9) / 0.5)))

        tail_score = math.cos(math.radians(rear_angle))
        align_score = math.cos(math.radians(align_angle))
        range_score = math.exp(-abs(distance - 420.0) / 450.0)
        closing = clamp((self.last_distance - distance) / 30.0, -0.8, 0.8)
        self.last_distance = distance
        situation = self._situation_score(distance, rear_angle, align_angle, self.own.state.position[2], self.own.state.speed)
        situation_delta = clamp(situation - self.last_situation, -1.0, 1.0)
        self.last_situation = situation
        enemy_distance, enemy_rear_angle, enemy_align_angle = self._pair_metrics(self.enemy, self.own)
        enemy_has_shot = (
            self.config.kill_range_min <= enemy_distance <= self.config.kill_range_max
            and enemy_rear_angle < self.config.rear_cone_deg
            and enemy_align_angle < self.config.align_cone_deg
        )
        defensive_penalty = self.config.safety_reward_weight if enemy_has_shot else 0.0
        kill_bonus = 1.5 if in_range and in_rear else 0.0
        far_penalty = 0.0012 * max(0.0, distance - self.config.kill_range_max)
        low_alt_penalty = 1.5 if self.own.state.position[2] < 500.0 else 0.0
        reward = 1.5 * tail_score + 1.25 * align_score + 1.1 * range_score + 0.6 * closing + kill_bonus
        reward -= far_penalty + low_alt_penalty
        reward += self.config.situation_reward_weight * situation_delta - defensive_penalty
        reward += 1.6 * self.rear_hold - 0.02 * float(np.mean(action * action))
        if milestone > self.rear_hold_milestone:
            reward += 4.0 * float(milestone - self.rear_hold_milestone)
            self.rear_hold_milestone = milestone

        success = self.rear_hold >= self.config.rear_hold_win_time
        crashed = self.own.telemetry.crashed or self.enemy.telemetry.crashed
        runaway = self.time > self.config.runaway_time and distance > self.config.runaway_distance
        terminated = success or crashed
        truncated = runaway or self.time >= self.config.max_time
        if success:
            reward += 30.0
        if crashed:
            reward -= 80.0
        if runaway:
            reward -= 35.0

        info = {
            "distance": distance,
            "rear_angle_deg": rear_angle,
            "align_angle_deg": align_angle,
            "rear_hold_time": self.rear_hold,
            "success": success,
            "crashed": crashed,
            "runaway": runaway,
            "enemy_mode": self.current_enemy_mode,
            "situation_score": situation,
            "enemy_has_shot": enemy_has_shot,
            "tactical_mode": self.tactical_mode,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def _select_tactical_mode(self, distance: float, rear_angle: float, align_angle: float) -> str:
        if self.own.state.position[2] < 760.0 or self.own.state.speed < 115.0:
            return "recover"
        closing = self.last_distance - distance
        if distance < 260.0 or (distance < 520.0 and closing > 12.0 and self.own.state.speed > self.enemy.state.speed + 35.0):
            return "extend"
        if self.current_enemy_mode == "weave":
            if distance > 1800.0:
                return "lag"
            if distance < 850.0 and rear_angle < 75.0 and align_angle > 50.0:
                return "high_yoyo"
            if rear_angle < 105.0 or align_angle < 95.0:
                return "lag"
            return "lag"
        if distance < 850.0 and rear_angle < 70.0 and align_angle > 55.0:
            return "high_yoyo"
        if distance > 1650.0:
            return "lead"
        if rear_angle < 60.0 and distance < 1300.0:
            return "tail"
        if align_angle > 90.0:
            return "lead"
        return "lag"

    def _sample_enemy_mode(self) -> str:
        if self.config.enemy_mode != "mixed":
            return self.config.enemy_mode
        modes = self.config.mixed_enemy_modes
        probs = np.array(self.config.mixed_enemy_probs, dtype=np.float64)
        probs = probs / max(float(probs.sum()), 1e-12)
        return str(self.rng.choice(modes, p=probs))

    def _step_circle_enemy(self) -> None:
        weave = self.current_enemy_mode == "weave"
        speed = 145.0 + (18.0 * math.sin(0.7 * self.time) if weave else 0.0)
        omega = speed / self.enemy_circle_radius
        self.enemy_circle_phase -= omega * self.config.dt
        phase = self.enemy_circle_phase
        x = self.enemy_circle_center[0] + self.enemy_circle_radius * math.cos(phase)
        y = self.enemy_circle_center[1] + self.enemy_circle_radius * math.sin(phase)
        vx = speed * math.sin(phase)
        vy = -speed * math.cos(phase)
        vz = 0.0
        altitude = self.enemy_target_alt
        if weave:
            altitude += 85.0 * math.sin(0.45 * self.time)
            vz = 85.0 * 0.45 * math.cos(0.45 * self.time)
        yaw = math.atan2(vy, vx)
        pitch = math.atan2(vz, max(math.hypot(vx, vy), 1.0))
        state = SixDofState.from_euler((x, y, altitude), speed, 0.0, pitch, yaw)
        state.velocity = (vx, vy, vz)
        state.omega = (0.0, 0.0, -omega)
        self.enemy.reset(state)

    def _obs(self) -> np.ndarray:
        rel_pos = rotate_world_to_body(self.own.state.attitude, sub(self.enemy.state.position, self.own.state.position))
        rel_vel = rotate_world_to_body(self.own.state.attitude, sub(self.enemy.state.velocity, self.own.state.velocity))
        own_roll, own_pitch, own_yaw = self.own.state.euler
        enemy_roll, enemy_pitch, enemy_yaw = self.enemy.state.euler
        distance, rear_angle, align_angle = self._metrics()
        return np.clip(np.array([
            rel_pos[0] / 2500.0, rel_pos[1] / 2500.0, rel_pos[2] / 1200.0,
            rel_vel[0] / 350.0, rel_vel[1] / 350.0, rel_vel[2] / 350.0,
            self.own.state.speed / 300.0, self.enemy.state.speed / 300.0,
            own_roll / math.pi, own_pitch / math.pi, own_yaw / math.pi,
            enemy_roll / math.pi, enemy_pitch / math.pi, enemy_yaw / math.pi,
            self.own.state.omega[0] / math.radians(220.0),
            self.own.state.omega[1] / math.radians(220.0),
            self.own.state.omega[2] / math.radians(220.0),
            distance / 3000.0, rear_angle / 180.0, align_angle / 180.0,
        ], dtype=np.float32), -10.0, 10.0)

    def _metrics(self) -> tuple[float, float, float]:
        return self._pair_metrics(self.own, self.enemy)

    def _pair_metrics(self, ego: SixDofAircraft, other: SixDofAircraft) -> tuple[float, float, float]:
        ego_to_other = sub(other.state.position, ego.state.position)
        other_to_ego = sub(ego.state.position, other.state.position)
        distance = norm(ego_to_other)
        tail_score = clamp(-dot(other.state.forward, normalize(other_to_ego)), -1.0, 1.0)
        align_score = clamp(dot(ego.state.forward, normalize(ego_to_other)), -1.0, 1.0)
        return distance, math.degrees(math.acos(tail_score)), math.degrees(math.acos(align_score))

    def _situation_score(self, distance: float, rear_angle: float, align_angle: float, altitude: float, speed: float) -> float:
        rear = math.cos(math.radians(rear_angle))
        align = math.cos(math.radians(align_angle))
        range_score = math.exp(-abs(distance - 500.0) / 750.0)
        altitude_score = clamp((altitude - 450.0) / 850.0, -1.0, 1.0)
        speed_score = clamp((speed - 95.0) / 130.0, -1.0, 1.0)
        return 0.36 * rear + 0.30 * align + 0.22 * range_score + 0.07 * altitude_score + 0.05 * speed_score
