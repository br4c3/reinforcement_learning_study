from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

from .math3d import (
    Vec3,
    add,
    clamp,
    cross,
    dot,
    euler_to_quat,
    mul,
    norm,
    normalize,
    quat_mul,
    quat_normalize,
    quat_to_euler,
    rotate_body_to_world,
    rotate_world_to_body,
    sub,
    wrap_pi,
)

Control = Tuple[float, float, float, float]


@dataclass
class SixDofState:
    position: Vec3
    velocity: Vec3
    attitude: Tuple[float, float, float, float]
    omega: Vec3

    @classmethod
    def from_euler(cls, position: Vec3, speed: float, roll: float, pitch: float, yaw: float) -> "SixDofState":
        q = euler_to_quat(roll, pitch, yaw)
        return cls(position, rotate_body_to_world(q, (speed, 0.0, 0.0)), q, (0.0, 0.0, 0.0))

    @property
    def euler(self) -> Vec3:
        return quat_to_euler(self.attitude)

    @property
    def forward(self) -> Vec3:
        return normalize(rotate_body_to_world(self.attitude, (1.0, 0.0, 0.0)))

    @property
    def speed(self) -> float:
        return norm(self.velocity)


@dataclass
class SixDofTelemetry:
    alpha: float = 0.0
    beta: float = 0.0
    airspeed: float = 160.0
    crashed: bool = False
    stalled: bool = False


class SixDofAircraft:
    def __init__(self) -> None:
        self.mass = 9200.0
        self.wing_area = 27.87
        self.span = 9.45
        self.chord = 3.45
        self.inertia = (12875.0, 75673.0, 85552.0)
        self.max_alpha = math.radians(28.0)
        self.max_body_rate = math.radians(220.0)
        self.max_body_accel = math.radians(420.0)
        self.state = SixDofState.from_euler((0.0, 0.0, 1200.0), 160.0, 0.0, 0.0, 0.0)
        self.telemetry = SixDofTelemetry()

    def reset(self, state: SixDofState) -> None:
        self.state = state
        self.telemetry = SixDofTelemetry(airspeed=state.speed)

    def step(self, control: Control, dt: float) -> None:
        elevator, aileron, rudder, throttle = control
        elevator = clamp(elevator, -1.0, 1.0)
        aileron = clamp(aileron, -1.0, 1.0)
        rudder = clamp(rudder, -1.0, 1.0)
        throttle = clamp(throttle, 0.0, 1.0)

        body_vel = rotate_world_to_body(self.state.attitude, self.state.velocity)
        u, v, w = body_vel
        speed = max(norm(body_vel), 1.0)
        alpha = clamp(math.atan2(w, max(abs(u), 1e-6)), -self.max_alpha, self.max_alpha)
        beta = clamp(math.asin(clamp(v / speed, -1.0, 1.0)), -math.radians(25.0), math.radians(25.0))

        qbar = 0.5 * 1.225 * speed * speed
        cl = 0.18 + 4.7 * alpha + 0.55 * elevator
        cd = 0.028 + 0.42 * alpha * alpha + 0.015 * abs(elevator)
        cy = -0.85 * beta + 0.22 * rudder

        lift = qbar * self.wing_area * cl
        drag = qbar * self.wing_area * cd
        side = qbar * self.wing_area * cy
        thrust = 76000.0 * throttle
        force_body = (
            thrust - drag * math.cos(alpha) + lift * math.sin(alpha),
            side,
            lift * math.cos(alpha) + drag * math.sin(alpha),
        )
        force_world = rotate_body_to_world(self.state.attitude, force_body)
        accel = (force_world[0] / self.mass, force_world[1] / self.mass, force_world[2] / self.mass - 9.80665)

        p, q, r = self.state.omega
        p_hat = p * self.span / (2.0 * speed)
        q_hat = q * self.chord / (2.0 * speed)
        r_hat = r * self.span / (2.0 * speed)
        roll_m = qbar * self.wing_area * self.span * (0.085 * aileron - 0.48 * p_hat)
        pitch_m = qbar * self.wing_area * self.chord * (-0.105 * elevator - 0.82 * q_hat)
        yaw_m = qbar * self.wing_area * self.span * (0.035 * rudder - 0.34 * r_hat)

        ix, iy, iz = self.inertia
        gyro = cross(self.state.omega, (ix * p, iy * q, iz * r))
        omega_dot = (
            clamp((roll_m - gyro[0]) / ix, -self.max_body_accel, self.max_body_accel),
            clamp((pitch_m - gyro[1]) / iy, -self.max_body_accel, self.max_body_accel),
            clamp((yaw_m - gyro[2]) / iz, -self.max_body_accel, self.max_body_accel),
        )

        vel = add(self.state.velocity, mul(accel, dt))
        vel_speed = norm(vel)
        vel = mul(normalize(vel), clamp(vel_speed, 60.0, 430.0))
        pos = add(self.state.position, mul(vel, dt))
        omega = add(self.state.omega, mul(omega_dot, dt))
        omega = tuple(clamp(x, -self.max_body_rate, self.max_body_rate) for x in omega)
        qdot = quat_mul(self.state.attitude, (0.0, omega[0], omega[1], omega[2]))
        attitude = quat_normalize(tuple(self.state.attitude[i] + 0.5 * qdot[i] * dt for i in range(4)))

        self.state = SixDofState(pos, vel, attitude, omega)
        self.telemetry = SixDofTelemetry(alpha, beta, self.state.speed, pos[2] <= 0.0, abs(alpha) > self.max_alpha * 0.98)


def chase_tail_control(own: SixDofAircraft, enemy: SixDofAircraft) -> Control:
    own_to_enemy = sub(enemy.state.position, own.state.position)
    distance = norm(own_to_enemy)
    tail_distance = clamp(520.0 + 0.08 * distance, 520.0, 680.0)
    target = sub(enemy.state.position, mul(enemy.state.forward, tail_distance))
    return track_point_control(own, target, distance)


def tactical_guidance_control(own: SixDofAircraft, enemy: SixDofAircraft, mode: str) -> Control:
    own_to_enemy = sub(enemy.state.position, own.state.position)
    distance = norm(own_to_enemy)
    enemy_right = normalize(rotate_body_to_world(enemy.state.attitude, (0.0, 1.0, 0.0)))
    turn_sign = 1.0 if enemy.state.omega[2] >= 0.0 else -1.0
    if abs(enemy.state.omega[2]) < math.radians(2.0):
        turn_sign = 1.0 if dot(own_to_enemy, enemy_right) >= 0.0 else -1.0

    tail_distance = clamp(520.0 + 0.08 * distance, 520.0, 720.0)
    lead_time = clamp(distance / 620.0, 0.35, 2.2)
    if mode == "lead":
        target = add(enemy.state.position, mul(enemy.state.velocity, lead_time))
        return track_point_control(own, target, distance, throttle_bias=0.10)
    if mode == "rejoin":
        tail = sub(enemy.state.position, mul(enemy.state.forward, tail_distance + 230.0))
        target = add(tail, (0.0, 0.0, 80.0))
        return track_point_control(own, target, distance, throttle_bias=0.20)
    if mode == "lag":
        tail = sub(enemy.state.position, mul(enemy.state.forward, tail_distance + 170.0))
        target = sub(tail, mul(enemy_right, 280.0 * turn_sign))
        throttle_bias = 0.16 if distance > 1800.0 else -0.08
        return track_point_control(own, target, distance, throttle_bias=throttle_bias)
    if mode == "high_yoyo":
        tail = sub(enemy.state.position, mul(enemy.state.forward, tail_distance + 80.0))
        target = add(tail, (0.0, 0.0, 260.0))
        return track_point_control(own, target, distance, throttle_bias=-0.18)
    if mode == "extend":
        target = add(add(own.state.position, mul(own.state.forward, 900.0)), (0.0, 0.0, 160.0))
        return track_point_control(own, target, distance, throttle_bias=0.22)
    if mode == "recover":
        target = add(add(own.state.position, mul(own.state.forward, 700.0)), (0.0, 0.0, 360.0))
        return track_point_control(own, target, distance, throttle_bias=0.35)
    return chase_tail_control(own, enemy)


def track_point_control(
    own: SixDofAircraft,
    target: Vec3,
    distance: float,
    throttle_bias: float = 0.0,
) -> Control:
    body_rel = rotate_world_to_body(own.state.attitude, sub(target, own.state.position))
    desired_yaw = math.atan2(body_rel[1], max(body_rel[0], 1.0))
    desired_pitch = math.atan2(body_rel[2], max(abs(body_rel[0]), 1.0))
    altitude_guard = 0.0
    if own.state.position[2] < 950.0:
        altitude_guard += clamp((950.0 - own.state.position[2]) / 750.0, 0.0, math.radians(22.0))
    if own.state.velocity[2] < -20.0:
        altitude_guard += clamp(-own.state.velocity[2] / 750.0, 0.0, math.radians(12.0))
    desired_pitch = max(desired_pitch, altitude_guard)
    if distance < 1300.0 and own.state.speed > 285.0:
        energy_guard = clamp((own.state.speed - 285.0) / 480.0, 0.0, math.radians(18.0))
        desired_pitch = max(desired_pitch, energy_guard)
    roll, pitch, _ = own.state.euler
    p, q, r = own.state.omega
    target_roll = clamp(2.6 * desired_yaw, -math.radians(70.0), math.radians(70.0))
    throttle = clamp(0.56 + norm(body_rel) / 2500.0 + throttle_bias, 0.28, 1.0)
    if own.state.speed > 280.0 and distance < 1300.0:
        throttle = min(throttle, 0.35)
    if own.state.speed > 340.0 and distance < 1200.0:
        throttle = min(throttle, 0.12)
    if own.state.position[2] < 700.0 or own.state.speed < 125.0:
        throttle = max(throttle, 0.85)
    return (
        clamp(-2.2 * (desired_pitch - pitch) - 0.55 * q, -1.0, 1.0),
        clamp(1.7 * wrap_pi(target_roll - roll) - 0.28 * p, -1.0, 1.0),
        clamp(0.45 * desired_yaw - 0.18 * r, -1.0, 1.0),
        throttle,
    )


def level_turn_control(
    aircraft: SixDofAircraft,
    target_altitude: float | None = None,
    target_roll_deg: float = 38.0,
) -> Control:
    roll, pitch, _ = aircraft.state.euler
    p, q, r = aircraft.state.omega
    if target_altitude is None:
        target_pitch = math.radians(4.0)
    else:
        altitude_error = clamp((target_altitude - aircraft.state.position[2]) / 700.0, math.radians(-10.0), math.radians(14.0))
        vertical_damping = clamp(aircraft.state.velocity[2] / 150.0, math.radians(-10.0), math.radians(10.0))
        target_pitch = clamp(math.radians(3.0) + altitude_error - vertical_damping, math.radians(-8.0), math.radians(14.0))
    return (
        clamp(-1.25 * (target_pitch - pitch) - 0.45 * q, -1.0, 1.0),
        clamp(1.45 * wrap_pi(math.radians(target_roll_deg) - roll) - 0.25 * p, -1.0, 1.0),
        clamp(-0.12 * r, -1.0, 1.0),
        0.76 if abs(target_roll_deg) > 1.0 else 0.60,
    )
