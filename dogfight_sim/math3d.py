from __future__ import annotations

import math
from typing import Tuple

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a: Vec3, scalar: float) -> Vec3:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Vec3) -> float:
    return math.sqrt(max(dot(a, a), 1e-12))


def normalize(a: Vec3) -> Vec3:
    n = norm(a)
    return (a[0] / n, a[1] / n, a[2] / n)


def quat_normalize(q: Quat) -> Quat:
    n = math.sqrt(max(sum(x * x for x in q), 1e-12))
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def quat_mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def euler_to_quat(roll: float, pitch: float, yaw: float) -> Quat:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return quat_normalize((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ))


def quat_to_euler(q: Quat) -> Vec3:
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (roll, pitch, yaw)


def rotate_body_to_world(q: Quat, v: Vec3) -> Vec3:
    w, x, y, z = q
    return (
        (1.0 - 2.0 * (y * y + z * z)) * v[0] + 2.0 * (x * y - z * w) * v[1] + 2.0 * (x * z + y * w) * v[2],
        2.0 * (x * y + z * w) * v[0] + (1.0 - 2.0 * (x * x + z * z)) * v[1] + 2.0 * (y * z - x * w) * v[2],
        2.0 * (x * z - y * w) * v[0] + 2.0 * (y * z + x * w) * v[1] + (1.0 - 2.0 * (x * x + y * y)) * v[2],
    )


def rotate_world_to_body(q: Quat, v: Vec3) -> Vec3:
    w, x, y, z = q
    return (
        (1.0 - 2.0 * (y * y + z * z)) * v[0] + 2.0 * (x * y + z * w) * v[1] + 2.0 * (x * z - y * w) * v[2],
        2.0 * (x * y - z * w) * v[0] + (1.0 - 2.0 * (x * x + z * z)) * v[1] + 2.0 * (y * z + x * w) * v[2],
        2.0 * (x * z + y * w) * v[0] + 2.0 * (y * z - x * w) * v[1] + (1.0 - 2.0 * (x * x + y * y)) * v[2],
    )
