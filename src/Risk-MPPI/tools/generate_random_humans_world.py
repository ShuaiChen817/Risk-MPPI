#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据现有 three_walking_humans.world 的头部模板，
随机生成 N 个行走人物（actor），并覆盖写回该 world 文件。

特点：
- 每次运行都会重新随机生成 N_ACTORS 个 human_XX
- 轨迹类型混合：直线、斜线、矩形、近似椭圆、8 字形
- 人物始终“面朝运动方向”（yaw 按切向方向设置）

用法（在启动 Gazebo 之前先运行一次）：

  cd ~/turtlebot_ws
  source install/setup.bash
  python3 src/Risk-MPPI/tools/generate_random_humans_world.py
  ros2 launch nav2_risk_mppi world.launch.py
"""

import math
import random
from pathlib import Path
from typing import List, Tuple

N_ACTORS = 30 #人物数量

# 位置采样范围（平面上），避开原点附近的小车
XMIN, XMAX = -10.0, 10.0
YMIN, YMAX = -10.0, 10.0
ROBOT_EXCLUSION_RADIUS = 2.0

# 速度范围（m/s），用于按路径长度自动分配时间
SPEED_MIN = 0.2
SPEED_MAX = 0.5

# 直线轨迹的原地转身时间（s）
TURN_TIME = 2.5


def rand_speed() -> float:
    return random.uniform(SPEED_MIN, SPEED_MAX)


def rand_center() -> Tuple[float, float]:
    """采样一个不太靠近原点的中心点。"""
    for _ in range(200):
        x = random.uniform(XMIN, XMAX)
        y = random.uniform(YMIN, YMAX)
        if math.hypot(x, y) > ROBOT_EXCLUSION_RADIUS:
            return x, y
    # 退化情况就返回边界附近
    return random.uniform(XMIN, XMAX), random.uniform(YMIN, YMAX)


def waypoints_to_xml(waypoints: List[Tuple[float, float, float, float]]) -> str:
    """
    waypoints: 列表元素为 (t, x, y, yaw)。
    """
    lines = []
    for t, x, y, yaw in waypoints:
        lines.append("          <waypoint>")
        lines.append(f"            <time>{t:.3f}</time>")
        lines.append(f"            <pose>{x:.3f} {y:.3f} 0 0 0 {yaw:.5f}</pose>")
        lines.append("          </waypoint>")
    return "\n".join(lines)


def build_actor_block(name: str, waypoints: List[Tuple[float, float, float, float]]) -> str:
    traj_xml = waypoints_to_xml(waypoints)
    return f"""    <actor name='{name}'>
      <pose>0 0 1 0 0 0</pose>
      <skin>
        <filename>file://media/models/walk.dae</filename>
        <scale>1</scale>
      </skin>
      <animation name='walking'>
        <filename>file://media/models/walk.dae</filename>
        <interpolate_x>0</interpolate_x>
      </animation>
      <script>
        <loop>1</loop>
        <auto_start>1</auto_start>
        <trajectory id='0' type='walking'>
{traj_xml}
        </trajectory>
      </script>
    </actor>
"""


def gen_line_waypoints() -> List[Tuple[float, float, float, float]]:
    """
    生成一条直线往返轨迹（带原地转身段）。
    """
    cx, cy = rand_center()
    length = random.uniform(4.0, 10.0)
    theta = random.uniform(-math.pi, math.pi)
    dx = math.cos(theta)
    dy = math.sin(theta)
    # 两端点
    x1 = cx - 0.5 * length * dx
    y1 = cy - 0.5 * length * dy
    x2 = cx + 0.5 * length * dx
    y2 = cy + 0.5 * length * dy
    speed = rand_speed()
    travel_time = length / max(1e-3, speed)

    t0 = 0.0
    t1 = t0 + travel_time
    t2 = t1 + TURN_TIME
    t3 = t2 + travel_time
    t4 = t3 + TURN_TIME

    yaw_forward = theta
    yaw_backward = (theta + math.pi) if theta < math.pi else (theta - math.pi)

    return [
        (t0, x1, y1, yaw_forward),
        (t1, x2, y2, yaw_forward),
        (t2, x2, y2, yaw_backward),
        (t3, x1, y1, yaw_backward),
        (t4, x1, y1, yaw_forward),
    ]


def gen_diag_waypoints() -> List[Tuple[float, float, float, float]]:
    """
    生成一条对角线往返轨迹（类似 7/8 号，但参数随机）。
    """
    cx, cy = rand_center()
    half = random.uniform(1.5, 3.0)
    # 随机选 NE-SW 或 NW-SE 对角线
    if random.random() < 0.5:
        # NE <-> SW
        x1, y1 = cx - half, cy - half
        x2, y2 = cx + half, cy + half
        yaw_fwd = math.pi / 4      # NE
        yaw_bwd = -3 * math.pi / 4  # SW
    else:
        # NW <-> SE
        x1, y1 = cx - half, cy + half
        x2, y2 = cx + half, cy - half
        yaw_fwd = -math.pi / 4       # SE
        yaw_bwd = 3 * math.pi / 4    # NW

    length = math.hypot(x2 - x1, y2 - y1)
    speed = rand_speed()
    travel_time = length / max(1e-3, speed)

    t0 = 0.0
    t1 = t0 + travel_time
    t2 = t1 + TURN_TIME
    t3 = t2 + travel_time
    t4 = t3 + TURN_TIME

    return [
        (t0, x1, y1, yaw_fwd),
        (t1, x2, y2, yaw_fwd),
        (t2, x2, y2, yaw_bwd),
        (t3, x1, y1, yaw_bwd),
        (t4, x1, y1, yaw_fwd),
    ]


def gen_rect_waypoints() -> List[Tuple[float, float, float, float]]:
    """
    生成一个随机长方形绕圈轨迹（顺时针）。
    """
    cx, cy = rand_center()
    half_w = random.uniform(1.5, 3.0)
    half_h = random.uniform(1.0, 2.5)

    pts = [
        (cx - half_w, cy - half_h, 0.0),              # 左下, 朝 +x
        (cx + half_w, cy - half_h, math.pi / 2),      # 右下, 朝 +y
        (cx + half_w, cy + half_h, math.pi),          # 右上, 朝 -x
        (cx - half_w, cy + half_h, -math.pi / 2),     # 左上, 朝 -y
        (cx - half_w, cy - half_h, 0.0),              # 回到左下
    ]

    # 按总边长和随机速度分配时间
    edges = []
    total_len = 0.0
    for i in range(len(pts) - 1):
        x0, y0, _ = pts[i]
        x1, y1, _ = pts[i + 1]
        L = math.hypot(x1 - x0, y1 - y0)
        edges.append(L)
        total_len += L

    speed = rand_speed()
    total_time = total_len / max(1e-3, speed)

    times = [0.0]
    acc = 0.0
    for L in edges:
        acc += total_time * (L / total_len)
        times.append(acc)

    waypoints = []
    for (x, y, yaw), t in zip(pts, times):
        waypoints.append((t, x, y, yaw))
    return waypoints


def gen_ellipse_waypoints() -> List[Tuple[float, float, float, float]]:
    """
    生成一个近似椭圆轨迹（用多边形离散），一次绕圈。
    """
    cx, cy = rand_center()
    a = random.uniform(2.0, 4.0)  # x 方向半轴
    b = random.uniform(1.0, 3.0)  # y 方向半轴
    n = 16  # 离散点数

    # 先算周长近似，用于时间分配
    pts = []
    perim = 0.0
    for i in range(n):
        t = 2 * math.pi * i / n
        x = cx + a * math.cos(t)
        y = cy + b * math.sin(t)
        pts.append((x, y, t))
        if i > 0:
            perim += math.hypot(x - pts[i - 1][0], y - pts[i - 1][1])
    perim += math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1])

    speed = rand_speed()
    total_time = perim / max(1e-3, speed)

    waypoints = []
    t_acc = 0.0
    for i in range(n + 1):
        x0, y0, phi = pts[i % n]
        # 切向方向
        dx = -a * math.sin(phi)
        dy = b * math.cos(phi)
        yaw = math.atan2(dy, dx)
        if i > 0:
            x_prev, y_prev, _ = pts[(i - 1) % n]
            L = math.hypot(x0 - x_prev, y0 - y_prev)
            t_acc += total_time * (L / perim)
        waypoints.append((t_acc, x0, y0, yaw))
    return waypoints


def gen_figure8_waypoints() -> List[Tuple[float, float, float, float]]:
    """
    生成一个近似“8 字”轨迹，用简单的参数方程离散。
    x = a * sin(t), y = a * sin(t) * cos(t)
    """
    cx, cy = rand_center()
    a = random.uniform(1.5, 3.0)
    n = 24

    pts = []
    perim = 0.0
    for i in range(n):
        t = 2 * math.pi * i / n
        x = cx + a * math.sin(t)
        y = cy + a * math.sin(t) * math.cos(t)
        pts.append((x, y, t))
        if i > 0:
            perim += math.hypot(x - pts[i - 1][0], y - pts[i - 1][1])
    perim += math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1])

    speed = rand_speed()
    total_time = perim / max(1e-3, speed)

    waypoints = []
    t_acc = 0.0
    for i in range(n + 1):
        x0, y0, phi = pts[i % n]
        # 切向方向：对参数方程求导
        dx = a * math.cos(phi)
        dy = a * (math.cos(2 * phi) - math.sin(2 * phi)) / 2.0
        yaw = math.atan2(dy, dx)
        if i > 0:
            x_prev, y_prev, _ = pts[(i - 1) % n]
            L = math.hypot(x0 - x_prev, y0 - y_prev)
            t_acc += total_time * (L / perim)
        waypoints.append((t_acc, x0, y0, yaw))
    return waypoints


def generate_world_with_random_humans(world_path: Path, n_actors: int = N_ACTORS) -> None:
    text = world_path.read_text(encoding="utf-8")

    # 拆分出 world 头部（去掉原来的 <actor> 段）
    if "<actor" in text:
        header, _ = text.split("<actor", 1)
        header = header.rstrip() + "\n"
    else:
        # 没有 actor，就直接截到 </world> 之前
        header, _ = text.split("</world>", 1)
        header = header.rstrip() + "\n"

    # 生成随机 actor
    actor_blocks: List[str] = []
    for i in range(1, n_actors + 1):
        name = f"human_{i:02d}"
        mode = random.choice(["line", "diag", "rect", "ellipse", "figure8"])
        if mode == "line":
            wps = gen_line_waypoints()
        elif mode == "diag":
            wps = gen_diag_waypoints()
        elif mode == "rect":
            wps = gen_rect_waypoints()
        elif mode == "ellipse":
            wps = gen_ellipse_waypoints()
        else:
            wps = gen_figure8_waypoints()
        actor_blocks.append(build_actor_block(name, wps))

    actors_xml = "\n".join(actor_blocks)
    footer = "  </world>\n</sdf>\n"

    new_text = header + actors_xml + footer
    world_path.write_text(new_text, encoding="utf-8")
    print(f"[generate_random_humans_world] 写入随机 humans 共 {n_actors} 个到 {world_path}")


def main():
    # 假定脚本位置为 <ws_root>/src/Risk-MPPI/tools/ 下
    # 使用脚本所在包的资源目录，不依赖工作空间名称。
    package_root = Path(__file__).resolve().parents[1]
    world_path = package_root / "world" / "three_walking_humans.world"
    if not world_path.exists():
        raise FileNotFoundError(f"未找到 world 文件: {world_path}")

    # 使用系统随机种子；如需复现，可改为 random.seed(固定值)
    random.seed()
    generate_world_with_random_humans(world_path, N_ACTORS)


if __name__ == "__main__":
    main()
