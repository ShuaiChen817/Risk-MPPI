#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 2 Humble + Gazebo Classic
批量生成动态障碍：
  1) 默认：圆柱（沿椭圆轨迹，用 SetEntityState“滑动”）
  2) --humans：人形 Actor（<script> 轨迹 + 行走动画，不再滑行）
  3) --humans --sfm：人形 + 社会力模型插件（需要编译 gazebo_sfm_plugin）

注意：
- 人形 Actor 由 Gazebo 自己驱动行走，因此本脚本不再给 Actor 开“移动定时器”。
- 请确保：
  - GAZEBO_MODEL_PATH 能找到你的人形模型（例如 person_walking）
  - 若使用 --sfm，GAZEBO_PLUGIN_PATH 能找到 libPedestrianSFMPlugin.so
"""

import math
import random
import time
import argparse
from typing import List, Tuple

import rclpy
from rclpy.node import Node

# Gazebo Classic (gazebo_ros) ROS 2 接口
from gazebo_msgs.srv import SpawnEntity, SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Twist


# ------------------------------
# 生成 SDF：人形 Actor（内置脚本轨迹）
# ------------------------------
def make_actor_xml_walk(
    name: str,
    mesh_uri: str,
    center,
    ax, by,
    period=12.0,
    n_pts=32,
    z=1.0,
    scale=1.0
) -> str:
    import math
    cx, cy = center
    wps = []
    for k in range(n_pts):
        t = period * k / n_pts
        th = 2.0 * math.pi * k / n_pts
        x = cx + ax * math.cos(th)
        y = cy + by * math.sin(th)
        wps.append(f"        <waypoint>{t:.3f} {x:.3f} {y:.3f} {z:.3f}</waypoint>")
    waypoints_xml = "\n".join(wps)
    return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <actor name="{name}">
    <pose>{cx:.3f} {cy:.3f} {z:.3f} 0 0 0</pose>
    <skin><filename>{mesh_uri}</filename><scale>{scale:.3f}</scale></skin>
    <animation name="walking">
      <filename>{mesh_uri}</filename>
      <interpolate_x>true</interpolate_x>
    </animation>
    <script>
      <loop>true</loop>
      <auto_start>true</auto_start>
      <trajectory id="0" type="walk">
        <use_world_frame>true</use_world_frame>
{waypoints_xml}
      </trajectory>
    </script>
  </actor>
</sdf>"""


# ------------------------------
# 生成 SDF：人形 Actor（社会力 SFM 插件）
# ------------------------------
def make_actor_xml_with_sfm(
    name: str,
    mesh_uri: str,
    center: Tuple[float, float],
    ax: float,
    by: float,
    z: float = 1.0,
    scale: float = 1.0,
    velocity: float = 0.9,
    radius: float = 0.35,
    goal_weight: float = 2.0,
    obstacle_weight: float = 80.0,
    social_weight: float = 15.0,
) -> str:
    cx, cy = center
    # 用一个小矩形巡逻路点作为“目标点”；实际避让由 SFM 决定
    pts = [
        (cx - ax, cy - by, z),
        (cx - ax, cy + by, z),
        (cx + ax, cy + by, z),
        (cx + ax, cy - by, z),
    ]
    wps = "\n".join([f"        <waypoint>{x:.3f} {y:.3f} {z:.3f}</waypoint>" for x, y, z in pts])

    return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <actor name="{name}">
    <skin>
      <filename>{mesh_uri}</filename>
      <scale>{scale:.3f}</scale>
    </skin>
    <animation name="walking">
      <filename>{mesh_uri}</filename>
    </animation>
    <plugin name="{name}_sfm" filename="libPedestrianSFMPlugin.so">
      <velocity>{velocity:.3f}</velocity>
      <radius>{radius:.3f}</radius>
      <goal_weight>{goal_weight:.3f}</goal_weight>
      <obstacle_weight>{obstacle_weight:.3f}</obstacle_weight>
      <social_weight>{social_weight:.3f}</social_weight>
      <trajectory>
        <cyclic>true</cyclic>
{wps}
      </trajectory>
    </plugin>
  </actor>
</sdf>"""


# ------------------------------
# 生成 SDF：圆柱模型（作为动态障碍“滑动”）
# ------------------------------
def make_cylinder_sdf(name: str, radius: float = 0.2, height: float = 1.2) -> str:
    return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <pose>0 0 {height/2:.3f} 0 0 0</pose>
    <static>false</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>{radius:.3f}</radius>
            <length>{height:.3f}</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{radius:.3f}</radius>
            <length>{height:.3f}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.1 0.1 0.8 1</ambient>
          <diffuse>0.1 0.1 0.8 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


# ------------------------------
# 采样不重叠的中心点
# ------------------------------
def sample_centers(n, xmin, xmax, ymin, ymax, clearance, seed=42) -> List[Tuple[float, float]]:
    rnd = random.Random(seed)
    pts: List[Tuple[float, float]] = []
    tries = 0
    while len(pts) < n and tries < 5000:
        tries += 1
        x = rnd.uniform(xmin, xmax)
        y = rnd.uniform(ymin, ymax)
        ok = True
        for (px, py) in pts:
            if math.hypot(x - px, y - py) < clearance:
                ok = False
                break
        if ok:
            pts.append((x, y))
    if len(pts) < n:
        raise RuntimeError("采样失败：范围过小或 clearance 过大，无法放下这么多物体。")
    return pts


# ------------------------------
# 主节点
# ------------------------------
class DynamicSpawner(Node):
    def __init__(self, args):
        super().__init__("dynamic_spawner")
        self.args = args

        # Service Clients
        self.cli_spawn = self.create_client(SpawnEntity, "/spawn_entity")
        if not self.cli_spawn.wait_for_service(timeout_sec=15.0):
            raise RuntimeError("等待 /spawn_entity 超时，请确认 gazebo_ros 已启动（world.launch.py）")

        # 只有“圆柱滑动”模式需要 SetEntityState
        self.cli_set = None
        if not args.humans:
            self.cli_set = self.create_client(SetEntityState, "/set_entity_state")
            if not self.cli_set.wait_for_service(timeout_sec=15.0):
                raise RuntimeError("等待 /set_entity_state 超时。")

        # 先批量生成
        self.names: List[str] = []
        self.centers: List[Tuple[float, float]] = sample_centers(
            n=args.n, xmin=args.xmin, xmax=args.xmax, ymin=args.ymin, ymax=args.ymax,
            clearance=args.clearance, seed=args.seed
        )
        self.spawn_all()

        # 再启动计时器（仅圆柱模式；Actor 由 Gazebo 自己行走）
        if not args.humans:
            self.theta = [2.0 * math.pi * i / args.n for i in range(args.n)]  # 相位错开
            self.timer = self.create_timer(args.dt, self._on_timer)
            self.t0 = time.time()

    # 批量生成
    def spawn_all(self):
        for i, (cx, cy) in enumerate(self.centers):
            name = f"dyn_{'hum' if self.args.humans else 'cyl'}_{i:02d}"
            if self.args.humans:
                if self.args.sfm:
                    xml = make_actor_xml_with_sfm(
                        name=name,
                        mesh_uri=self.args.mesh_uri,
                        center=(cx, cy),
                        ax=self.args.ax,
                        by=self.args.by,
                        z=max(1.0, self.args.z),
                        scale=self.args.scale,
                        velocity=self.args.speed,
                    )
                else:
                    # 用脚本轨迹的 Actor（更简单，立刻有走路动画）
                    period = max(2.0, self.args.period)  # 一圈时间
                    xml = make_actor_xml_walk(
                        name=name,
                        mesh_uri=self.args.mesh_uri,
                        center=(cx, cy),
                        ax=self.args.ax,
                        by=self.args.by,
                        period=period,
                        n_pts=self.args.n_traj_pts,
                        z=max(1.0, self.args.z),
                        scale=self.args.scale,
                    )
            else:
                # 圆柱 SDF；位置在 spawn 时由 Pose 指定，后续用 SetEntityState“滑动”
                xml = make_cylinder_sdf(name=name, radius=self.args.cyl_radius, height=self.args.cyl_height)

            # 发送 spawn 请求
            req = SpawnEntity.Request()
            req.name = name
            req.xml = xml
            # 初始位姿：圆柱放地面，Actor 的 z 在 SDF 里已经设定
            pose = Pose()
            pose.position.x = float(cx)
            pose.position.y = float(cy)
            pose.position.z = 1.0 
            req.initial_pose = pose
            req.robot_namespace = ""

            self.get_logger().info(f"[spawn] {name} at ({cx:.2f}, {cy:.2f})")
            fut = self.cli_spawn.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
            if not fut.done() or fut.result() is None:
                self.get_logger().error(f"[spawn] {name} 失败（超时或错误）")
            else:
                self.names.append(name)

        if self.args.humans:
            self.get_logger().info("[info] Actor 由 Gazebo 自动行走（脚本或 SFM），脚本不会再用 SetEntityState。")

    # 定时器：仅圆柱“滑动”模式
    def _on_timer(self):
        if not self.cli_set:
            return
        t = time.time() - self.t0
        for i, name in enumerate(self.names):
            # 椭圆参数：中心 = centers[i]，半轴 = ax/by，角速度 = 2π/period
            theta = self.theta[i] + 2.0 * math.pi * (t / max(1e-6, self.args.period))
            x = self.centers[i][0] + self.args.ax * math.cos(theta)
            y = self.centers[i][1] + self.args.by * math.sin(theta)

            req = SetEntityState.Request()
            req.state = EntityState()
            req.state.name = name
            req.state.pose.position.x = float(x)
            req.state.pose.position.y = float(y)
            req.state.pose.position.z = float(self.args.z)
            req.state.twist = Twist()

            self.cli_set.call_async(req)  # 异步即可；不阻塞

        # 可选：打印帧率/状态
        # self.get_logger().info(f"[move] t={t:.2f}")


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8, help='障碍物/行人数')
    ap.add_argument('--xmin', type=float, default=-10.0)
    ap.add_argument('--xmax', type=float, default=10.0)
    ap.add_argument('--ymin', type=float, default=-10.0)
    ap.add_argument('--ymax', type=float, default=10.0)
    ap.add_argument('--clearance', type=float, default=0.8, help='生成时中心点之间的最小间距')

    # 椭圆轨迹参数（圆柱滑动 & Actor 轨迹都用得到）
    ap.add_argument('--ax', type=float, default=0.8, help='椭圆半轴 x')
    ap.add_argument('--by', type=float, default=0.6, help='椭圆半轴 y')
    ap.add_argument('--period', type=float, default=12.0, help='一圈时间（秒）')
    ap.add_argument('--dt', type=float, default=0.05, help='圆柱滑动模式的更新周期（秒）')
    ap.add_argument('--z', type=float, default=0.0, help='圆柱的 z（Actor 的 z 在 SDF 中设 1.0）')

    # 圆柱尺寸（仅圆柱模式）
    ap.add_argument('--cyl_radius', type=float, default=0.2)
    ap.add_argument('--cyl_height', type=float, default=1.2)

    # 人形 Actor 相关
    ap.add_argument('--humans', action='store_true', help='使用人形 Actor（不再使用 SetEntityState）')
    ap.add_argument('--sfm', action='store_true', help='为 Actor 挂载社会力 SFM 插件（需安装 gazebo_sfm_plugin）')
    ap.add_argument('--mesh_uri', type=str, default='model://person_walking/meshes/walking.dae',
                    help='人形模型（DAE）路径，支持 model:// 或 file:/// 绝对路径')
    ap.add_argument('--scale', type=float, default=1.0, help='人形缩放比例')
    ap.add_argument('--n_traj_pts', type=int, default=32, help='Actor 轨迹采样点数（脚本轨迹模式）')
    ap.add_argument('--speed', type=float, default=0.9, help='SFM 期望速度（m/s，仅 --sfm 时生效）')

    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rclpy.init()
    try:
        node = DynamicSpawner(args)
        rclpy.spin(node)  # 圆柱模式需要持续驱动；Actor 模式留着也行（无害）
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
