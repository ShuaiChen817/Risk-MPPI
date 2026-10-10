#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SpawnEntity


@dataclass
class Bounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


def ellipse_points(cx: float, cy: float, a: float, b: float, phase: float,
                   num: int, period: float, face_forward: bool, z: float = 1.0) -> List[Tuple[float, float, float, float]]:
    """Generate waypoints (t, x, y, yaw) along an ellipse centered at (cx,cy).
    Time t in seconds, positions in meters, yaw in radians.
    """
    pts = []
    for k in range(num):
        theta = phase + 2.0 * math.pi * (k / num)
        x = cx + a * math.cos(theta)
        y = cy + b * math.sin(theta)
        if face_forward:
            dx = -a * math.sin(theta)
            dy =  b * math.cos(theta)
            yaw = math.atan2(dy, dx)
        else:
            yaw = 0.0
        t = period * (k / (num - 1 if num > 1 else 1))
        pts.append((t, x, y, yaw))
    # ensure loop closure at T
    pts[-1] = (period, pts[0][1], pts[0][2], pts[0][3])
    return pts


def sdf_actor(name: str, waypoints: List[Tuple[float, float, float, float]],
              skin_uri: str = 'model://person_walking/meshes/walking.dae',
              scale: float = 0.8, z: float = 1.0) -> str:
    """Build an SDF <actor> with a looping trajectory defined by waypoints.
    waypoints: list of (t, x, y, yaw)
    """
    wp_xml = []
    for t, x, y, yaw in waypoints:
        wp_xml.append(f"""
          <waypoint>
            <time>{t:.3f}</time>
            <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.6f}</pose>
          </waypoint>
        """)
    wps = "\n".join(wp_xml)
    xml = f"""
<sdf version='1.7'>
  <actor name='{name}'>
    <pose>0 0 {z:.3f} 0 0 0</pose>
    <skin>
      <filename>{skin_uri}</filename>
      <scale>{scale:.3f}</scale>
    </skin>
    <animation name='walking'>
      <filename>{skin_uri}</filename>
      <interpolate_x>false</interpolate_x>
    </animation>
    <script>
      <loop>true</loop>
      <auto_start>true</auto_start>
      <trajectory id='0' type='walking'>
{wps}
      </trajectory>
    </script>
  </actor>
</sdf>
"""
    return xml


class HumanSpawner(Node):
    def __init__(self, args):
        super().__init__('spawn_elliptical_humans')
        self.declare_parameters('', [
            ('num', 3),
            ('bounds', [-10.0, 10.0, -10.0, 10.0]),
            ('a_range', [4.0, 6.0]),
            ('b_range', [2.0, 4.0]),
            ('period', 18.0),
            ('face_forward', True),
            ('z', 1.0),
            ('min_clearance', 0.8),
            ('obstacle_radius', 0.35),
            ('actor_radius', 0.30),
            ('samples_per_ellipse', 72),
            ('name_prefix', 'human'),
            ('seed', 0),
            ('ignore_obstacles', False),
            ('relax_enable', True),
            ('relax_steps', 6),
            ('relax_clearance_step', 0.2),
            ('relax_shrink_factor', 0.85),  # 每步把半轴乘以该系数
            ('skin_uri', 'model://person_walking/meshes/walking.dae'),
            ('skin_fallback', 'file://media/models/walk.dae'),
            ('skin_scale', 0.8),
        ])

        if args is not None:
            # 允许通过 argparse 传入的简单参数，稍后统一 reload
            self._args = args
        else:
            self._args = None

        seed = int(self.get_parameter('seed').value)
        if seed != 0:
            random.seed(seed)

        # 读取参数到实例变量
        self._reload_params()

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST,
                         depth=10)
        self.obs_ready = False
        self.obstacles: List[Tuple[float, float]] = []
        self.create_subscription(ModelStates, '/model_states', self._ms_cb, qos)

        self.cli = self.create_client(SpawnEntity, '/spawn_entity')

        self.timer = self.create_timer(0.2, self._tick)
        self.stage = 0

    def _reload_params(self):
        bvals = self.get_parameter('bounds').value
        self.bounds = Bounds(float(bvals[0]), float(bvals[1]), float(bvals[2]), float(bvals[3]))
        self.num = int(self.get_parameter('num').value)
        self.a_range = tuple(float(x) for x in self.get_parameter('a_range').value)
        self.b_range = tuple(float(x) for x in self.get_parameter('b_range').value)
        self.period = float(self.get_parameter('period').value)
        self.face_forward = bool(self.get_parameter('face_forward').value)
        self.z = float(self.get_parameter('z').value)
        self.min_clearance = float(self.get_parameter('min_clearance').value)
        self.obstacle_radius = float(self.get_parameter('obstacle_radius').value)
        self.actor_radius = float(self.get_parameter('actor_radius').value)
        self.samples_per_ellipse = int(self.get_parameter('samples_per_ellipse').value)
        self.name_prefix = str(self.get_parameter('name_prefix').value)
        self.ignore_obstacles = bool(self.get_parameter('ignore_obstacles').value)
        self.relax_enable = bool(self.get_parameter('relax_enable').value)
        self.relax_steps = int(self.get_parameter('relax_steps').value)
        self.relax_clearance_step = float(self.get_parameter('relax_clearance_step').value)
        self.relax_shrink_factor = float(self.get_parameter('relax_shrink_factor').value)
        self.skin_uri = str(self.get_parameter('skin_uri').value)
        self.skin_fallback = str(self.get_parameter('skin_fallback').value)
        self.skin_scale = float(self.get_parameter('skin_scale').value)

    def _ms_cb(self, msg: ModelStates):
        obs = []
        for name, pose in zip(msg.name, msg.pose):
            # 过滤出树等静态障碍（示例：名字以 pine_ 开头）
            if name.startswith('pine_'):
                obs.append((pose.position.x, pose.position.y))
        if obs:
            self.obstacles = obs
            self.obs_ready = True

    def _tick(self):
        # 每次 tick 先刷新一次参数（确保 CLI/ros-args 生效）
        self._reload_params()
        if self.stage != 0:
            return
        if not self.obs_ready and not self.ignore_obstacles:
            self.get_logger().info('等待 /model_states 获得障碍物位置…(可用 ignore_obstacles:=true 跳过)')
            return
        if not self.cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('spawn_entity 服务不可用')
            return

        self.get_logger().info(
            f'获得 {len(self.obstacles)} 个障碍，开始采样生成 {self.num} 个 actor (ignore={self.ignore_obstacles})'
        )
        placements = self._plan_actors()
        if not placements:
            self.get_logger().error('无法找到合法的起点与椭圆参数，请放宽边界或减小数量')
            self.stage = 99
            return

        # 逐个 spawn
        ok = True
        for i, (cx, cy, a, b, phase) in enumerate(placements):
            name = f"{self.name_prefix}_{i+1:02d}"
            wps = ellipse_points(cx, cy, a, b, phase,
                                 num=self.samples_per_ellipse,
                                 period=self.period,
                                 face_forward=self.face_forward,
                                 z=self.z)
            xml = sdf_actor(name, wps, skin_uri=self.skin_uri, scale=self.skin_scale, z=self.z)
            req = SpawnEntity.Request()
            req.name = name
            req.xml = xml
            req.robot_namespace = ''
            req.reference_frame = ''
            req.initial_pose = Pose()
            fut = self.cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=20.0)
            res = fut.result() if fut.done() else None
            if res is None:
                self.get_logger().error(f'spawn {name} 失败：服务无响应')
                ok = False
            elif not res.success:
                self.get_logger().warn(f'spawn {name} 返回失败：{res.status_message}，尝试使用备用皮肤URI')
                # 尝试备用皮肤（Gazebo自带 walk.dae）
                xml2 = sdf_actor(name, wps, skin_uri=self.skin_fallback, scale=self.skin_scale, z=self.z)
                req2 = SpawnEntity.Request()
                req2.name = name
                req2.xml = xml2
                req2.robot_namespace = ''
                req2.reference_frame = ''
                req2.initial_pose = Pose()
                fut2 = self.cli.call_async(req2)
                rclpy.spin_until_future_complete(self, fut2, timeout_sec=20.0)
                res2 = fut2.result() if fut2.done() else None
                if res2 is None or not res2.success:
                    msg = res2.status_message if (res2 is not None and hasattr(res2, 'status_message')) else 'unknown'
                    self.get_logger().error(f'spawn {name} 仍失败（备用皮肤）：{msg}')
                    ok = False
                else:
                    self.get_logger().info(f'spawn {name} 成功（使用备用皮肤）@ ({cx:.2f},{cy:.2f}), a={a:.1f}, b={b:.1f}')
            else:
                self.get_logger().info(f'spawn {name} 成功 @ ({cx:.2f},{cy:.2f}), a={a:.1f}, b={b:.1f}')
        if ok:
            self.get_logger().info('全部 actor 生成完成')
        self.stage = 99

    def _plan_actors(self) -> List[Tuple[float, float, float, float, float]]:
        results: List[Tuple[float, float, float, float, float]] = []
        max_trials = 2000
        # 放松策略：多轮尝试，逐步减小 clearance、缩小半轴
        relax_levels = [0]
        if self.relax_enable:
            relax_levels = list(range(0, max(1, self.relax_steps + 1)))
        for lvl in relax_levels:
            clr = max(0.0, self.min_clearance - lvl * self.relax_clearance_step)
            shrink = (self.relax_shrink_factor ** lvl)
            a_min, a_max = self.a_range[0] * shrink, self.a_range[1] * shrink
            b_min, b_max = self.b_range[0] * shrink, self.b_range[1] * shrink
            self.get_logger().info(f'放松级别 {lvl}: clearance={clr:.2f}, a~[{a_min:.2f},{a_max:.2f}], b~[{b_min:.2f},{b_max:.2f}]')
            trials = 0
            while len(results) < self.num and trials < max_trials:
                trials += 1
                a = random.uniform(a_min, a_max)
                b = random.uniform(b_min, b_max)
                # 边界收缩，留 0.3m 富余
                cx = random.uniform(self.bounds.xmin + a + 0.3, self.bounds.xmax - a - 0.3)
                cy = random.uniform(self.bounds.ymin + b + 0.3, self.bounds.ymax - b - 0.3)
                phase = random.uniform(0.0, 2.0 * math.pi)
                if not self._ellipse_ok(cx, cy, a, b, phase, results, clr):
                    continue
                results.append((cx, cy, a, b, phase))
            if len(results) >= self.num:
                break
        return results

    def _ellipse_ok(self, cx: float, cy: float, a: float, b: float, phase: float,
                     existing: List[Tuple[float, float, float, float, float]], clr: float) -> bool:
        # 采样椭圆点，检查与障碍物/已有 actor 的最小距离
        r_actor = self.actor_radius
        margin = clr
        pts = ellipse_points(cx, cy, a, b, phase, num=max(24, self.samples_per_ellipse//2), period=self.period, face_forward=True, z=self.z)
        # 边界检查
        for _, x, y, _ in pts:
            if not (self.bounds.xmin <= x <= self.bounds.xmax and self.bounds.ymin <= y <= self.bounds.ymax):
                return False
        # 与障碍物（可忽略）
        if not self.ignore_obstacles:
            r_obs = self.obstacle_radius
            min_d_ok = r_actor + r_obs + margin
            for ox, oy in self.obstacles:
                for _, x, y, _ in pts:
                    if (x - ox) * (x - ox) + (y - oy) * (y - oy) < (min_d_ok * min_d_ok):
                        return False
        # 与已放置的椭圆（中心距和半轴粗略检查 + 采样最近点）
        for ex_cx, ex_cy, ex_a, ex_b, ex_phase in existing:
            other = ellipse_points(ex_cx, ex_cy, ex_a, ex_b, ex_phase, num=36, period=self.period, face_forward=True, z=self.z)
            for _, x1, y1, _ in pts:
                for _, x2, y2, _ in other:
                    if (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) < (2 * r_actor + margin) ** 2:
                        return False
        return True


def main():
    parser = argparse.ArgumentParser(description='Spawn multiple elliptical-walking humans (Actor) in Gazebo, avoiding trees.')
    parser.add_argument('--num', type=int, default=None, help='生成人数')
    parser.add_argument('--bounds', nargs=4, type=float, default=None, metavar=('XMIN','XMAX','YMIN','YMAX'))
    parser.add_argument('--a-range', nargs=2, type=float, default=None)
    parser.add_argument('--b-range', nargs=2, type=float, default=None)
    parser.add_argument('--period', type=float, default=None)
    parser.add_argument('--clearance', type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = HumanSpawner(args)
    if args.num is not None:
        node.set_parameters([rclpy.parameter.Parameter('num', rclpy.Parameter.Type.INTEGER, args.num)])
    if args.bounds is not None:
        node.set_parameters([rclpy.parameter.Parameter('bounds', rclpy.Parameter.Type.DOUBLE_ARRAY, args.bounds)])
    if args.a_range is not None:
        node.set_parameters([rclpy.parameter.Parameter('a_range', rclpy.Parameter.Type.DOUBLE_ARRAY, args.a_range)])
    if args.b_range is not None:
        node.set_parameters([rclpy.parameter.Parameter('b_range', rclpy.Parameter.Type.DOUBLE_ARRAY, args.b_range)])
    if args.period is not None:
        node.set_parameters([rclpy.parameter.Parameter('period', rclpy.Parameter.Type.DOUBLE, args.period)])
    if args.clearance is not None:
        node.set_parameters([rclpy.parameter.Parameter('min_clearance', rclpy.Parameter.Type.DOUBLE, args.clearance)])

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
