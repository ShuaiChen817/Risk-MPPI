#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, GetModelList
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import ModelStates

# ======= 参数与数据结构 =======
@dataclass
class Ellipse:
    cx: float
    cy: float
    ax: float
    by: float
    omega: float
    phi: float

@dataclass
class Human:
    name: str
    ellipse: Ellipse

def dist2(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

# ======= 主节点 =======
class HumansEllipses(Node):
    def __init__(self, args):
        super().__init__('humans_ellipses_spawner')

        self.n = args.n
        self.bounds = (args.xmin, args.xmax, args.ymin, args.ymax)
        self.clearance = args.clearance
        self.ax = args.ax
        self.by = args.by
        self.omega = args.omega
        self.mesh_uri = args.mesh_uri  # file://media/models/walk.dae
        self.scale = args.scale
        self.prefix = args.actor_prefix
        self.rate_hz = args.rate
        self.min_center_gap = max(self.clearance*2.0, max(self.ax, self.by)*2.2)

        # ROS 2 clients
        self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
        self.set_state_cli = self.create_client(SetEntityState, '/set_entity_state')

        # 订阅静态模型位置（来自 gazebo_ros_state 插件）
        self.model_states_sub = self.create_subscription(
            ModelStates, '/model_states', self._on_model_states, 10
        )
        self.world_models_xy = {}  # name -> (x,y)

        # 定时器
        self.t0 = time.time()
        self.humans : List[Human] = []

        self.get_logger().info('[init] waiting for /spawn_entity & /set_entity_state...')
        self.spawn_cli.wait_for_service()
        self.set_state_cli.wait_for_service()
        self.get_logger().info('[init] services ready.')

        # 先等待一次 /model_states，拿到静态物体集合用于采样避碰
        self._wait_model_states()
        self._spawn_humans_nonoverlap()

        # 定时更新位姿
        self.timer = self.create_timer(1.0/self.rate_hz, self._on_timer)

    # ---------- 模型状态 ----------
    def _on_model_states(self, msg: ModelStates):
        for i, name in enumerate(msg.name):
            p = msg.pose[i].position
            self.world_models_xy[name] = (p.x, p.y)

    def _wait_model_states(self):
        t0 = time.time()
        while rclpy.ok() and (time.time()-t0) < 3.0 and not self.world_models_xy:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f'[model_states] collected {len(self.world_models_xy)} models.')

    # ---------- 采样非重叠圆心 ----------
    def _spawn_humans_nonoverlap(self):
        xmin, xmax, ymin, ymax = self.bounds
        centers : List[Tuple[float,float]] = []

        # 把明显是机器人/人本体之外的静态模型名字当作障碍（你可以按需过滤名字前缀）
        static_xy = []
        for name, (x,y) in self.world_models_xy.items():
            # 排除 Gazebo 内部模型与将要生成的 actor 前缀
            if name.startswith(self.prefix) or name in ('ground_plane','tb3'):
                continue
            static_xy.append((x,y))

        for i in range(self.n):
            ok = False
            ax_i = self.ax
            by_i = self.by
            for tries in range(200):
                cx = random.uniform(xmin+ax_i+self.clearance, xmax-ax_i-self.clearance)
                cy = random.uniform(ymin+by_i+self.clearance, ymax-by_i-self.clearance)

                # 距已有中心点
                if any(dist2((cx,cy), c) < (self.min_center_gap**2) for c in centers):
                    continue
                # 距静态障碍
                if any(dist2((cx,cy), s) < ((max(ax_i,by_i)+self.clearance)**2) for s in static_xy):
                    continue

                centers.append((cx,cy))
                ok = True
                break

            # 实在放不下，缩小半轴重试（不改已有）
            if not ok:
                ax_i *= 0.8
                by_i *= 0.8
                centers.append((
                    random.uniform(xmin+ax_i+self.clearance, xmax-ax_i-self.clearance),
                    random.uniform(ymin+by_i+self.clearance, ymax-by_i-self.clearance)
                ))
                self.get_logger().warn(f'[spawn] human_{i:02d} fallback with smaller ellipse.')

            phi = random.uniform(0.0, 2*math.pi)
            name = f'{self.prefix}{i:02d}'
            ell = Ellipse(centers[-1][0], centers[-1][1], ax_i, by_i, self.omega, phi)
            self.humans.append(Human(name, ell))

        # 执行 spawn
        for h in self.humans:
            ok = self._spawn_one_actor(h)
            self.get_logger().info(f'[spawn] {h.name} at ({h.ellipse.cx:.2f}, {h.ellipse.cy:.2f}) -> {ok}')

    # ---------- 生成一个 actor ----------
    def _spawn_one_actor(self, human: Human) -> bool:
        # 生成最小 actor SDF（禁用脚本自动位移；动画由 DAE 骨骼播放）
        actor_sdf = f"""<?xml version='1.0'?>
<sdf version='1.7'>
  <actor name='{human.name}'>
    <pose>{human.ellipse.cx} {human.ellipse.cy} 1 0 0 0</pose>
    <skin>
      <filename>{self.mesh_uri}</filename>
      <scale>{self.scale}</scale>
    </skin>
    <animation name='walking'>
      <filename>{self.mesh_uri}</filename>
      <interpolate_x>true</interpolate_x>
    </animation>
    <script>
      <loop>true</loop>
      <auto_start>true</auto_start>
      <trajectory id='0' type='walk'>
        <waypoint><time>0.0</time><pose>0 0 0 0 0 0</pose></waypoint>
        <waypoint><time>1.0</time><pose>0 0 0 0 0 0</pose></waypoint>
      </trajectory>
    </script>
  </actor>
</sdf>"""
        req = SpawnEntity.Request()
        req.name = human.name
        req.xml = actor_sdf
        req.robot_namespace = ''
        req.reference_frame = 'world'

        fut = self.spawn_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        return bool(fut.result() and fut.result().success)

    # ---------- 定时推进位姿 ----------
    def _on_timer(self):
        t = time.time() - self.t0
        # 障碍（静态）列表（每次用最新）
        static_xy = []
        for name,(x,y) in self.world_models_xy.items():
            if name.startswith(self.prefix) or name in ('ground_plane','tb3'):
                continue
            static_xy.append((x,y))

        for h in self.humans:
            e = h.ellipse
            # 预测下一步
            nx = e.cx + e.ax*math.cos(e.omega*t + e.phi)
            ny = e.cy + e.by*math.sin(e.omega*t + e.phi)

            # 若靠近静态障碍，临时“掉头”
            hit = False
            for (ox,oy) in static_xy:
                if dist2((nx,ny),(ox,oy)) < (self.clearance**2):
                    hit = True
                    break
            if hit:
                # 反向推进（等价反转相位），并稍作退让
                nx = e.cx + e.ax*math.cos(-(e.omega*t) + e.phi)
                ny = e.cy + e.by*math.sin(-(e.omega*t) + e.phi)

            # 写入 Gazebo
            state = EntityState()
            state.name = h.name
            state.pose.position.x = float(nx)
            state.pose.position.y = float(ny)
            state.pose.position.z = 1.0
            # yaw 指向切向方向，便于视觉自然
            dxdt = -e.ax*e.omega*math.sin(e.omega*t + e.phi)
            dydt =  e.by*e.omega*math.cos(e.omega*t + e.phi)
            yaw = math.atan2(dydt, dxdt)
            cy = math.cos(0.5*yaw); sy = math.sin(0.5*yaw)
            state.pose.orientation.z = sy
            state.pose.orientation.w = cy

            req = SetEntityState.Request()
            req.state = state
            self.set_state_cli.call_async(req)  # 异步发送即可

# ======= 入口 =======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--actor-prefix', type=str, default='human_')
    parser.add_argument('--xmin', type=float, default=-10.0)
    parser.add_argument('--xmax', type=float, default= 10.0)
    parser.add_argument('--ymin', type=float, default=-10.0)
    parser.add_argument('--ymax', type=float, default= 10.0)
    parser.add_argument('--clearance', type=float, default=0.7,
                        help='最小到静态障碍和相互之间的间距（米）')
    parser.add_argument('--ax', type=float, default=0.8)
    parser.add_argument('--by', type=float, default=0.6)
    parser.add_argument('--omega', type=float, default=0.35)
    parser.add_argument('--mesh_uri', type=str, default='file://media/models/walk.dae')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--rate', type=float, default=20.0)

    args = parser.parse_args()
    rclpy.init()
    node = HumansEllipses(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
