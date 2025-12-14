#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只为 Gazebo Actor（human_*）生成/同步单个“躯干圆柱”代理；
可选随机游走。ROS 2 Humble + Gazebo Classic 11。
"""
import math, random, time, argparse, textwrap
from typing import Dict, Tuple, List

import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates, EntityState
from gazebo_msgs.srv import SetEntityState, SpawnEntity
from geometry_msgs.msg import Pose

PROXY_SUFFIX = "_proxy"
IGNORE_NAMES = {"ground_plane","sun"}

def quat_z_w_of_yaw(yaw: float):
    return math.sin(yaw*0.5), math.cos(yaw*0.5)

def yaw_of_qz_qw(qz: float, qw: float) -> float:
    return 2.0 * math.atan2(qz, qw)

def sdf_cylinder_model(model_name: str, radius: float, length: float,
                       z_bottom: float) -> str:
    """
    生成完全透明但具有真实物理碰撞的圆柱体。
    """
    z_mid = z_bottom + 0.5 * length
    mass = 8.0
    Ixx = 1/12 * mass * (3 * radius * radius + length * length)
    Iyy = Ixx
    Izz = 0.5 * mass * radius * radius

    return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>false</static>
    <allow_auto_disable>false</allow_auto_disable>
    <pose>0 0 0 0 0 0</pose>

    <link name="link">
      <inertial>
        <mass>{mass:.3f}</mass>
        <inertia>
          <ixx>{Ixx:.6f}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyx>0</iyx><iyy>{Iyy:.6f}</iyy><iyz>0</iyz>
          <izx>0</izx><izy>0</izy><izz>{Izz:.6f}</izz>
        </inertia>
      </inertial>

      <!-- 真实物理碰撞体 -->
      <collision name="collision">
        <pose>0 0 {z_mid:.3f} 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>{radius:.3f}</radius>
            <length>{length:.3f}</length>
          </cylinder>
        </geometry>
        <surface>
          <contact><ode/></contact>
          <friction>
            <ode><mu>1.0</mu><mu2>1.0</mu2></ode>
          </friction>
        </surface>
      </collision>

      <!-- 完全透明的视觉模型 -->
      <visual name="visual">
        <pose>0 0 {z_mid:.3f} 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>{radius:.3f}</radius>
            <length>{length:.3f}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0 0 0 0</ambient>
          <diffuse>0 0 0 0</diffuse>
          <specular>0 0 0 0</specular>
          <emissive>0 0 0 0</emissive>
        </material>
        <transparency>1.0</transparency>
        <cast_shadows>false</cast_shadows>
      </visual>

      <gravity>true</gravity>
      <kinematic>false</kinematic>
      <self_collide>false</self_collide>
    </link>
  </model>
</sdf>
"""



class HumansTorsoProxies(Node):
    def __init__(self, args):
        super().__init__('humans_torso_proxies')

        self.prefix = args.actor_prefix
        self.rate   = max(1.0, float(args.rate))
        self.dt     = 1.0/self.rate
        self.autospawn = bool(args.autospawn)

        # 躯干参数
        self.torso_r = args.torso_radius
        self.torso_h = args.torso_height
        self.torso_z0 = args.torso_bottom_z

        # wander 参数
        self.enable_wander = bool(args.wander)
        self.xmin, self.xmax = args.xmin, args.xmax
        self.ymin, self.ymax = args.ymin, args.ymax
        self.speed_min, self.speed_max = args.speed_min, args.speed_max
        self.clearance = args.clearance
        self.turn_on_bounce = bool(args.turn_on_bounce)
        self.rng = random.Random(args.seed) if args.seed is not None else random.Random()

        # 服务
        self.set_cli = None
        for name in ['/set_entity_state','/gazebo/set_entity_state']:
            cli = self.create_client(SetEntityState, name)
            if cli.wait_for_service(timeout_sec=2.0):
                self.set_cli = cli
                self.get_logger().info(f'使用 SetEntityState: {name}')
                break
        if self.set_cli is None:
            raise RuntimeError('未找到 SetEntityState')

        self.spawn_cli = None
        if self.autospawn:
            self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
            if not self.spawn_cli.wait_for_service(timeout_sec=5.0):
                raise RuntimeError('未等到 /spawn_entity')

        # 订阅
        self.states = None
        self.create_subscription(ModelStates, '/model_states', self._on_states, 10)

        # 状态缓存
        self.actors: List[str] = []
        self.pos: Dict[str, Tuple[float,float]] = {}
        self.vel: Dict[str, Tuple[float,float]] = {}
        self.known_spawned = set()

        # 初始化
        self._wait_states(3.0)
        self._init_actors()
        self.timer = self.create_timer(self.dt, self._on_timer)

    def _on_states(self, msg: ModelStates):
        self.states = msg

    def _wait_states(self, sec=3.0):
        t0 = time.monotonic()
        while self.states is None and (time.monotonic()-t0) < sec:
            rclpy.spin_once(self, timeout_sec=0.1)

    def _set_pose(self, name, x, y, yaw):
        qz,qw = quat_z_w_of_yaw(yaw)
        st = EntityState()
        st.name = name
        st.pose.position.x = float(x)
        st.pose.position.y = float(y)
        st.pose.position.z = 0.0
        st.pose.orientation.z = qz
        st.pose.orientation.w = qw
        req = SetEntityState.Request(); req.state = st
        self.set_cli.call_async(req)

    def _spawn_model(self, name, x, y, yaw):
        if not self.spawn_cli: return
        sdf = sdf_cylinder_model(name, self.torso_r, self.torso_h, self.torso_z0)
        pose = Pose()
        pose.position.x = float(x); pose.position.y = float(y)
        qz,qw = quat_z_w_of_yaw(yaw)
        pose.orientation.z = qz; pose.orientation.w = qw
        req = SpawnEntity.Request()
        req.name = name; req.xml = sdf; req.initial_pose = pose; req.reference_frame = 'world'
        fut = self.spawn_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        ok = fut.done() and fut.result() and fut.result().success
        self.get_logger().info(f'spawn {name}: {ok}')

    def _init_actors(self):
        if self.states:
            self.actors = [n for n in self.states.name if n.startswith(self.prefix) and not n.endswith(PROXY_SUFFIX)]
        self.get_logger().info(f'发现 actors: {self.actors}')

        if self.enable_wander:
            for n, p in zip(self.states.name, self.states.pose):
                if n in self.actors:
                    x,y = p.position.x, p.position.y
                    ang = self.rng.uniform(-math.pi, math.pi)
                    spd = self.rng.uniform(self.speed_min, self.speed_max)
                    self.pos[n] = (x,y)
                    self.vel[n] = (spd*math.cos(ang), spd*math.sin(ang))

    def _on_timer(self):
        if not self.states: return

        pose_map = {}
        for n,p in zip(self.states.name, self.states.pose):
            if n in self.actors:
                yaw = yaw_of_qz_qw(p.orientation.z, p.orientation.w)
                pose_map[n] = (p.position.x, p.position.y, yaw)

        # wander
        if self.enable_wander:
            for n in self.actors:
                x,y = pose_map.get(n,(0,0,0))[:2]
                vx,vy = self.vel.get(n,(0,0))
                nx,ny = x + vx*self.dt, y + vy*self.dt
                if nx<self.xmin or nx>self.xmax or ny<self.ymin or ny>self.ymax:
                    if self.turn_on_bounce:
                        vx,vy = -vx,-vy
                        nx,ny = x + vx*self.dt, y + vy*self.dt
                    else:
                        ang = self.rng.uniform(-math.pi, math.pi)
                        spd = self.rng.uniform(self.speed_min, self.speed_max)
                        vx,vy = spd*math.cos(ang), spd*math.sin(ang)
                        nx,ny = x + vx*self.dt, y + vy*self.dt
                yaw = math.atan2(vy,vx)
                self._set_pose(n, nx, ny, yaw)
                self.vel[n] = (vx,vy)
                pose_map[n] = (nx,ny,yaw)

        # 躯干代理同步
        for n in self.actors:
            if n not in pose_map: continue
            x,y,yaw = pose_map[n]
            proxy_name = f'{n}_torso{PROXY_SUFFIX}'
            if proxy_name not in self.known_spawned and self.autospawn:
                self._spawn_model(proxy_name, x, y, yaw)
                self.known_spawned.add(proxy_name)
            self._set_pose(proxy_name, x, y, yaw)

def main():
    ap = argparse.ArgumentParser(description='生成并同步 humans 躯干圆柱代理')
    ap.add_argument('--actor-prefix', default='human_')
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--autospawn', action='store_true')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--torso-radius', type=float, default=0.45)
    ap.add_argument('--torso-height', type=float, default=1.6)
    ap.add_argument('--torso-bottom-z', type=float, default=0.0)
    ap.add_argument('--wander', action='store_true')
    ap.add_argument('--xmin', type=float, default=-10)
    ap.add_argument('--xmax', type=float, default=10)
    ap.add_argument('--ymin', type=float, default=-10)
    ap.add_argument('--ymax', type=float, default=10)
    ap.add_argument('--speed-min', type=float, default=0.3)
    ap.add_argument('--speed-max', type=float, default=1.1)
    ap.add_argument('--clearance', type=float, default=0.7)
    ap.add_argument('--turn-on-bounce', action='store_true')
    args = ap.parse_args()

    rclpy.init()
    try:
        node = HumansTorsoProxies(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
