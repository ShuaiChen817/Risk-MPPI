#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 Gazebo Classic Actor（human_*）生成/同步“多圆柱”碰撞代理（torso/head/legL/legR）
- 每个部件是一个独立的 Gazebo model（带 <collision> cylinder）
- 仅做整体平移/旋转（不会跟随四肢摆动），但轮廓更接近“人”
- 供 Lidar 命中 -> Nav2 代价地图产生障碍；RViz 也能看到 /scan 的回波

需求：
- ROS 2 Humble
- Gazebo Classic 11（加载了 gazebo_ros_factory 与 gazebo_ros_state）
- 世界里已有 <actor name="human_01">…（会走的那个人）

用法示例：
python3 human_multi_cylinder_proxies.py --actor-prefix human_ --autospawn --rate 20
"""

import os, math, argparse, textwrap
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates, EntityState
from gazebo_msgs.srv import SetEntityState, SpawnEntity
from geometry_msgs.msg import Pose

COMP_NAMES = ("torso", "head", "legL", "legR")
PROXY_SUFFIX = "_proxy"  # 生成的模型名：human_01_torso_proxy 等

def sdf_cylinder_model(model_name: str, radius: float, length: float,
                       z_bottom: float, rgba=(0,0,0,0.03)):
    """
    生成一个带圆柱 collision 的最小模型（visual 半透明，不碍视野）
    z_bottom 是底面在世界坐标的 z；collision/visual 都以长度中心抬起 length/2
    """
    r,g,b,a = rgba
    z_mid = z_bottom + length * 0.5
    return textwrap.dedent(f"""\
    <?xml version="1.0" ?>
    <sdf version="1.6">
      <model name="{model_name}">
        <static>false</static>
        <allow_auto_disable>false</allow_auto_disable>
        <pose>0 0 0 0 0 0</pose>
        <link name="link">
          <collision name="collision">
            <pose>0 0 {z_mid:.3f} 0 0 0</pose>
            <geometry><cylinder><radius>{radius:.3f}</radius><length>{length:.3f}</length></cylinder></geometry>
            <surface><contact><collide_without_contact>true</collide_without_contact></contact></surface>
          </collision>
          <visual name="visual">
            <pose>0 0 {z_mid:.3f} 0 0 0</pose>
            <geometry><cylinder><radius>{radius:.3f}</radius><length>{length:.3f}</length></cylinder></geometry>
            <material>
              <ambient>{r} {g} {b} {a}</ambient>
              <diffuse>{r} {g} {b} {a}</diffuse>
            </material>
            <cast_shadows>false</cast_shadows>
            <transparency>{max(0.0, min(1.0, 1.0-a)):.3f}</transparency>
          </visual>
          <gravity>false</gravity>
          <kinematic>true</kinematic>
        </link>
      </model>
    </sdf>
    """)

def yaw_of_quat_z_w(qz: float, qw: float) -> float:
    return 2.0 * math.atan2(qz, qw)

def quat_z_w_of_yaw(yaw: float):
    return math.sin(yaw*0.5), math.cos(yaw*0.5)

def rot2d(x: float, y: float, yaw: float):
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (cy*x - sy*y, sy*x + cy*y)

class MultiProxy(Node):
    def __init__(self, args):
        super().__init__('human_multi_cylinder_proxies')

        # 参数
        self.prefix     = args.actor_prefix
        self.rate       = args.rate
        self.autospawn  = args.autospawn
        self.max_models = args.max_models if args.max_models > 0 else 9999

        # 几何（默认接近成人尺寸；可根据你的场景/雷达高度微调）
        # 躯干
        self.torso_r    = args.torso_radius
        self.torso_len  = args.torso_height
        self.torso_offx = args.torso_offx
        self.torso_offy = args.torso_offy
        self.torso_offz = args.torso_bottom_z

        # 头
        self.head_r     = args.head_radius
        self.head_len   = args.head_height
        self.head_offx  = args.head_offx
        self.head_offy  = args.head_offy
        self.head_offz  = args.head_bottom_z

        # 腿（左右对称，用一个 y 偏移控制外扩）
        self.leg_r      = args.leg_radius
        self.leg_len    = args.leg_height
        self.leg_offx   = args.leg_offx
        self.leg_offy   = args.leg_offy  # 绝对值用于左右
        self.leg_offz   = args.leg_bottom_z

        # 服务：set_entity_state
        self.set_cli = None
        for name in ['/set_entity_state', '/gazebo/set_entity_state']:
            cli = self.create_client(SetEntityState, name)
            if cli.wait_for_service(timeout_sec=2.0):
                self.set_cli = cli
                self.get_logger().info(f'使用 SetEntityState: {name}')
                break
        if self.set_cli is None:
            raise RuntimeError('未找到 SetEntityState 服务（需要 libgazebo_ros_state.so）')

        # 服务：spawn_entity（可选）
        self.spawn_cli = None
        if self.autospawn:
            self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
            if not self.spawn_cli.wait_for_service(timeout_sec=5.0):
                raise RuntimeError('等待 /spawn_entity 超时（需要 libgazebo_ros_factory.so）')

        # 订阅模型状态
        self.states = None
        self.create_subscription(ModelStates, '/model_states', self._on_states, 10)
        self.timer = self.create_timer(1.0/max(self.rate, 1e-3), self._on_timer)

        self.known = set()
        self.get_logger().info(f'prefix={self.prefix}, rate={self.rate}Hz, autospawn={self.autospawn}')

    def _on_states(self, msg: ModelStates):
        self.states = msg

    def _spawn_model(self, name: str, sdf_xml: str, x: float, y: float, yaw: float):
        if not self.autospawn or self.spawn_cli is None:
            return False
        qz, qw = quat_z_w_of_yaw(yaw)
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = 0.0  # z 由 SDF 内部的 collision/visual pose 决定了底/中高度，这里留 0
        pose.orientation.z = qz
        pose.orientation.w = qw

        req = SpawnEntity.Request()
        req.name = name
        req.xml = sdf_xml
        req.initial_pose = pose
        req.reference_frame = 'world'
        fut = self.spawn_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        ok = fut.done() and fut.result() and fut.result().success
        self.get_logger().info(f'生成 {name}: {ok}')
        return ok

    def _set_pose(self, name: str, x: float, y: float, yaw: float):
        qz, qw = quat_z_w_of_yaw(yaw)
        st = EntityState()
        st.name = name
        st.pose.position.x = float(x)
        st.pose.position.y = float(y)
        st.pose.position.z = 0.0
        st.pose.orientation.z = qz
        st.pose.orientation.w = qw
        req = SetEntityState.Request()
        req.state = st
        self.set_cli.call_async(req)

    def _ensure_component(self, base_name: str, comp: str, world_x: float, world_y: float, yaw: float):
        """
        确保某个部件存在（必要时生成），并更新其位姿
        comp in {"torso","head","legL","legR"}
        """
        model_name = f'{base_name}_{comp}{PROXY_SUFFIX}'

        # 不重复 spawn：依赖 states 列表
        existing = set(self.states.name) if self.states else set()

        # 选择尺寸 & 相对偏移（在 actor 坐标系）
        if comp == "torso":
            radius, length = self.torso_r, self.torso_len
            offx, offy, offz = self.torso_offx, self.torso_offy, self.torso_offz
        elif comp == "head":
            radius, length = self.head_r, self.head_len
            offx, offy, offz = self.head_offx, self.head_offy, self.head_offz
        elif comp == "legL":
            radius, length = self.leg_r, self.leg_len
            offx, offy, offz = self.leg_offx, abs(self.leg_offy), self.leg_offz
        elif comp == "legR":
            radius, length = self.leg_r, self.leg_len
            offx, offy, offz = self.leg_offx, -abs(self.leg_offy), self.leg_offz
        else:
            return

        # 旋转到世界系
        dx, dy = rot2d(offx, offy, yaw)
        px, py = world_x + dx, world_y + dy

        # 如需生成
        if (model_name not in existing) and (model_name not in self.known):
            sdf = sdf_cylinder_model(model_name, radius, length, z_bottom=offz)
            self._spawn_model(model_name, sdf, px, py, yaw)
            self.known.add(model_name)

        # 更新位姿
        self._set_pose(model_name, px, py, yaw)

    def _on_timer(self):
        if self.states is None:
            return
        count = 0
        for name, pose in zip(self.states.name, self.states.pose):
            if not name.startswith(self.prefix):           # 只处理 human_*
                continue
            if name.endswith(PROXY_SUFFIX):                # 忽略已有代理
                continue
            count += 1
            if count > self.max_models:
                break

            x = pose.position.x
            y = pose.position.y
            yaw = yaw_of_quat_z_w(pose.orientation.z, pose.orientation.w)

            # 为该 actor 的四个部件生成/同步
            for comp in COMP_NAMES:
                self._ensure_component(name, comp, x, y, yaw)

def main():
    ap = argparse.ArgumentParser(description='Gazebo Actor 的“多圆柱”碰撞代理（torso/head/legL/legR）自动生成与同步')
    ap.add_argument('--actor-prefix', type=str, default='human_', help='匹配的 actor 名字前缀（默认 human_）')
    ap.add_argument('--rate', type=float, default=20.0, help='同步频率Hz（默认20）')
    ap.add_argument('--autospawn', action='store_true', help='缺失时自动 /spawn_entity 生成代理')
    ap.add_argument('--max-models', type=int, default=0, help='最多跟踪的 actor 数（0=不限）')

    # 躯干（默认近似：半径0.25，高1.6；底面 z=0.0）
    ap.add_argument('--torso-radius', type=float, default=0.45)
    ap.add_argument('--torso-height', type=float, default=1.60)
    ap.add_argument('--torso-bottom-z', type=float, default=0.00)
    ap.add_argument('--torso-offx', type=float, default=0.00)
    ap.add_argument('--torso-offy', type=float, default=0.00)

    # 头（默认：半径0.18，高0.40；底面 z=1.60）
    ap.add_argument('--head-radius', type=float, default=0.18)
    ap.add_argument('--head-height', type=float, default=0.40)
    ap.add_argument('--head-bottom-z', type=float, default=1.60)
    ap.add_argument('--head-offx', type=float, default=0.00)
    ap.add_argument('--head-offy', type=float, default=0.00)

    # 腿（默认：半径0.15，高0.60；底面 z=0.00；左右 y=±0.10，x=0.00）
    ap.add_argument('--leg-radius', type=float, default=0.15)
    ap.add_argument('--leg-height', type=float, default=0.60)
    ap.add_argument('--leg-bottom-z', type=float, default=0.00)
    ap.add_argument('--leg-offx', type=float, default=0.00)
    ap.add_argument('--leg-offy', type=float, default=0.10)

    args = ap.parse_args()
    rclpy.init()
    try:
        node = MultiProxy(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
