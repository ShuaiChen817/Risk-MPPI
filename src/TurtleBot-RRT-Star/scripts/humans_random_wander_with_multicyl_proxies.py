#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机安置 + 游走避障 的 Gazebo Actor（human_*）
并为每个 Actor 生成/同步 4 段“多圆柱”碰撞代理（torso/head/legL/legR）。

- 仅依赖 ROS2 Humble + Gazebo Classic（gazebo_ros_state + gazebo_ros_factory）
- world 里需预放足够数量的 <actor name="human_XX">（建议 human_01..human_30）
- 建议把 <animation> 的 <interpolate_x>false</interpolate_x>，并把 <script><trajectory> 改成“原地站立”，
  由本脚本统一驱动 Actor 的位姿（这样视觉与碰撞 0 延迟一致；否则容易“动画根运动”相对漂移）

用法示例：
python3 humans_random_wander_with_multicyl_proxies.py \
  --n 12 --actor-prefix human_ --xmin -10 --xmax 10 --ymin -10 --ymax 10 \
  --clearance 0.7 --speed-min 0.3 --speed-max 0.9 --rate 30 --autospawn \
  --ros-args -p use_sim_time:=true
"""

import os, math, time, random, argparse, textwrap
from typing import List, Tuple, Dict

import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates, EntityState
from gazebo_msgs.srv import SetEntityState, SpawnEntity
from geometry_msgs.msg import Pose

IGNORE_NAMES = {'ground_plane', 'sun'}
PROXY_SUFFIX = "_proxy"
# 只使用躯干圆柱作为碰撞代理
COMP_NAMES = ("torso",)

def quat_from_yaw(yaw: float):
    return (0.0, 0.0, math.sin(yaw*0.5), math.cos(yaw*0.5))

def yaw_from_qz_qw(qz: float, qw: float):
    return 2.0 * math.atan2(qz, qw)

def rot2d(x: float, y: float, yaw: float):
    c, s = math.cos(yaw), math.sin(yaw)
    return c*x - s*y, s*x + c*y

def sdf_cylinder_model(model_name: str, radius: float, length: float,
                       z_bottom: float, rgba=(0,0,0,0.03)) -> str:
    r,g,b,a = rgba
    z_mid = z_bottom + length * 0.5
    return f"""<?xml version="1.0" ?>
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
        <material><ambient>{r} {g} {b} {a}</ambient><diffuse>{r} {g} {b} {a}</diffuse></material>
        <cast_shadows>false</cast_shadows>
        <transparency>{max(0.0, min(1.0, 1.0-a)):.3f}</transparency>
      </visual>
      <gravity>false</gravity>
      <kinematic>true</kinematic>
    </link>
  </model>
</sdf>"""

class HumansRandomWander(Node):
    def __init__(self, args):
        super().__init__('humans_random_wander_with_multicyl_proxies')
        # 区域与随机化
        self.n = args.n
        self.xmin, self.xmax = args.xmin, args.xmax
        self.ymin, self.ymax = args.ymin, args.ymax
        self.clearance = args.clearance
        self.prefix = args.actor_prefix
        self.rate = args.rate
        self.seed = args.seed
        if self.seed is not None:
            random.seed(self.seed)

        # 速度/避障
        self.speed_min = args.speed_min
        self.speed_max = args.speed_max
        self.turn_on_bounce = args.turn_on_bounce
        self.dt = 1.0 / max(1e-3, self.rate)

        # 多圆柱尺寸（可按需调）
        self.torso = dict(radius=args.torso_radius, height=args.torso_height, z0=args.torso_bottom_z, off=(0.0,0.0))
        self.head  = dict(radius=args.head_radius,  height=args.head_height,  z0=args.head_bottom_z,  off=(0.0,0.0))
        self.leg   = dict(radius=args.leg_radius,   height=args.leg_height,   z0=args.leg_bottom_z,   offx=args.leg_offx, offy=args.leg_offy)

        # ROS 服务
        self.set_cli = None
        for name in ['/set_entity_state', '/gazebo/set_entity_state']:
            cli = self.create_client(SetEntityState, name)
            if cli.wait_for_service(timeout_sec=2.0):
                self.set_cli = cli
                self.get_logger().info(f'使用 SetEntityState: {name}')
                break
        if self.set_cli is None:
            raise RuntimeError('未找到 SetEntityState 服务（需要 gazebo_ros_state）')

        self.spawn_cli = None
        if args.autospawn:
            self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
            if not self.spawn_cli.wait_for_service(timeout_sec=5.0):
                raise RuntimeError('未等到 /spawn_entity（需要 gazebo_ros_factory）')

        # 订阅世界状态
        self.states = None
        self.create_subscription(ModelStates, '/model_states', self._on_states, 10)

        # 数据
        self.actors: List[str] = []            # 被选中的 actor 名称
        self.pos: Dict[str, Tuple[float,float]] = {}   # 当前 xy
        self.yaw: Dict[str, float] = {}        # 朝向
        self.vel: Dict[str, Tuple[float,float]] = {}   # 速度向量

        self.known_spawned = set()             # 已生成过的代理
        self.timer = None

        # 流程
        self._init_world_and_start()

    # ------------------- 工具 -------------------
    def _on_states(self, msg: ModelStates):
        self.states = msg

    def _wait_states(self, sec=3.0):
        t0 = time.monotonic()
        while self.states is None and (time.monotonic()-t0) < sec:
            rclpy.spin_once(self, timeout_sec=0.1)

    def _static_obstacles(self) -> List[Tuple[float,float,str]]:
        if self.states is None:
            return []
        res = []
        for name, pose in zip(self.states.name, self.states.pose):
            if name in IGNORE_NAMES:
                continue
            # 忽略动态（human_* 及其代理）
            if name.startswith(self.prefix):
                continue
            if name.endswith(PROXY_SUFFIX):
                continue
            res.append((pose.position.x, pose.position.y, name))
        return res

    def _dist_ok(self, x, y, pts, min_d) -> bool:
        for ox, oy, _ in pts:
            if math.hypot(x-ox, y-oy) < min_d:
                return False
        return True

    def _sample_free_xy(self, occ, tries=400) -> Tuple[float,float]:
        for _ in range(tries):
            x = random.uniform(self.xmin, self.xmax)
            y = random.uniform(self.ymin, self.ymax)
            if self._dist_ok(x, y, occ, self.clearance):
                return x, y
        raise RuntimeError('在给定范围内找不到足够的无碰撞随机点，调大范围或减小 clearance')

    def _spawn_model_once(self, name: str, sdf: str, x: float, y: float, yaw: float):
        if self.spawn_cli is None:
            return
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = float(x), float(y), 0.0
        _, _, qz, qw = quat_from_yaw(yaw)
        pose.orientation.z, pose.orientation.w = qz, qw

        req = SpawnEntity.Request()
        req.name = name
        req.xml = sdf
        req.initial_pose = pose
        req.reference_frame = 'world'
        fut = self.spawn_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        ok = fut.done() and fut.result() and fut.result().success
        self.get_logger().info(f'spawn {name}: {ok}')

    def _set_entity_pose(self, name: str, x: float, y: float, yaw: float):
        _, _, qz, qw = quat_from_yaw(yaw)
        st = EntityState()
        st.name = name
        st.pose.position.x, st.pose.position.y, st.pose.position.z = float(x), float(y), 0.0
        st.pose.orientation.z, st.pose.orientation.w = qz, qw
        req = SetEntityState.Request(); req.state = st
        self.set_cli.call_async(req)

    # ------------------- 初始化并启动 -------------------
    def _init_world_and_start(self):
        self._wait_states(3.0)
        if self.states is None:
            self.get_logger().warn('未收到 /model_states，先照默认空场处理')

        # 1) 收集候选 actor
        candidates = [n for n in (self.states.name if self.states else [])
                      if n.startswith(self.prefix) and not n.endswith(PROXY_SUFFIX)]
        if not candidates:
            raise RuntimeError(f'未在 /model_states 里找到任何以 "{self.prefix}" 开头的 actor。请在 world 里预置 human_01.. 等')
        random.shuffle(candidates)
        self.actors = candidates[:self.n]

        # 2) 采样“安全出生点”
        statics = self._static_obstacles()
        occ = [(x,y,name) for (x,y,name) in statics]  # 先考虑静态
        # 也避免彼此之间过近
        for name in self.actors:
            x, y = self._sample_free_xy(occ, tries=500)
            yaw = random.uniform(-math.pi, math.pi)
            spd = random.uniform(self.speed_min, self.speed_max)
            vx, vy = spd*math.cos(yaw), spd*math.sin(yaw)
            self.pos[name] = (x, y)
            self.yaw[name] = yaw
            self.vel[name] = (vx, vy)
            # 占位，避免下一个挤在一起
            occ.append((x, y, name))

        # 3) 把 actor 放到初始位姿
        for name in self.actors:
            x,y = self.pos[name]; yaw = self.yaw[name]
            self._set_entity_pose(name, x, y, yaw)

        # 4) 启动定时器
        self.timer = self.create_timer(self.dt, self._on_timer)
        self.get_logger().info(f'选中 {len(self.actors)} 个 actors: {self.actors}')

    # ------------------- 运行时：游走 + 代理 -------------------
    def _ensure_multicyl_proxy(self, base: str, comp: str, world_x: float, world_y: float, yaw: float):
        # 选尺寸/偏移（actor 坐标系）
        if comp == "torso":
            radius, height, z0 = self.torso["radius"], self.torso["height"], self.torso["z0"]
            offx, offy = self.torso["off"]
        elif comp == "head":
            radius, height, z0 = self.head["radius"], self.head["height"], self.head["z0"]
            offx, offy = self.head["off"]
        elif comp == "legL":
            radius, height, z0 = self.leg["radius"], self.leg["height"], self.leg["z0"]
            offx, offy = self.leg["offx"], abs(self.leg["offy"])
        elif comp == "legR":
            radius, height, z0 = self.leg["radius"], self.leg["height"], self.leg["z0"]
            offx, offy = self.leg["offx"], -abs(self.leg["offy"])
        else:
            return
        # 旋转偏移 → 世界
        dx, dy = rot2d(offx, offy, yaw)
        px, py = world_x + dx, world_y + dy

        # 若未生成则生成一次
        model_name = f'{base}_{comp}{PROXY_SUFFIX}'
        existing = set(self.states.name) if self.states else set()
        if (model_name not in existing) and (model_name not in self.known_spawned) and (self.spawn_cli is not None):
            sdf = sdf_cylinder_model(model_name, radius, height, z_bottom=z0)
            self._spawn_model_once(model_name, sdf, px, py, yaw)
            self.known_spawned.add(model_name)
        # 持续对齐
        self._set_entity_pose(model_name, px, py, yaw)

    def _will_collide(self, nx: float, ny: float, name: str, statics, margin: float) -> bool:
        # 对静态障碍与其他人做距离检查
        for ox, oy, oname in statics:
            if math.hypot(nx-ox, ny-oy) < margin:
                return True
        for other in self.actors:
            if other == name: 
                continue
            ox, oy = self.pos[other]
            if math.hypot(nx-ox, ny-oy) < margin:
                return True
        # 越界
        if nx < self.xmin or nx > self.xmax or ny < self.ymin or ny > self.ymax:
            return True
        return False

    def _on_timer(self):
        if self.states is None:
            return
        statics = self._static_obstacles()

        # 逐人游走 + 反弹避障
        for name in self.actors:
            x,y = self.pos[name]
            vx, vy = self.vel[name]
            yaw = math.atan2(vy, vx) if (abs(vx)+abs(vy))>1e-6 else self.yaw[name]

            # 预判下一步
            nx, ny = x + vx*self.dt, y + vy*self.dt
            margin = self.clearance

            # 碰壁或将撞静态/他人 → 反弹（简单镜像）
            if self._will_collide(nx, ny, name, statics, margin):
                if self.turn_on_bounce:
                    # 试着反弹：翻转速度或随机换向
                    vx, vy = -vx, -vy
                    nx, ny = x + vx*self.dt, y + vy*self.dt
                    # 仍不行就随机新方向
                    if self._will_collide(nx, ny, name, statics, margin):
                        ang = random.uniform(-math.pi, math.pi)
                        spd = random.uniform(self.speed_min, self.speed_max)
                        vx, vy = spd*math.cos(ang), spd*math.sin(ang)
                        nx, ny = x + vx*self.dt, y + vy*self.dt
                else:
                    # 保守起见直接抽新方向
                    ang = random.uniform(-math.pi, math.pi)
                    spd = random.uniform(self.speed_min, self.speed_max)
                    vx, vy = spd*math.cos(ang), spd*math.sin(ang)
                    nx, ny = x + vx*self.dt, y + vy*self.dt

            # 确认更新
            self.pos[name] = (nx, ny)
            self.vel[name] = (vx, vy)
            yaw = math.atan2(vy, vx); self.yaw[name] = yaw

            # 先更新 actor
            self._set_entity_pose(name, nx, ny, yaw)
            # 再更新四段代理
            for comp in COMP_NAMES:
                self._ensure_multicyl_proxy(name, comp, nx, ny, yaw)

def main():
    ap = argparse.ArgumentParser(description='随机安置 + 游走避障 的 Actor（human_*） + 多圆柱代理（torso/head/legL/legR）')
    ap.add_argument('--n', type=int, default=8, help='启用的 actor 数量')
    ap.add_argument('--actor-prefix', type=str, default='human_', help='actor 名字前缀')
    ap.add_argument('--xmin', type=float, default=-6.0)
    ap.add_argument('--xmax', type=float, default= 6.0)
    ap.add_argument('--ymin', type=float, default=-6.0)
    ap.add_argument('--ymax', type=float, default= 6.0)
    ap.add_argument('--clearance', type=float, default=0.7, help='与静态/他人的间距（m）')
    ap.add_argument('--rate', type=float, default=30.0)
    ap.add_argument('--speed-min', type=float, default=0.3)
    ap.add_argument('--speed-max', type=float, default=0.9)
    ap.add_argument('--turn-on-bounce', action='store_true', help='启用“反弹”策略（否则碰撞时抽新方向）')
    ap.add_argument('--autospawn', action='store_true', help='缺少代理时自动 /spawn_entity 生成')
    ap.add_argument('--seed', type=int, default=None)
    # 多圆柱尺寸（可调）
    ap.add_argument('--torso-radius', type=float, default=0.28)
    ap.add_argument('--torso-height', type=float, default=1.60)
    ap.add_argument('--torso-bottom-z', type=float, default=0.00)
    ap.add_argument('--head-radius', type=float, default=0.18)
    ap.add_argument('--head-height', type=float, default=0.40)
    ap.add_argument('--head-bottom-z', type=float, default=1.60)
    ap.add_argument('--leg-radius', type=float, default=0.15)
    ap.add_argument('--leg-height', type=float, default=0.60)
    ap.add_argument('--leg-bottom-z', type=float, default=0.00)
    ap.add_argument('--leg-offx', type=float, default=0.00)
    ap.add_argument('--leg-offy', type=float, default=0.12)

    args = ap.parse_args()
    rclpy.init()
    try:
        node = HumansRandomWander(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
