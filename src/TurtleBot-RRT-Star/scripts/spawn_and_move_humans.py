#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, time, json, random, argparse
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity, SetEntityState
from gazebo_msgs.msg import ModelStates, EntityState

IGNORES = {'ground_plane'}
DEFAULT_Z = 1.0                  # actor 通常皮肤脚底在 z≈1
DEFAULT_CLEARANCE = 0.6
DEFAULT_AX, DEFAULT_BY = 0.8, 0.6
DEFAULT_OMEGA = 0.35
DEFAULT_RATE = 30.0

class MoveEntities(Node):
    def __init__(self, args):
        super().__init__('move_entities')

        # ------- 参数 -------
        self.mode = args.mode                 # 'cylinders' or 'actors'
        self.n = args.n
        self.xmin, self.xmax = args.xmin, args.xmax
        self.ymin, self.ymax = args.ymin, args.ymax
        self.clearance = args.clearance
        self.z = args.z
        self.prefix = args.prefix
        self.ax, self.by = args.ax, args.by
        self.omega = args.omega
        self.rate_hz = args.rate
        self.seed = args.seed
        if self.seed is not None:
            random.seed(self.seed)

        self.actor_names = []
        if args.actor_names:
            self.actor_names = [s.strip() for s in args.actor_names.split(',') if s.strip()]

        # ------- Gazebo 服务 -------
        # set_entity_state 既可能是 /set_entity_state 也可能是 /gazebo/set_entity_state
        self.set_cli = None
        for name in ['/set_entity_state', '/gazebo/set_entity_state']:
            cli = self.create_client(SetEntityState, name)
            if cli.wait_for_service(timeout_sec=2.0):
                self.set_cli = cli
                self.get_logger().info(f'使用 SetEntityState: {name}')
                break
        if self.set_cli is None:
            raise RuntimeError('未找到 SetEntityState（需要 libgazebo_ros_state.so）')

        # cylinders 模式才需要 spawn
        if self.mode == 'cylinders':
            if not args.sdf:
                raise RuntimeError('--mode cylinders 需要提供 --sdf')
            self.sdf_path = os.path.expanduser(args.sdf)
            if not os.path.isfile(self.sdf_path):
                raise FileNotFoundError(self.sdf_path)
            with open(self.sdf_path, 'r') as f:
                self.sdf_xml = f.read()
            self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
            if not self.spawn_cli.wait_for_service(timeout_sec=5.0):
                raise RuntimeError('等待 /spawn_entity 超时（确认加载了 libgazebo_ros_factory.so）')

        # 订阅 /model_states
        self.states = None
        self.sub = self.create_subscription(ModelStates, '/model_states', self._on_states, 10)

        # 数据
        self.names: List[str] = []
        self.centers: List[List[float]] = []
        self.phases: List[float] = []

        self.t0 = time.monotonic()

        # 主流程
        if self.mode == 'cylinders':
            self._spawn_cylinders_then_move()
        else:
            self._pick_actors_then_move(args.actor_prefix)

    # ====== 读取世界状态 ======
    def _on_states(self, msg: ModelStates):
        self.states = msg

    def _wait_states(self, sec=2.0):
        t0 = time.monotonic()
        while self.states is None and (time.monotonic() - t0) < sec:
            rclpy.spin_once(self, timeout_sec=0.1)

    def _occupied_points(self) -> List[Tuple[float,float,str]]:
        self._wait_states(2.0)
        if self.states is None:
            self.get_logger().warn('2s 未收到 /model_states，先按空场处理')
            return []
        occ = []
        for name, pose in zip(self.states.name, self.states.pose):
            if name in IGNORES:
                continue
            occ.append((pose.position.x, pose.position.y, name))
        return occ

    # ====== 采样与碰撞间距检查 ======
    def _far_enough(self, x: float, y: float, occ: List[Tuple[float,float,str]]) -> bool:
        for ox, oy, _ in occ:
            if math.hypot(x - ox, y - oy) < self.clearance:
                return False
        return True

    def _sample_xy(self) -> Tuple[float, float]:
        return random.uniform(self.xmin, self.xmax), random.uniform(self.ymin, self.ymax)

    # ====== cylinders：spawn + move ======
    def _spawn_cylinders_then_move(self):
        occ = self._occupied_points()
        attempts = 250
        for i in range(self.n):
            ok = False
            for _ in range(attempts):
                x, y = self._sample_xy()
                if self._far_enough(x, y, occ):
                    ok = True
                    break
            if not ok:
                self.get_logger().warn(f'圆柱 {i+1} 选位失败（尝试 {attempts} 次），跳过')
                continue

            name = f'{self.prefix}{i+1}'
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = float(x), float(y), float(self.z)
            pose.orientation.w = 1.0

            req = SpawnEntity.Request()
            req.name = name
            req.xml = self.sdf_xml
            req.initial_pose = pose
            req.robot_namespace = ''
            req.reference_frame = 'world'

            fut = self.spawn_cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
            if not fut.done() or fut.result() is None or not fut.result().success:
                msg = fut.result().status_message if fut.result() else 'no result'
                self.get_logger().warn(f'[{name}] 生成失败：{msg}')
                continue

            self.get_logger().info(f'[{name}] 生成于 x={x:.2f}, y={y:.2f}, z={self.z:.2f}')
            self.names.append(name)
            self.centers.append([x, y])
            occ.append((x, y, name))

        if not self.names:
            raise RuntimeError('没有成功生成任何圆柱。')

        self._setup_phases_and_start()

    # ====== actors：选择已存在的 actor 并移动 ======
    def _pick_actors_then_move(self, actor_prefix: str):
        # 等 /model_states
        self._wait_states(3.0)
        if self.states is None:
            raise RuntimeError('未收到 /model_states，无法查找 actor')

        # 先从 /model_states 里取所有候选名字
        candidates = []
        for name, pose in zip(self.states.name, self.states.pose):
            if name in IGNORES: 
                continue
            if self.actor_names:
                if name in self.actor_names:
                    candidates.append((name, pose.position.x, pose.position.y))
            else:
                # 用前缀筛（默认 human_）
                if actor_prefix and name.startswith(actor_prefix):
                    candidates.append((name, pose.position.x, pose.position.y))

        if not candidates:
            raise RuntimeError('actors 模式：没有在 /model_states 里找到匹配的 actor 名称。'
                               '请用 --actor-names human_01,human_02 或 --actor-prefix human_ 先在 world 里放好 actor。')

        # 只取前 n 个
        candidates = candidates[:self.n] if self.n > 0 else candidates

        for name, x, y in candidates:
            self.names.append(name)
            self.centers.append([float(x), float(y)])

        self.get_logger().info(f'将移动 {len(self.names)} 个 actors：{self.names}')
        self._setup_phases_and_start()

    def _setup_phases_and_start(self):
        # 均匀相位，避免“同频同相”
        n = len(self.names)
        self.phases = [2.0 * math.pi * i / n for i in range(n)]

        print('\n=== READY ===')
        print('names   =', json.dumps(self.names, ensure_ascii=False))
        print('centers =', json.dumps([[round(x,3), round(y,3)] for x,y in self.centers], ensure_ascii=False))
        self.get_logger().info(
            f'MOVE N={len(self.names)} | axes=({self.ax},{self.by}) '
            f'omega={self.omega} | rate={self.rate_hz}Hz | z={self.z}'
        )
        self.timer = self.create_timer(1.0 / max(1e-3, self.rate_hz), self._on_timer)

    # ====== 定时器：更新位置（两种模式通用）=====
    def _on_timer(self):
        t = time.monotonic() - self.t0
        z = self.z

        for i, name in enumerate(self.names):
            cx, cy = self.centers[i]
            phi = self.phases[i]
            x = cx + self.ax * math.cos(self.omega * t + phi)
            y = cy + self.by * math.sin(self.omega * t + phi)

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = x, y, z
            pose.orientation.w = 1.0

            req = SetEntityState.Request()
            st = EntityState()
            st.name = name
            st.pose = pose
            req.state = st
            self.set_cli.call_async(req)

def main():
    ap = argparse.ArgumentParser(description='移动圆柱或已有 actor（Gazebo Classic, ROS2）')
    ap.add_argument('--mode', choices=['cylinders','actors'], default='actors',
                    help='actors：移动已存在的人形 Actor；cylinders：生成并移动圆柱（默认 actors）')
    # 通用/运动参数
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--xmin', type=float, default=-6.0)
    ap.add_argument('--xmax', type=float, default= 6.0)
    ap.add_argument('--ymin', type=float, default=-6.0)
    ap.add_argument('--ymax', type=float, default= 6.0)
    ap.add_argument('--clearance', type=float, default=DEFAULT_CLEARANCE)
    ap.add_argument('--z', type=float, default=DEFAULT_Z)
    ap.add_argument('--ax', type=float, default=DEFAULT_AX)
    ap.add_argument('--by', type=float, default=DEFAULT_BY)
    ap.add_argument('--omega', type=float, default=DEFAULT_OMEGA)
    ap.add_argument('--rate', type=float, default=DEFAULT_RATE)
    ap.add_argument('--seed', type=int, default=None)

    # cylinders 模式特有
    ap.add_argument('--sdf', type=str, default=None,
                    help='仅 cylinders 模式使用：圆柱 SDF 路径')

    # actors 模式特有
    ap.add_argument('--actor-prefix', type=str, default='human_',
                    help='actors 模式：按前缀从 /model_states 挑选（默认 human_）')
    ap.add_argument('--actor-names', type=str, default='',
                    help='actors 模式：逗号分隔的 actor 名称列表（优先于前缀）')

    args = ap.parse_args()

    rclpy.init()
    node = MoveEntities(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
