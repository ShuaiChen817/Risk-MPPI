#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, time, json, random, argparse
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity, SetEntityState
from gazebo_msgs.msg import ModelStates, EntityState


IGNORES = {'ground_plane'}       # 不计入碰撞间距检查的模型
DEFAULT_PREFIX = 'cyl_m_'
DEFAULT_Z = 0.0
DEFAULT_CLEARANCE = 0.6          # 与任意已有模型的最小距离（米）
DEFAULT_AX, DEFAULT_BY = 0.6, 0.4
DEFAULT_OMEGA = 0.35
DEFAULT_RATE = 30.0


class SpawnAndMove(Node):
    def __init__(self, args):
        super().__init__('spawn_and_move_cylinders')

        # ------- 参数 -------
        self.n = args.n
        self.xmin, self.xmax = args.xmin, args.xmax
        self.ymin, self.ymax = args.ymin, args.ymax
        self.clearance = args.clearance
        self.z = args.z
        self.prefix = args.prefix
        self.sdf_path = os.path.expanduser(args.sdf)
        self.ax, self.by = args.ax, args.by
        self.omega = args.omega
        self.rate_hz = args.rate
        self.seed = args.seed
        if self.seed is not None:
            random.seed(self.seed)

        if not os.path.isfile(self.sdf_path):
            raise FileNotFoundError(self.sdf_path)
        with open(self.sdf_path, 'r') as f:
            self.sdf_xml = f.read()

        # ------- Gazebo 服务 -------
        self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
        if not self.spawn_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('等待 /spawn_entity 超时（确认加载了 libgazebo_ros_factory.so）')

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

        # 订阅 /model_states 用于“避让已有模型”
        self.states = None
        self.sub = self.create_subscription(ModelStates, '/model_states',
                                            self._on_states, 10)

        # 数据：生成后写入
        self.names: List[str] = []
        self.centers: List[List[float]] = []
        self.phases: List[float] = []

        # 运动计时
        self.t0 = time.monotonic()

        # 执行流程：先 spawn，再启动移动定时器
        self._spawn_all()
        self._start_motion()

    # ====== 读取世界状态 ======
    def _on_states(self, msg: ModelStates):
        self.states = msg

    def _occupied_points(self) -> List[Tuple[float,float,str]]:
        # 最多等 2 秒拿到一帧
        t0 = time.monotonic()
        while self.states is None and (time.monotonic() - t0) < 2.0:
            rclpy.spin_once(self, timeout_sec=0.1)
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

    # ====== 批量生成 ======
    def _spawn_all(self):
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
                self.get_logger().warn(f'第 {i+1} 个找位失败（尝试 {attempts} 次），跳过')
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

        # 均匀相位，避免同频共相
        n = len(self.names)
        self.phases = [2.0 * math.pi * i / n for i in range(n)]

        # 便于你调试查看
        print('\n=== SPAWN 完成 ===')
        print('names =', json.dumps(self.names, ensure_ascii=False))
        print('centers =', json.dumps([[round(x,3), round(y,3)] for x,y in self.centers], ensure_ascii=False))

    # ====== 开始移动 ======
    def _start_motion(self):
        self.get_logger().info(
            f'MOVE N={len(self.names)} | axes=({self.ax},{self.by}) '
            f'omega={self.omega} | rate={self.rate_hz}Hz | z={self.z}'
        )
        self.timer = self.create_timer(1.0 / max(1e-3, self.rate_hz), self._on_timer)

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

            # 异步调用，不等待结果（避免阻塞导致“跳”）
            self.set_cli.call_async(req)


def main():
    ap = argparse.ArgumentParser(description='随机生成 + 自动移动动态圆柱（Gazebo Classic）')
    ap.add_argument('--n', type=int, default=8, help='圆柱数量（默认 8）')
    ap.add_argument('--xmin', type=float, default=-6.0)
    ap.add_argument('--xmax', type=float, default= 6.0)
    ap.add_argument('--ymin', type=float, default=-6.0)
    ap.add_argument('--ymax', type=float, default= 6.0)
    ap.add_argument('--clearance', type=float, default=DEFAULT_CLEARANCE,
                    help='与任意已有模型的最小间距（米）')
    ap.add_argument('--z', type=float, default=DEFAULT_Z, help='spawn 与运动高度（米）')
    ap.add_argument('--sdf', type=str, required=True,
                    help='圆柱 SDF 路径，比如 ~/turtlebot_ws/src/Risk-MPPI/world/cylinder_model/model.sdf')
    ap.add_argument('--prefix', type=str, default=DEFAULT_PREFIX, help='名称前缀（默认 cyl_m_）')
    ap.add_argument('--ax', type=float, default=DEFAULT_AX, help='椭圆半轴 a')
    ap.add_argument('--by', type=float, default=DEFAULT_BY, help='椭圆半轴 b')
    ap.add_argument('--omega', type=float, default=DEFAULT_OMEGA, help='角速度 rad/s')
    ap.add_argument('--rate', type=float, default=DEFAULT_RATE, help='更新频率 Hz')
    ap.add_argument('--seed', type=int, default=None, help='随机种子（可复现）')
    args = ap.parse_args()

    rclpy.init()
    node = SpawnAndMove(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
