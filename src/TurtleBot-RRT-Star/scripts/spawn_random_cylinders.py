#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, math, random, argparse, time
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.msg import ModelStates

IGNORES = {'ground_plane'}
DEFAULT_PREFIX = 'cyl_m_'   # 生成的圆柱名称前缀
DEFAULT_Z = 0.0
DEFAULT_CLEARANCE = 0.6     # 与任意已存在模型的最小间距（米），按你圆柱半径适当调

class RandCylSpawner(Node):
    def __init__(self, args):
        super().__init__('rand_cyl_spawner')
        self.n = args.n
        self.bbox = (args.xmin, args.xmax, args.ymin, args.ymax)  # 采样区域
        self.clearance = args.clearance
        self.z = args.z
        self.prefix = args.prefix
        self.sdf_path = os.path.expanduser(args.sdf)
        self.seed = args.seed
        if self.seed is not None:
            random.seed(self.seed)

        if not os.path.isfile(self.sdf_path):
            raise FileNotFoundError(self.sdf_path)
        with open(self.sdf_path, 'r') as f:
            self.sdf_xml = f.read()

        # /spawn_entity
        self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
        if not self.spawn_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('等待 /spawn_entity 超时（确认已加载 gazebo_ros_factory 插件）')

        # /model_states
        self.states: ModelStates = None
        self.sub = self.create_subscription(ModelStates, '/model_states',
                                            self._on_states, 10)

    def _on_states(self, msg: ModelStates):
        self.states = msg

    def _occupied_points(self) -> List[Tuple[float,float,str]]:
        """收集当前世界中所有模型的 (x,y,name)"""
        occupied = []
        if self.states is None:
            # 等一帧
            t0 = time.time()
            while self.states is None and (time.time()-t0)<2.0:
                rclpy.spin_once(self, timeout_sec=0.1)
            if self.states is None:
                self.get_logger().warn('2s 内未收到 /model_states，先按空场处理')
                return occupied

        for name, pose in zip(self.states.name, self.states.pose):
            if name in IGNORES:
                continue
            occupied.append((pose.position.x, pose.position.y, name))
        return occupied

    def _far_enough(self, x: float, y: float, occ: List[Tuple[float,float,str]]) -> bool:
        for ox, oy, _ in occ:
            if math.hypot(x-ox, y-oy) < self.clearance:
                return False
        return True

    def _sample_xy(self) -> Tuple[float,float]:
        xmin, xmax, ymin, ymax = self.bbox
        return random.uniform(xmin, xmax), random.uniform(ymin, ymax)

    def run(self):
        occ = self._occupied_points()              # 已有模型的占据点
        names, centers = [], []

        attempts_limit = 200
        for i in range(self.n):
            ok = False
            for _ in range(attempts_limit):
                x, y = self._sample_xy()
                if self._far_enough(x, y, occ):
                    ok = True
                    break
            if not ok:
                self.get_logger().warn(f'第 {i+1} 个点在 {attempts_limit} 次尝试内未找到合适位置，跳过')
                continue

            name = f'{self.prefix}{i+1}'
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = float(self.z)
            pose.orientation.w = 1.0

            req = SpawnEntity.Request()
            req.name = name
            req.xml = self.sdf_xml
            req.robot_namespace = ''
            req.initial_pose = pose
            req.reference_frame = 'world'

            fut = self.spawn_cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
            if not fut.done() or fut.result() is None or not fut.result().success:
                self.get_logger().warn(f'[{name}] 生成失败：{fut.result().status_message if fut.result() else "no result"}')
                continue

            self.get_logger().info(f'[{name}] 生成于 x={x:.2f}, y={y:.2f}, z={self.z:.2f}')
            names.append(name)
            centers.append([round(x,3), round(y,3)])
            occ.append((x, y, name))   # 加入占据，保证新点不与已生成的冲突

        # 输出给下一个脚本使用
        print('\n=== SPAWN 完成 ===')
        print('names =', json.dumps(names, ensure_ascii=False))
        print('centers_json =', json.dumps(centers, ensure_ascii=False))
        print('\n接下来运行 dynamic_obstacle_manager.py，例如：')
        print('python3 ~/turtlebot_ws/src/TurtleBot-RRT-Star/scripts/dynamic_obstacle_manager.py --ros-args \\')
        print(f"  -p names:='{json.dumps(names)}' \\")
        print(f"  -p centers:='{json.dumps(centers)}' \\")
        print("  -p ellipse_axes:='[0.6,0.4]' -p omega:=0.35 -p z:=0.0 -p rate_hz:=30.0")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8, help='生成圆柱数量（默认 8）')
    ap.add_argument('--bbox', nargs=4, type=float, metavar=('XMIN','XMAX','YMIN','YMAX'),
                    required=False, default=None,
                    help='采样区域矩形边界（世界坐标）')
    ap.add_argument('--xmin', type=float, default=-6.0)
    ap.add_argument('--xmax', type=float, default= 6.0)
    ap.add_argument('--ymin', type=float, default=-6.0)
    ap.add_argument('--ymax', type=float, default= 6.0)
    ap.add_argument('--clearance', type=float, default=DEFAULT_CLEARANCE,
                    help='与任意已有模型的最小间距（米）')
    ap.add_argument('--z', type=float, default=DEFAULT_Z, help='spawn 高度（米）')
    ap.add_argument('--sdf', type=str, required=True,
                    help='圆柱 SDF 路径（例如 ~/turtlebot_ws/src/TurtleBot-RRT-Star/world/cylinder_model/model.sdf）')
    ap.add_argument('--prefix', type=str, default=DEFAULT_PREFIX, help='名称前缀')
    ap.add_argument('--seed', type=int, default=None, help='随机数种子（可复现）')
    args = ap.parse_args()

    # 允许两种写法：--bbox xmin xmax ymin ymax，或单独四个参数
    if args.bbox is not None:
        args.xmin, args.xmax, args.ymin, args.ymax = args.bbox

    rclpy.init()
    node = RandCylSpawner(args)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
