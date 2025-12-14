#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose


def rejection_sample(n: int, bbox: Tuple[float, float, float, float],
                     min_dist: float, max_tries_per_point: int = 2000,
                     seed: int = 42) -> List[Tuple[float, float]]:
    random.seed(seed)
    xmin, xmax, ymin, ymax = bbox
    pts: List[Tuple[float, float]] = []
    for _ in range(n):
        ok = False
        for _try in range(max_tries_per_point):
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)
            ok = True
            for (px, py) in pts:
                if (x - px) ** 2 + (y - py) ** 2 < (min_dist ** 2):
                    ok = False
                    break
            if ok:
                pts.append((x, y))
                break
        if not ok:
            break
    return pts


class ForestSpawner(Node):
    def __init__(self,
                 n_trees: int = 80,
                 bbox: Tuple[float, float, float, float] = (-10.0, 10.0, -10.0, 10.0),
                 clearance: float = 0.7,
                 prefix: str = "tree_",
                 seed: int = 123,
                 model_path: str = os.path.expanduser('~/.gazebo/models/pine_tree/model.sdf'),
                 z: float = 0.0,
                 random_yaw: bool = True):
        super().__init__('forest_spawner')
        self.cli = self.create_client(SpawnEntity, '/spawn_entity')
        self.get_logger().info("等待 /spawn_entity ...")
        if not self.cli.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("等待 /spawn_entity 超时。请确认 world 里加载了 libgazebo_ros_factory.so 插件。")

        # 读取树模型（带树冠/材质）
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"树模型不存在：{model_path}")
        with open(model_path, 'r') as f:
            self.tree_xml = f.read()

        self.n = n_trees
        self.bbox = bbox
        self.min_dist = clearance
        self.prefix = prefix
        self.seed = seed
        self.z = z
        self.random_yaw = random_yaw

        self.positions = rejection_sample(self.n, self.bbox, self.min_dist, seed=self.seed)
        if len(self.positions) < self.n:
            self.get_logger().warn(f"按最小间距只放下了 {len(self.positions)}/{self.n} 棵。"
                                   f"可适当减小 clearance 或扩大区域。")

    def spawn_all(self):
        spawned = 0
        for i, (x, y) in enumerate(self.positions):
            name = f"{self.prefix}{i+1}"
            yaw = random.uniform(-math.pi, math.pi) if self.random_yaw else 0.0

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = self.z
            # yaw -> quaternion
            pose.orientation.z = math.sin(yaw / 2.0)
            pose.orientation.w = math.cos(yaw / 2.0)

            req = SpawnEntity.Request()
            req.name = name
            req.xml = self.tree_xml           # 直接用树模型的 SDF
            req.robot_namespace = ""
            req.reference_frame = "world"
            req.initial_pose = pose

            fut = self.cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
            ok = fut.done() and fut.result() is not None
            if ok and "Successfully" in (fut.result().status_message or ""):
                spawned += 1
                self.get_logger().info(f"{name} @({x:.2f},{y:.2f},{self.z:.2f}), yaw={yaw:.2f} rad")
            else:
                msg = getattr(fut.result(), 'status_message', 'unknown')
                self.get_logger().warn(f"生成 {name} 失败：{msg}")
        self.get_logger().info(f"森林生成完成：{spawned}/{len(self.positions)} 棵。")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="在现有世界中生成静态森林（使用 pine_tree 模型）")
    parser.add_argument('--n', type=int, default=100, help='树的数量（默认 80，对应 20x20m、0.2棵/㎡）')
    parser.add_argument('--xmin', type=float, default=-10.0)
    parser.add_argument('--xmax', type=float, default=10.0)
    parser.add_argument('--ymin', type=float, default=-10.0)
    parser.add_argument('--ymax', type=float, default=10.0)
    parser.add_argument('--clearance', type=float, default=0.7, help='树间最小间距 m（默认 0.7）')
    parser.add_argument('--prefix', type=str, default='pine_', help='实体名前缀')
    parser.add_argument('--seed', type=int, default=123, help='随机种子（可重现）')
    parser.add_argument('--model', type=str,
                        default=os.path.expanduser('~/.gazebo/models/pine_tree/model.sdf'),
                        help='树模型 model.sdf 路径')
    parser.add_argument('--z', type=float, default=0.0, help='树基座的 z（地面一般 0）')
    parser.add_argument('--no_random_yaw', action='store_true', help='不随机朝向')
    args = parser.parse_args()

    rclpy.init()
    node = ForestSpawner(
        n_trees=args.n,
        bbox=(args.xmin, args.xmax, args.ymin, args.ymax),
        clearance=args.clearance,
        prefix=args.prefix,
        seed=args.seed,
        model_path=args.model,
        z=args.z,
        random_yaw=not args.no_random_yaw
    )
    node.spawn_all()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
