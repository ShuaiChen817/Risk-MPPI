#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import json
import ast
from typing import List

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SetEntityState, GetEntityState
from gazebo_msgs.msg import EntityState


class DynamicObstacleManager(Node):
    """
    按 dangermap 思路：
      - 障碍物沿参数化轨迹运动（椭圆/圆）
      - 关键改动：
          * 用 ROS 模拟时钟 self.get_clock().now() 计算 t
          * 服务调用异步 fire-and-forget，不阻塞等待
    """

    def __init__(self):
        super().__init__(
            'dynamic_obstacle_manager',
            allow_undeclared_parameters=False,
            automatically_declare_parameters_from_overrides=False
        )

        # -------- 参数（和你原来一致，可 --ros-args -p 覆盖）--------
        self.declare_parameter('names', ["cyl_m_1", "cyl_m_2", "cyl_m_3", "cyl_m_4", "cyl_m_5", "cyl_m_6", "cyl_m_7", "cyl_m_8"])
        self.declare_parameter('centers', '[[2.088, 2.749], [1.044, -3.608], [4.577, -4.084], [-1.869, -5.191], [-0.035, -5.408], [5.723, 1.526], [4.603, -0.6], [-3.047, -3.008]]')
        self.declare_parameter('ellipse_axes', [0.5, 0.35])
        self.declare_parameter('omega', 0.25)
        self.declare_parameter('phases', [])
        self.declare_parameter('z', 0.0)
        self.declare_parameter('rate_hz', 30.0)   # 稍提一点频率更丝滑
        self.declare_parameter('t0_offset', 0.0)

        # 读取参数
        self.names: List[str] = list(self.get_parameter('names').value)

        _centers_raw = self.get_parameter('centers').value
        if isinstance(_centers_raw, str):
            try:
                self.centers = json.loads(_centers_raw)
            except Exception:
                self.centers = ast.literal_eval(_centers_raw)
        else:
            self.centers = list(_centers_raw)

        axes = list(self.get_parameter('ellipse_axes').value)
        self.ax: float = float(axes[0])
        self.by: float = float(axes[1])
        self.omega: float = float(self.get_parameter('omega').value)
        self.z: float = float(self.get_parameter('z').value)
        self.rate_hz: float = float(self.get_parameter('rate_hz').value)
        self.t0_offset: float = float(self.get_parameter('t0_offset').value)

        phases_param = list(self.get_parameter('phases').value)
        if phases_param:
            self.phases: List[float] = [float(p) for p in phases_param]
        else:
            n = max(1, len(self.names))
            self.phases = [2.0 * math.pi * i / n for i in range(n)]

        if len(self.centers) != len(self.names):
            raise ValueError('centers 的长度必须与 names 对应')
        if len(self.phases) != len(self.names):
            self.get_logger().warn('phases 长度与 names 不一致，自动匹配')
            self.phases = (self.phases + [0.0] * len(self.names))[:len(self.names)]

        # -------- 服务：优先 /set_entity_state，退回 /gazebo/set_entity_state --------
        svc_candidates = ['/set_entity_state', '/gazebo/set_entity_state']
        self.set_cli = None
        for svc in svc_candidates:
            cli = self.create_client(SetEntityState, svc)
            if cli.wait_for_service(timeout_sec=1.0):
                self.set_cli = cli
                self.get_logger().info(f'使用 SetEntityState 服务: {svc}')
                break
            self.destroy_client(cli)
        if self.set_cli is None:
            self.get_logger().error('找不到 SetEntityState 服务（/set_entity_state 或 /gazebo/set_entity_state）')
            raise SystemExit(1)

        # 仅做启动时校验（不阻塞运行）
        get_candidates = ['/get_entity_state', '/gazebo/get_entity_state']
        self.get_cli = None
        for svc in get_candidates:
            cli = self.create_client(GetEntityState, svc)
            if cli.wait_for_service(timeout_sec=0.5):
                self.get_cli = cli
                break
            self.destroy_client(cli)

        if self.get_cli is None:
            self.get_logger().warn('没有 get_entity_state 服务，跳过名称校验')
        else:
            missing = []
            for nm in self.names:
                req = GetEntityState.Request()
                req.name = nm
                fut = self.get_cli.call_async(req)
                # 不阻塞，给一点时间后检查（可选）
                rclpy.spin_until_future_complete(self, fut, timeout_sec=0.1)
                if not fut.done() or not fut.result() or not fut.result().success:
                    missing.append(nm)
            if missing:
                self.get_logger().warn(f'以下实体未找到：{missing}（请确认已 spawn 且名字一致）')

        # -------- 日志 --------
        self.get_logger().info(
            f"Moving: {self.names}\n"
            f"Centers: {self.centers}, ellipse_axes=({self.ax},{self.by}), "
            f"omega={self.omega}, rate={self.rate_hz}Hz, z={self.z}"
        )

        # 用“ROS 时钟”设定 t0（有 /clock 且本节点 use_sim_time=true 时即为仿真时钟）
        self.t0_ros = self._now_sec() + self.t0_offset

        # 定时器（主循环）
        self.timer = self.create_timer(1.0 / max(1e-3, self.rate_hz), self._on_timer)

    def _now_sec(self) -> float:
        """返回 ROS 时钟的秒（若 use_sim_time=true 则为仿真时间）。"""
        return self.get_clock().now().nanoseconds * 1e-9

    # ==== 核心：更新实体（异步 fire-and-forget，不等待回包）====
    def _on_timer(self):
        t = self._now_sec() - self.t0_ros
        z = self.z

        for i, name in enumerate(self.names):
            cx, cy = float(self.centers[i][0]), float(self.centers[i][1])
            phi = float(self.phases[i])
            x = cx + self.ax * math.cos(self.omega * t + phi)
            y = cy + self.by * math.sin(self.omega * t + phi)

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.w = 1.0

            req = SetEntityState.Request()
            st = EntityState()
            st.name = name
            st.pose = pose
            req.state = st

            # 不等待结果，直接异步发送，避免阻塞导致“丢帧→大跳步”
            self.set_cli.call_async(req)


def main():
    rclpy.init()
    node = DynamicObstacleManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
