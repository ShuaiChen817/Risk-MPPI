#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于论文方法(极坐标 Md / ΔMd / Mr) + 以 odom 为坐标系的局部风险图:

- 使用 /scan 构建极坐标危险图 Md(t)、ΔMd(t)、Mr(t)
- 然后投影成以机器人为中心的局部 OccupancyGrid
- OccupancyGrid 的 frame_id = 'odom'
- origin 会跟随机器人移动，使机器人始终位于栅格中心

假设 LiDAR 为 360° 扫描。
"""

import math
from typing import Optional, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import Pose

from tf2_ros import Buffer, TransformListener, TransformException


class ScanRiskMapBuilderPaperOdom(Node):
    def __init__(self):
        super().__init__('scan_risk_map_builder_paper_odom')

        # ------------------ ROS 参数 ------------------
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('risk_topic', '/risk_map')

        # LiDAR 感知最大半径（米）
        self.declare_parameter('r_max', 4.0)

        # 机器人与障碍物的“安全半径”（米），用于式(1)
        self.declare_parameter('r_robot', 0.25)
        self.declare_parameter('r_obstacle', 0.35)

        # 径向步长（米），论文中用 r，这里给成参数可调
        self.declare_parameter('radial_step', 0.25)

        # 高斯影响强度 σ（米），以及最大危险值 µ_max
        self.declare_parameter('sigma', 0.3)
        self.declare_parameter('mu_max', 1.0)

        # 动态危险图衰减系数 γ
        self.declare_parameter('gamma_decay', 0.9)

        # 最终输出 OccupancyGrid 的分辨率（米/格）
        self.declare_parameter('cart_resolution', 0.05)

        # 发布频率限制（Hz）
        self.declare_parameter('publish_rate_hz', 15.0)

        scan_topic = self.get_parameter('scan_topic').value
        risk_topic = self.get_parameter('risk_topic').value

        self.r_max = float(self.get_parameter('r_max').value)
        self.r_robot = float(self.get_parameter('r_robot').value)
        self.r_obs = float(self.get_parameter('r_obstacle').value)
        self.radial_step = float(self.get_parameter('radial_step').value)
        self.sigma = float(self.get_parameter('sigma').value)
        self.mu_max = float(self.get_parameter('mu_max').value)
        self.gamma = float(self.get_parameter('gamma_decay').value)
        self.cart_resolution = float(self.get_parameter('cart_resolution').value)

        # ------------------ 极坐标网格结构 ------------------
        self.grid_initialized = False
        self.theta_min: float = 0.0
        self.theta_max: float = 0.0
        self.dtheta: float = 0.0
        self.m_theta: int = 0          # 角度扇区数 m
        self.n_radial: int = 0         # 径向环数 n

        self.r_centers: Optional[np.ndarray] = None
        self.theta_centers: Optional[np.ndarray] = None
        self.R_grid: Optional[np.ndarray] = None
        self.Theta_grid: Optional[np.ndarray] = None
        self.X_grid: Optional[np.ndarray] = None
        self.Y_grid: Optional[np.ndarray] = None

        # 上一帧静态危险图 µ_ij(t-1)
        self.prev_mu: Optional[np.ndarray] = None

        # γ^i（径向衰减因子），shape (n,1)
        self.gamma_power: Optional[np.ndarray] = None

        # 发布频率控制
        self.last_pub_time = self.get_clock().now()

        # TF: odom -> scan_frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ROS 通信
        qos = 10
        self.create_subscription(LaserScan, scan_topic, self.scan_callback, qos)
        self.risk_pub = self.create_publisher(OccupancyGrid, risk_topic, qos)

        self.get_logger().info(
            f'[ScanRiskMapBuilderPaperOdom] init. scan="{scan_topic}", risk="{risk_topic}", '
            f'r_max={self.r_max}, r_robot={self.r_robot}, r_obs={self.r_obs}, '
            f'radial_step={self.radial_step}, cart_res={self.cart_resolution}'
        )

    # ====================== 极坐标网格初始化 ======================
    def init_polar_grid_from_scan(self, scan: LaserScan):
        """使用论文式(1)根据 /scan 初始化极坐标栅格。"""
        theta_min = float(scan.angle_min)
        theta_max = float(scan.angle_max)

        # 360° 雷达一般是 [-pi,pi] 或 [0,2pi]，这里直接按原范围用
        if theta_max <= theta_min:
            theta_min = -math.pi
            theta_max = math.pi

        self.theta_min = theta_min
        self.theta_max = theta_max

        # ---- 式(1): dθ = 1/2 * arcsin( r' / (r'+r) ) ----
        r = self.r_robot
        r_p = self.r_obs
        ratio = max(0.0, min(1.0, r_p / (r_p + r)))
        dtheta = 0.5 * math.asin(ratio)

        # 防止 dθ 太小，至少不比 LiDAR 的 angle_increment 小
        min_dtheta = abs(scan.angle_increment) if scan.angle_increment != 0.0 else (math.pi / 180.0)
        if dtheta < min_dtheta:
            dtheta = min_dtheta

        # m = (θmax - θmin)/dθ
        total_angle = self.theta_max - self.theta_min
        m = max(1, int(total_angle / dtheta))
        dtheta = total_angle / m  # 重新均匀分布

        self.dtheta = dtheta
        self.m_theta = m

        # ---- 径向：按 radial_step 划分 n 个环 ----
        dr = self.radial_step
        n = max(1, int(self.r_max / dr))
        self.n_radial = n

        # 每个环中心半径
        r_edges = np.linspace(0.0, n * dr, n + 1, dtype=np.float32)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        self.r_centers = r_centers

        # 每个扇区中心角度
        theta_edges = np.linspace(self.theta_min, self.theta_max, m + 1, dtype=np.float32)
        theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        self.theta_centers = theta_centers

        # 构造极坐标网格 + 对应笛卡尔坐标
        R_grid, Theta_grid = np.meshgrid(r_centers, theta_centers, indexing='ij')  # (n,m)
        X_grid = R_grid * np.cos(Theta_grid)
        Y_grid = R_grid * np.sin(Theta_grid)

        self.R_grid = R_grid
        self.Theta_grid = Theta_grid
        self.X_grid = X_grid
        self.Y_grid = Y_grid

        # γ^i, i=1..n
        radial_indices = np.arange(1, n + 1, dtype=np.float32).reshape(n, 1)
        self.gamma_power = np.power(self.gamma, radial_indices)

        self.grid_initialized = True

        self.get_logger().info(
            f'[ScanRiskMapBuilderPaperOdom] Polar grid ready: n={n} (dr={dr:.3f}), '
            f'm={m} (dθ={dtheta*180.0/math.pi:.2f}°), '
            f'θ∈[{self.theta_min:.2f},{self.theta_max:.2f}]'
        )

    # ====================== Scan 回调 ======================
    def scan_callback(self, scan: LaserScan):
        # 发布频率控制
        rate = float(self.get_parameter('publish_rate_hz').value)
        now = self.get_clock().now()
        if (now - self.last_pub_time).nanoseconds < (1e9 / max(1.0, rate)):
            return
        self.last_pub_time = now

        if not self.grid_initialized:
            self.init_polar_grid_from_scan(scan)

        if not scan.ranges:
            return

        # Step1: /scan → 有障碍的格子列表
        obstacle_cells = self._scan_to_obstacle_cells(scan)
        if not obstacle_cells:
            mu = np.zeros((self.n_radial, self.m_theta), dtype=np.float32)
        else:
            # Step2: 静态危险图 Md(t)
            mu = self._build_static_danger_map(obstacle_cells)

        # Step3: 动态危险图 ΔMd(t)
        phi = self._build_dynamic_danger_map(mu)

        # Step4: 最终风险图 Mr(t) = Md + ΔMd
        mr = mu + phi
        mr = np.clip(mr, 0.0, self.mu_max)

        self.prev_mu = mu.copy()

        # Step5: 投影到 odom 下的局部 OccupancyGrid
        grid_msg = self._project_risk_to_cartesian_odom(scan, mr)
        self.risk_pub.publish(grid_msg)

    # ====================== Step1: scan → 障碍格子 ======================
    def _scan_to_obstacle_cells(self, scan: LaserScan) -> List[Tuple[int, int]]:
        ranges = np.array(scan.ranges, dtype=np.float32)
        angle_min = float(scan.angle_min)
        angle_inc = float(scan.angle_increment)

        n = self.n_radial
        m = self.m_theta
        dr = self.radial_step

        obstacle_mask = np.zeros((n, m), dtype=bool)

        for k, r in enumerate(ranges):
            if not np.isfinite(r):
                continue
            if r <= 0.0 or r > self.r_max:
                continue

            theta = angle_min + k * angle_inc

            # 径向 index i
            i = int(r / dr)
            if i < 0 or i >= n:
                continue

            # 角度 index j
            # 这里仍然用 [theta_min, theta_max) 范围
            if theta < self.theta_min:
                theta += 2.0 * math.pi
            if theta >= self.theta_max:
                theta -= 2.0 * math.pi

            if theta < self.theta_min or theta >= self.theta_max:
                continue

            j = int((theta - self.theta_min) / self.dtheta)
            if j < 0 or j >= m:
                continue

            obstacle_mask[i, j] = True

        cells = list(zip(*np.where(obstacle_mask)))
        return cells

    # ====================== Step2: Md(t) ======================
    def _build_static_danger_map(self, obstacle_cells: List[Tuple[int, int]]) -> np.ndarray:
        n, m = self.n_radial, self.m_theta
        mu = np.zeros((n, m), dtype=np.float32)

        X = self.X_grid
        Y = self.Y_grid

        sigma = self.sigma
        sigma2 = 2.0 * sigma * sigma
        influence_radius = 0.75 # 论文使用 1m

        for (i0, j0) in obstacle_cells:
            x0 = X[i0, j0]
            y0 = Y[i0, j0]

            dx = X - x0
            dy = Y - y0
            dist = np.sqrt(dx * dx + dy * dy)

            mask = dist < influence_radius
            contrib = np.exp(-(dist[mask] * dist[mask]) / sigma2)
            #mu[mask] += contrib
            mu[mask] = np.maximum(mu[mask], contrib)

            np.clip(mu, 0.0, self.mu_max, out=mu)

        return mu

    # ====================== Step3: ΔMd(t) ======================
    def _build_dynamic_danger_map(self, mu: np.ndarray) -> np.ndarray:
        n, m = self.n_radial, self.m_theta
        phi = np.zeros((n, m), dtype=np.float32)

        if self.prev_mu is None or self.prev_mu.shape != mu.shape:
            return phi

        delta = mu - self.prev_mu
        phi = delta * self.gamma_power  # γ^i * Δµ_ij
        return phi

    # ====================== Step4: 投影到 odom 下的局部格子 ======================

    def _project_risk_to_cartesian_odom(self, scan: LaserScan, mr: np.ndarray) -> OccupancyGrid:
        """
        将 Mr(t) 投影到一个以机器人为中心的局部栅格：
        - frame_id = 'odom'
        - 通过 TF 查 odom -> scan_frame 的位姿 (tx, ty, yaw)
        - 栅格范围: [-r_max, r_max] x [-r_max, r_max]
        - origin = (tx - r_max, ty - r_max)
        假定 LiDAR 为 360°：只按 Rc<=r_max 裁剪，但用 (Xr,Yr) 算角度索引。
        """
        r_max = self.r_max
        res = self.cart_resolution
        size = int((2.0 * r_max) / res)
        if size < 1:
            size = 1

        width = size
        height = size

        # 机器人在 odom 下的位置/朝向
        scan_frame = scan.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform('odom', scan_frame, Time())
            tx = tf.transform.translation.x
            ty = tf.transform.translation.y
            q  = tf.transform.rotation
            yaw = self._yaw_from_quaternion(q)
        except TransformException as ex:
            self.get_logger().warn(
                f'lookup_transform(odom->{scan_frame}) failed: {ex}. '
                f'Using (0,0,0) for risk map.'
            )
            tx = 0.0
            ty = 0.0
            yaw = 0.0

        # 在 odom 下构造每个栅格中心的 (x,y)
        xs = (np.arange(width, dtype=np.float32) + 0.5) * res + (tx - r_max)
        ys = (np.arange(height, dtype=np.float32) + 0.5) * res + (ty - r_max)
        Xc, Yc = np.meshgrid(xs, ys)  # (H,W)

        # 先减平移，得到 odom 下、相对于雷达原点的向量
        dx = Xc - tx
        dy = Yc - ty

        # 再旋转到雷达坐标系 (scan_frame)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        Xr =  cos_y * dx + sin_y * dy    # 旋转 -yaw
        Yr = -sin_y * dx + cos_y * dy

        Rc = np.sqrt(Xr * Xr + Yr * Yr)
        Thetac = np.arctan2(Yr, Xr)      # [-pi, pi]

        # 360°：只按半径裁剪
        mask = (Rc <= r_max)

        grid = -np.ones((height, width), dtype=np.int8)

        if np.any(mask):
            theta_min = self.theta_min
            theta_max = self.theta_max
            two_pi = 2.0 * math.pi

            Thetac_adj = Thetac.copy()
            Thetac_adj[Thetac_adj < theta_min] += two_pi
            Thetac_adj[Thetac_adj >= theta_max] -= two_pi

            i_idx = np.floor(Rc[mask] / self.radial_step).astype(np.int32)
            i_idx = np.clip(i_idx, 0, self.n_radial - 1)

            j_idx = np.floor((Thetac_adj[mask] - theta_min) / self.dtheta).astype(np.int32)
            j_idx = np.clip(j_idx, 0, self.m_theta - 1)

            values = mr[i_idx, j_idx]  # 0..µ_max
            norm = np.clip(values / max(self.mu_max, 1e-6), 0.0, 1.0)
            risk_uint8 = (norm * 100.0).astype(np.int8)

            flat_grid = grid.reshape(-1)
            flat_mask = mask.reshape(-1)
            flat_grid[flat_mask] = risk_uint8
            grid = flat_grid.reshape(height, width)

        # 注意这里传的是 tx, ty（机器人在 odom 里的位置）
        return self._build_grid_msg_odom(grid, tx, ty)

    def _build_grid_msg_odom(self, grid: np.ndarray, rx: float, ry: float) -> OccupancyGrid:
        """把 numpy 栅格打包成 OccupancyGrid，frame='odom'。"""
        h, w = grid.shape

        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'

        msg.info.resolution = self.cart_resolution
        msg.info.width = w
        msg.info.height = h

        origin = Pose()
        origin.position.x = rx - self.r_max
        origin.position.y = ry - self.r_max
        origin.position.z = 0.0
        origin.orientation.w = 1.0
        msg.info.origin = origin

        msg.data = grid.flatten().tolist()
        return msg

    def _yaw_from_quaternion(self, q):
        """从 geometry_msgs/Quaternion 计算 yaw（绕 z 旋转角）"""
        x = q.x
        y = q.y
        z = q.z
        w = q.w
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def main():
    rclpy.init()
    node = ScanRiskMapBuilderPaperOdom()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
