#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import numpy as np

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

class RiskMapBuilder(Node):
    """
    Phase-1: 基于 Nav2 /local_costmap/costmap 的近似实现
    - Md: 对 costmap 做高斯扩散与归一化
    - ΔMd: 帧间差分并做衰减 (gamma)
    - Mr: Md + alpha * ΔMd
    发布：/risk_map (OccupancyGrid, 0..100)，未知格保留为 -1 以便可视化
    """
    def __init__(self):
        super().__init__('risk_map_builder')

        # 参数
        self.declare_parameter('input_costmap_topic', '/local_costmap/costmap')
        self.declare_parameter('publish_topic', '/risk_map')
        self.declare_parameter('sigma_cells', 3)        # 高斯半径(格)
        self.declare_parameter('gamma_delta', 0.6)      # ΔMd 衰减系数
        self.declare_parameter('alpha_delta', 0.7)      # Mr = Md + alpha*ΔMd
        self.declare_parameter('unknown_penalty', 0.0)  # 为未知格增加的最低风险(0~1)，Phase-1 先默认 0
        self.declare_parameter('publish_rate_hz', 15.0) # 限频发布
        # === Lidar FOV Mask 相关参数 ===
        self.declare_parameter('use_lidar_fov_mask', True)   # 是否启用雷达范围Mask
        self.declare_parameter('lidar_max_range', 3.5)       # 雷达最大有效半径（单位：米）
        self.declare_parameter('lidar_fov_angle', 6.283185307179586)  # 视场角，默认2π=360°


        self.prev_md = None
        self.prev_pub_stamp = self.get_clock().now()
        self.cached_info = None

        # 订阅 Nav2 的本地代价图
        in_topic = self.get_parameter('input_costmap_topic').value
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, in_topic, self.on_costmap, 10
        )

        # 发布风险图
        out_topic = self.get_parameter('publish_topic').value
        self.risk_pub = self.create_publisher(OccupancyGrid, out_topic, 10)

        self.get_logger().info(
            f'RiskMapBuilder started. input="{in_topic}", publish="{out_topic}"'
        )
    def apply_lidar_fov_mask(self, risk_grid: np.ndarray, resolution: float) -> np.ndarray:
        """
        只保留雷达半径范围内的 risk，超出范围的格子标为 unknown(-1)
        假设 local costmap 是 rolling window，机器人在栅格中心附近
        """
        if risk_grid is None:
            return risk_grid

        use_mask = bool(self.get_parameter('use_lidar_fov_mask').value)
        if not use_mask:
            return risk_grid

        lidar_range = float(self.get_parameter('lidar_max_range').value)
        fov_angle = float(self.get_parameter('lidar_fov_angle').value)

        H, W = risk_grid.shape
        # 以栅格中心作为机器人位置
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0

        # 构造每个格子的相对坐标（米）
        xs = (np.arange(W, dtype=np.float32) - cx) * resolution
        ys = (np.arange(H, dtype=np.float32) - cy) * resolution
        xx, yy = np.meshgrid(xs, ys)

        rr = np.sqrt(xx * xx + yy * yy)          # 到机器人的距离
        tt = np.arctan2(yy, xx)                  # 相对朝向角

        # 半径之外全设为 unknown
        mask_out = rr > lidar_range

        # 如果你以后需要限制视场角（比如180°），可以再加一个角度Mask：
        # half_fov = 0.5 * fov_angle
        # mask_out |= (np.abs(tt) > half_fov)

        # risk_grid 目前是 float(0~1 或 0~100)，这里用 -1 表示 unknown
        risk_grid = risk_grid.copy()
        risk_grid[mask_out] = -1.0

        return risk_grid


    def on_costmap(self, msg: OccupancyGrid):
        # 速率限制
        rate = float(self.get_parameter('publish_rate_hz').value)
        now = self.get_clock().now()
        if (now - self.prev_pub_stamp).nanoseconds < (1e9 / max(1.0, rate)):
            return
        self.prev_pub_stamp = now

        # 转数组
        H, W = msg.info.height, msg.info.width
        data = np.array(msg.data, dtype=np.int16).reshape(H, W)

        # 记录 meta（用于构造输出栅格）
        self.cached_info = msg.info

        # unknown mask
        unknown_mask = (data < 0)

        # 归一化到[0,1]
        grid = np.clip(data, 0, 100).astype(np.float32) / 100.0
        # Nav2 的 costmap 常用 0..100，个别配置也会 0..255；这里先做 0..100 处理
        # 若你用的是 0..255，可把 100 换成 255

        # 可选：对未知格施加“最低风险”（Phase-1 默认关闭）
        unk_pen = float(self.get_parameter('unknown_penalty').value)
        if unk_pen > 0.0:
            grid = np.where(unknown_mask, np.maximum(grid, unk_pen), grid)

        # ------- Md: 对栅格做高斯扩散（近障更危险） -------
        k = int(self.get_parameter('sigma_cells').value)
        md = grid.copy()
        if k > 0:
            xs = np.arange(-k, k + 1, dtype=np.float32)
            sigma = max(1.0, float(k) / 2.0)
            g1 = np.exp(-(xs ** 2) / (2 * sigma * sigma))
            g1 /= np.sum(g1)

            # 行卷积
            md = np.apply_along_axis(lambda r: np.convolve(r, g1, mode='same'), 1, md)
            # 列卷积
            md = np.apply_along_axis(lambda c: np.convolve(c, g1, mode='same'), 0, md)

        # ------- ΔMd: 帧间差分并做衰减(gamma) -------
        dm = np.zeros_like(md, dtype=np.float32)
        if self.prev_md is not None and self.prev_md.shape == md.shape:
            gamma = float(self.get_parameter('gamma_delta').value)
            dm = (md - self.prev_md) * gamma
        self.prev_md = md

        # ------- Mr: Md + alpha*ΔMd -------
        alpha = float(self.get_parameter('alpha_delta').value)
        mr = md + alpha * dm
        mr = np.clip(mr, 0.0, 1.0)


         # ====== 新增：只在雷达可见范围内保留风险 ======
        use_fov = bool(self.get_parameter('use_lidar_fov_mask').value)
        if use_fov:
            # 1) 读取雷达半径(米) 和 costmap 分辨率(米/格)
            lidar_range = float(self.get_parameter('lidar_max_range').value)
            res = msg.info.resolution

            # 2) 当前风险图的高宽
            H, W = mr.shape

            # 3) 假设机器人在 local_costmap 中心附近（Nav2 rolling_window 默认是这样）
            cx = (W - 1) / 2.0
            cy = (H - 1) / 2.0

            # 4) 为每个栅格生成它相对于中心的实际坐标 (米)
            xs = (np.arange(W, dtype=np.float32) - cx) * res  # 每一列的 x 坐标
            ys = (np.arange(H, dtype=np.float32) - cy) * res  # 每一行的 y 坐标
            xx, yy = np.meshgrid(xs, ys)                      # 得到每个栅格的 (x,y)

            # 5) 计算每个栅格到中心的距离 r
            rr = np.sqrt(xx * xx + yy * yy)

            # 6) r 大于雷达半径的格子，都认为“雷达看不到”
            outside_fov = rr > lidar_range

            # 7) 把这些格子也加入 unknown_mask（圆外 = unknown）
            unknown_mask = np.logical_or(unknown_mask, outside_fov)


         # 先把风险归一化到 0~100
        risk_u8 = (mr * 100.0).astype(np.int16)

        # 计算网格半径
        res = msg.info.resolution
        lidar_range = float(self.get_parameter('lidar_max_range').value)

        # 半径对应多少格（真实圆半径网格数）
        radius_cells = lidar_range / res

        # 为了让矩形小于圆，取内接正方形边长： R / sqrt(2)
        half_side_cells = int(radius_cells / np.sqrt(2.0)) - 1
        if half_side_cells < 1:
            half_side_cells = 1

        # 中心索引
        H, W = mr.shape
        cx = W // 2
        cy = H // 2

        # 裁剪窗口
        x0 = max(0, int(cx - half_side_cells))
        x1 = min(W, int(cx + half_side_cells))
        y0 = max(0, int(cy - half_side_cells))
        y1 = min(H, int(cy + half_side_cells))

        # 裁剪风险图与 unknown_mask
        risk_crop = risk_u8[y0:y1, x0:x1]
        unknown_crop = unknown_mask[y0:y1, x0:x1]

        # ==== 构造新的 OccupancyGrid ====
        out = OccupancyGrid()
        out.header = msg.header
        out.info = msg.info

        out.info.width = risk_crop.shape[1]
        out.info.height = risk_crop.shape[0]

        # 调整 origin，使裁剪后的位置正确
        out.info.origin.position.x = msg.info.origin.position.x + x0 * res
        out.info.origin.position.y = msg.info.origin.position.y + y0 * res

        # unknown 区域= -1，其余是风险
        final_grid = np.where(unknown_crop, -1, risk_crop).astype(np.int8)

        out.data = final_grid.flatten().tolist()
        self.risk_pub.publish(out)

def main():
    rclpy.init()
    node = RiskMapBuilder()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
