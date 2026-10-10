#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
riskmap_marker_viz.py

功能：
- 订阅 /risk_map (nav_msgs/OccupancyGrid)
- 只在“雷达半径范围内”的栅格位置撒彩色小点（Marker）
- 点越接近高风险，颜色越偏红；低风险偏蓝
- 点尺寸很小、贴地、半透明，尽量不遮挡绿色全局路径

使用方式：
- 单独开一个终端运行本节点：
    python3 src/Risk-MPPI/nodes/riskmap_marker_viz.py

- 在 RViz 中：
    1) 不再显示 /risk_map 的 OccupancyGrid（可以直接 Remove 掉）
    2) Add → Marker → Topic 选 /riskmap_markers
    3) 建议在 Path 的 Display 设置里，把 Z Offset 调成 0.05~0.1，
       这样绿色路径会“浮在”这些小点上方，不会被遮挡。
"""

import rclpy
from rclpy.node import Node

import numpy as np

from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class RiskmapMarkerViz(Node):
    def __init__(self):
        super().__init__('riskmap_marker_viz')

        # -------- 参数设置 --------
        # 输入 risk map topic
        self.declare_parameter('riskmap_topic', '/risk_map')
        # 输出 marker topic
        self.declare_parameter('marker_topic', '/riskmap_markers')
        # 小于等于这个值的风险不画（0=free 不画）
        self.declare_parameter('min_risk_value', 0)
        # 雷达半径（米）：只在这个圆内撒点（和 risk_map_builder 里的 lidar_max_range 对应）
        self.declare_parameter('lidar_max_range', 3.0)

        riskmap_topic = self.get_parameter('riskmap_topic').value
        marker_topic = self.get_parameter('marker_topic').value

        # 订阅 risk map
        self.sub = self.create_subscription(
            OccupancyGrid,
            riskmap_topic,
            self.on_map,
            10
        )
        # 发布 marker
        self.pub = self.create_publisher(Marker, marker_topic, 10)

        self.get_logger().info(
            f'[RiskmapMarkerViz] 启动 ✓ 订阅: {riskmap_topic} → 发布: {marker_topic}'
        )

    # ==================== 回调：收到 /risk_map ====================
    def on_map(self, msg: OccupancyGrid):
        # msg.data 是一维 list，需要 reshape 成 (H, W)
        if not msg.data:
            return

        H = msg.info.height
        W = msg.info.width
        data = np.array(msg.data, dtype=np.int16).reshape(H, W)

        # 构造 Marker
        marker = Marker()
        marker.header = msg.header
        marker.ns = 'riskmap'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # 每个点的大小 —— 用分辨率的一小部分，这样点比较小，不挡路
        res = msg.info.resolution
        marker.scale.x = res / 2.0   # 越小点越细
        marker.scale.y = res / 2.0
        marker.scale.z = 0.01        # 厚度无所谓，很小即可

        # 默认颜色占位（真正颜色逐点设置）
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0

        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        min_risk = int(self.get_parameter('min_risk_value').value)
        lidar_range = float(self.get_parameter('lidar_max_range').value)
        lidar_range_sq = lidar_range * lidar_range

        # 这里我们假设 local_costmap 是 rolling window，
        # 机器人在 costmap 中心附近。
        # 所以机器人在 map frame 中的大致位置可以估成：
        robot_cell_x = W / 2.0
        robot_cell_y = H / 2.0
        robot_x = origin_x + (robot_cell_x + 0.5) * res
        robot_y = origin_y + (robot_cell_y + 0.5) * res

        points = []
        colors = []

        # ===== 遍历每个格子，把风险值 > min_risk 且在雷达半径内的点画出来 =====
        for y in range(H):
            for x in range(W):
                v = int(data[y, x])

                # -1 = unknown 或 0 = free（以及 <= min_risk）就不画
                if v <= 0:
                    continue

                # 计算这个栅格中心的世界坐标 (wx, wy)
                wx = origin_x + (x + 0.5) * res
                wy = origin_y + (y + 0.5) * res

                # 只保留以机器人为圆心、雷达半径为半径的圆盘内的点
                dx = wx - robot_x
                dy = wy - robot_y
                if dx * dx + dy * dy > lidar_range_sq:
                    continue

                # 生成点
                p = Point()
                p.x = wx
                p.y = wy
                # 让点尽量贴地，这样路径（如果 Z Offset 设置 >0）会在上面，不被遮挡
                p.z = 0.0

                points.append(p)

                # 颜色映射：低风险 → 蓝，高风险 → 红
                # v: 1~100 → 0~1
                # 归一化到 0~1
                t = max(0.0, min(1.0, float(v) / 100.0))

                # （可选）增加一点对比度，让高风险更红，低风险更蓝
                gamma = 1.2
                t = t ** gamma

                color = ColorRGBA()
                # 不要太透明：基本不透明
                color.a = 0.9

                if t < 0.5:
                    # 0.0 ~ 0.5: 蓝(0,0,1) → 黄(1,1,0)
                    k = t / 0.5  # 0~1
                    color.r = k          # 0 → 1
                    color.g = k          # 0 → 1
                    color.b = 1.0 - k    # 1 → 0
                else:
                    # 0.5 ~ 1.0: 黄(1,1,0) → 红(1,0,0)
                    k = (t - 0.5) / 0.5  # 0~1
                    color.r = 1.0              # 一直是 1
                    color.g = 1.0 - k          # 1 → 0
                    color.b = 0.0              # 一直是 0


                colors.append(color)

        marker.points = points
        marker.colors = colors

        self.pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = RiskmapMarkerViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
