#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GlobalPathAssessor
------------------
订阅:
  - /plan_raw (nav_msgs/Path): 原始全局路径 (A*/RRT* 输出)
  - /risk_map (nav_msgs/OccupancyGrid): 论文版风险图 Mr = Md + ΔMd

发布:
  - /plan (nav_msgs/Path): 经过风险 + 不确定性 + 长度评估后选出的最佳路径

核心功能:
  1) 围绕 /plan_raw 生成多条左右偏移 (lateral offsets) 的候选路径
  2) 在 risk_map 上对每条路径积分:
       - 路径长度 J_len
       - 路径平均风险 J_risk
       - 路径不确定性 J_unc (unknown/越界比重)
  3) 使用多目标代价:
       J = w_L * J_len_norm + w_R * J_risk_norm + w_U * J_unc
     选出代价最小的路径发布到 /plan
"""

import math
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped


class GlobalPathAssessor(Node):
    def __init__(self):
        super().__init__('global_path_assessor')

        # === QoS 设置 ===
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )

        # === 参数 ===
        self.declare_parameter('risk_map_topic', '/risk_map')
        self.declare_parameter('plan_raw_topic', '/plan_raw')
        self.declare_parameter('plan_topic', '/plan')

        # lateral offsets (m): 中间0, 左右正负偏移
        self.declare_parameter('lateral_offsets', [-0.6, -0.3, 0.0, 0.3, 0.6])

        # 风险图相关
        self.declare_parameter('risk_collision_threshold', 85)  # >= 这个值算硬碰撞
        self.declare_parameter('unknown_as_high_risk', 0.7)     # unknown cell 视为中高risk(0~1)

        # 路径采样步长(米)
        self.declare_parameter('sample_step', 0.10)

        # 多目标权重
        self.declare_parameter('weight_length', 1.0)
        self.declare_parameter('weight_risk', 3.0)
        self.declare_parameter('weight_uncertainty', 2.0)

        # === 读取参数 ===
        self.risk_map_topic = self.get_parameter('risk_map_topic').get_parameter_value().string_value
        self.plan_raw_topic = self.get_parameter('plan_raw_topic').get_parameter_value().string_value
        self.plan_topic = self.get_parameter('plan_topic').get_parameter_value().string_value

        self.lateral_offsets = [
            float(x) for x in self.get_parameter('lateral_offsets').get_parameter_value().double_array_value
        ]
        if len(self.lateral_offsets) == 0:
            self.lateral_offsets = [0.0]

        self.risk_collision_threshold = float(
            self.get_parameter('risk_collision_threshold').get_parameter_value().double_value
        )
        self.unknown_as_high_risk = float(
            self.get_parameter('unknown_as_high_risk').get_parameter_value().double_value
        )

        self.sample_step = float(self.get_parameter('sample_step').get_parameter_value().double_value)

        self.weight_length = float(self.get_parameter('weight_length').get_parameter_value().double_value)
        self.weight_risk = float(self.get_parameter('weight_risk').get_parameter_value().double_value)
        self.weight_uncertainty = float(self.get_parameter('weight_uncertainty').get_parameter_value().double_value)

        # === 订阅和发布 ===
        self.risk_map_sub = self.create_subscription(
            OccupancyGrid,
            self.risk_map_topic,
            self.risk_map_callback,
            qos
        )
        self.plan_raw_sub = self.create_subscription(
            Path,
            self.plan_raw_topic,
            self.plan_raw_callback,
            qos
        )
        self.plan_pub = self.create_publisher(Path, self.plan_topic, qos)

        # candidate path publishers（一个偏移一条线）
        self.candidate_publishers = []
        for i in range(len(self.lateral_offsets)):
            pub = self.create_publisher(Path, f'/candidate_path_{i}', 10)
            self.candidate_publishers.append(pub)


        # === 内部缓存 ===
        self.risk_map = None          # np.ndarray (H, W)
        self.risk_origin_x = 0.0
        self.risk_origin_y = 0.0
        self.risk_resolution = 0.05
        self.risk_width = 0
        self.risk_height = 0
        self.risk_frame = 'odom'

        self.get_logger().info(
            f'GlobalPathAssessor started. '
            f'risk_map_topic={self.risk_map_topic}, '
            f'plan_raw_topic={self.plan_raw_topic}, '
            f'plan_topic={self.plan_topic}, '
            f'lateral_offsets={self.lateral_offsets}'
        )

    # =========================
    #   回调: 接收 RiskMap
    # =========================
    def risk_map_callback(self, msg: OccupancyGrid):
        w = msg.info.width
        h = msg.info.height
        if w == 0 or h == 0:
            return

        data = np.array(msg.data, dtype=np.int16).reshape(h, w)

        self.risk_map = data
        self.risk_origin_x = msg.info.origin.position.x
        self.risk_origin_y = msg.info.origin.position.y
        self.risk_resolution = float(msg.info.resolution)
        self.risk_width = w
        self.risk_height = h
        self.risk_frame = msg.header.frame_id or 'odom'

    # =========================
    #   回调: 接收原始全局路径
    # =========================
    def plan_raw_callback(self, msg: Path):
        if self.risk_map is None:
            self.get_logger().warn('No risk map yet, republishing /plan_raw as /plan.')
            # 如果还没有riskmap，就直接原样转发
            self.plan_pub.publish(msg)
            return

        if len(msg.poses) < 2:
            self.get_logger().warn('Received /plan_raw with too few poses.')
            self.plan_pub.publish(msg)
            return

        # 1) 生成多条偏移路径
        candidate_paths = self.generate_offset_paths(msg)
        # ==== 发布所有候选路径用于 RViz 可视化 ====
        for i, p in enumerate(candidate_paths):
            p.header.stamp = self.get_clock().now().to_msg()
            self.candidate_publishers[i].publish(p)


        # 2) 对每条路径在 risk_map 上积分，算 J_len, J_risk, J_unc
        costs = []
        valid_paths = []

        for idx, path in enumerate(candidate_paths):
            J_len, J_risk, J_unc, valid = self.evaluate_path(path)
            if not valid:
                # 硬碰撞 or 全部越界，丢弃
                continue
            costs.append((J_len, J_risk, J_unc, idx))
            valid_paths.append((path, J_len, J_risk, J_unc))

        if not costs:
            self.get_logger().warn('All candidate paths invalid, using raw path.')
            self.plan_pub.publish(msg)
            return

        # 3) 多目标归一化 + 综合代价
        best_idx = self.select_best_path(costs)
        best_path, J_len, J_risk, J_unc = valid_paths[best_idx]

        self.get_logger().info(
            f'Selected path #{best_idx} '
            f'(len={J_len:.2f}, risk={J_risk:.3f}, unc={J_unc:.3f}) '
            f'from {len(candidate_paths)} candidates.'
        )

        # 4) 发布最佳路径到 /plan
        best_path.header = msg.header  # 保持同一参考系和时间戳
        self.plan_pub.publish(best_path)

    # =========================
    #   生成偏移路径
    # =========================
    def generate_offset_paths(self, center_path: Path) -> List[Path]:
        """
        根据原始路径生成多条左右偏移的候选路径。
        :param center_path: 原始 Path
        :return: List[Path] 候选路径列表 (与 lateral_offsets 对应)
        """
        poses = center_path.poses
        n = len(poses)
        if n < 2:
            return [center_path]

        # 1) 计算每个点对应的切线方向 theta
        thetas = []
        for i in range(n):
            if i == 0:
                x1 = poses[i].pose.position.x
                y1 = poses[i].pose.position.y
                x2 = poses[i + 1].pose.position.x
                y2 = poses[i + 1].pose.position.y
            elif i == n - 1:
                x1 = poses[i - 1].pose.position.x
                y1 = poses[i - 1].pose.position.y
                x2 = poses[i].pose.position.x
                y2 = poses[i].pose.position.y
            else:
                x1 = poses[i - 1].pose.position.x
                y1 = poses[i - 1].pose.position.y
                x2 = poses[i + 1].pose.position.x
                y2 = poses[i + 1].pose.position.y

            dx = x2 - x1
            dy = y2 - y1
            theta = math.atan2(dy, dx)
            thetas.append(theta)

        # 2) 为每个 offset 生成一条新路径
        candidate_paths: List[Path] = []

        for d in self.lateral_offsets:
            new_path = Path()
            new_path.header = center_path.header
            new_poses: List[PoseStamped] = []

            for i in range(n):
                base_pose = poses[i]
                x = base_pose.pose.position.x
                y = base_pose.pose.position.y
                theta = thetas[i]

                # 法线方向 (左= +d, 右= -d)
                nx = -math.sin(theta)
                ny = math.cos(theta)

                x_d = x + d * nx
                y_d = y + d * ny

                p = PoseStamped()
                p.header = base_pose.header
                p.pose = base_pose.pose
                p.pose.position.x = x_d
                p.pose.position.y = y_d

                new_poses.append(p)

            new_path.poses = new_poses
            candidate_paths.append(new_path)

        return candidate_paths

    # =========================
    #   在 RiskMap 上评估单条路径
    # =========================
    def evaluate_path(self, path: Path) -> Tuple[float, float, float, bool]:
        """
        对一条路径进行评估:
          - J_len: 总长度
          - J_risk: 平均风险 (0~1)
          - J_unc: 不确定性比例 (0~1)
          - valid: 若路径包含硬碰撞或完全越界则 False
        """
        if self.risk_map is None:
            return 0.0, 0.0, 0.0, False

        pts = [(p.pose.position.x, p.pose.position.y) for p in path.poses]
        if len(pts) < 2:
            return 0.0, 0.0, 0.0, False

        # 1) 按 sample_step 对路径进行均匀采样
        samples: List[Tuple[float, float]] = []
        total_length = 0.0

        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-6:
                continue

            # 该段需要采样多少个点
            num_steps = max(1, int(seg_len / self.sample_step))
            for k in range(num_steps):
                t = k / float(num_steps)
                xs = x1 + t * dx
                ys = y1 + t * dy
                samples.append((xs, ys))
            total_length += seg_len

        if total_length < 1e-6 or len(samples) == 0:
            return 0.0, 0.0, 0.0, False

        # 2) 对每个采样点在 risk_map 上查值
        risk_sum = 0.0
        unknown_length = 0.0
        step_len = total_length / len(samples)

        collision_detected = False

        for (xs, ys) in samples:
            in_map, val = self.lookup_risk(xs, ys)
            if not in_map:
                # 越界视为 unknown / high_uncertainty
                unknown_length += step_len
                # 也可以选择给一定risk, 这里使用 unknown_as_high_risk
                risk_sum += self.unknown_as_high_risk * step_len
                continue

            if val < 0:
                # unknown cell
                unknown_length += step_len
                risk_sum += self.unknown_as_high_risk * step_len
                continue

            # 归一化到 0~1
            r_norm = max(0.0, min(1.0, val / 100.0))
            risk_sum += r_norm * step_len

            # 硬碰撞判断
            #if val >= self.risk_collision_threshold:
                #collision_detected = True
               # break

        if collision_detected:
            return total_length, 1.0, 1.0, False  # 标记为无效路径

        # 平均风险
        J_risk = risk_sum / max(total_length, 1e-6)
        # 不确定性比例
        J_unc = unknown_length / max(total_length, 1e-6)

        return total_length, J_risk, J_unc, True

    # =========================
    #   risk_map 查表
    # =========================
    def lookup_risk(self, x: float, y: float) -> Tuple[bool, float]:
        """
        把世界坐标 (x,y) 映射到 risk_map 栅格，并返回 risk 值.
        :return: (in_map, risk_value)
                 in_map=False 表示越界; risk_value 为 int(-1 or 0~100)
        """
        # 将世界坐标转成栅格索引
        col = int((x - self.risk_origin_x) / self.risk_resolution)
        row = int((y - self.risk_origin_y) / self.risk_resolution)

        if row < 0 or row >= self.risk_height or col < 0 or col >= self.risk_width:
            return False, -1.0

        val = float(self.risk_map[row, col])
        return True, val

    # =========================
    #   多目标归一化 & 选最优路径
    # =========================
    def select_best_path(self, costs):
        """
        采用 Pareto 候选集的全局路径选择：
        - 目标：J_len, J_risk, J_unc 全部越小越好
        - 第一步：用支配关系筛选 Pareto 前沿（去掉明显更差的路径）
        - 第二步：在 Pareto 集合里按“风险优先、长度其次、不确定性兜底”的策略选一条
        :param costs: List[(J_len, J_risk, J_unc, idx)]
        :return: best_idx (int) or None
        """

        logger = self.get_logger()

        if not costs:
            logger.warn('[global_path_assessor] No valid candidate path, cannot select best.')
            return None

        eps = 1e-6  # 用于浮点比较的小阈值

        def dominates(a, b):
            """
            判断 a 是否支配 b
            a, b: (J_len, J_risk, J_unc)
            条件：
              1) a 在所有维度上都不比 b 差
              2) 且至少一个维度严格更好
            """
            L1, R1, U1 = a
            L2, R2, U2 = b

            not_worse = (L1 <= L2 + eps) and (R1 <= R2 + eps) and (U1 <= U2 + eps)
            strictly_better = (L1 < L2 - eps) or (R1 < R2 - eps) or (U1 < U2 - eps)
            return not_worse and strictly_better

        # =============================
        # 1) 计算 Pareto 前沿
        # =============================
        pareto_flags = [True] * len(costs)

        for i, (L1, R1, U1, idx1) in enumerate(costs):
            if not pareto_flags[i]:
                continue
            for j, (L2, R2, U2, idx2) in enumerate(costs):
                if i == j:
                    continue
                if dominates((L2, R2, U2), (L1, R1, U1)):
                    # path j 在三个指标上都不比 i 差，且至少一个更好
                    # → i 被支配，不能属于 Pareto 集合
                    pareto_flags[i] = False
                    break

        pareto_costs = [c for c, flag in zip(costs, pareto_flags) if flag]

        # 打印调试信息，看看 Pareto 集合里有哪些路径
        logger.info('[global_path_assessor] Pareto candidates:')
        for (L, R, U, idx) in pareto_costs:
            logger.info(f'  - path #{idx}: len={L:.2f}, risk={R:.3f}, unc={U:.3f}')

        # 如果所有路径都被淘汰（理论上不太可能），fallback 到原始 costs
        if not pareto_costs:
            logger.warn('[global_path_assessor] Pareto set empty, fallback to all candidates.')
            pareto_costs = costs

        # =============================
        # 2) 在 Pareto 集合内部选最终路径
        #    策略：先风险最小 → 再长度最短 → 再不确定性最小
        # =============================

        # 2.1 先选风险最小的那些
        min_risk = min(c[1] for c in pareto_costs)  # c[1] = J_risk
        # 给一个小 epsilon，避免浮点误差导致奇怪效果
        eps_risk = 1e-3
        risk_best = [c for c in pareto_costs if c[1] <= min_risk + eps_risk]

        # 2.2 在这些“同等级风险”的路径中选长度最短的那些
        min_len = min(c[0] for c in risk_best)  # c[0] = J_len
        eps_len = 1e-3
        len_best = [c for c in risk_best if c[0] <= min_len + eps_len]

        # 2.3 如果还有多条，就选不确定性 J_unc 最小的那一条
        #     （这里直接用 min 即可）
        best_L, best_R, best_U, best_idx = min(len_best, key=lambda c: c[2])  # c[2] = J_unc

        logger.info(
            f'[global_path_assessor] Selected path #{best_idx} '
            f'(len={best_L:.2f}, risk={best_R:.3f}, unc={best_U:.3f}) '
            f'from {len(costs)} candidates (Pareto size = {len(pareto_costs)}).'
        )

        return best_idx


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPathAssessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
