#!/usr/bin/env python3
"""
@brief: log-MPPI控制器ROS2节点，适配TurtleBot3
优先跟踪 Nav2 的 /plan（绿色线），没有 /plan 时用 RViz 点生成的直线插点路径。
MPPI 若返回无效控制，自动切备用控制器；外层提供速度/转向增益与最小前进速度。
"""

import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from std_msgs.msg import Bool

from mppi_controller import TurtleBot3MPPI


class LogMPPIControllerNode(Node):
    def __init__(self):
        super().__init__('log_mppi_controller')

        # —— 参数 ——（你原有的 + 我新增的外层控制参数）
        self.declare_parameter('control_frequency', 20.0)
        self.declare_parameter('max_linear_velocity', 1.0)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('robot_radius', 0.22)

        self.declare_parameter('num_samples', 800)
        self.declare_parameter('time_horizon', 1.0)
        self.declare_parameter('exploration_variance', [0.1, 0.2])
        self.declare_parameter('lambda_temperature', 1.0)
        self.declare_parameter('num_iterations', 1)
        self.declare_parameter('use_log_normal', True)
        self.declare_parameter('log_normal_mu', 0.0)
        self.declare_parameter('log_normal_std', 0.3)

        self.declare_parameter('collision_cost', 1000.0)
        self.declare_parameter('state_weights', [1.0, 1.0, 0.5])
        self.declare_parameter('goal_tolerance', 0.2)

        # 外层速度策略参数（可在命令行改）
        self.declare_parameter('speed_gain', 1.8)             # 线速度增益（作用在MPPI输出外层）
        self.declare_parameter('turn_gain',  1.2)             # 角速度增益（外层）
        self.declare_parameter('min_forward_speed', 0.25)     # 朝向误差小于阈值时的最小线速
        self.declare_parameter('heading_ok_threshold', 0.6)   # 朝向误差阈值（弧度）
        self.declare_parameter('lookahead_distance', 1.0)     # 前瞻距离（m）

        # —— 运行时状态 —— 
        self.current_pose = None          # 保存 (位姿Pose)；用于算 yaw
        self.current_velocity = None      # 保存 (线/角速度)
        self.global_path: Path = None     # /plan 或 自建直线插点路径
        self.target_reached = False
        self.has_odom = False             # 是否已收到 /odom

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # —— 订阅 —— 
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        self.create_subscription(Path, '/plan', self.path_callback, qos)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self.costmap_callback, qos)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, qos)

        # —— 发布 —— 
        self.cmd_vel_pub   = self.create_publisher(Twist, '/cmd_vel', qos)
        self.status_pub    = self.create_publisher(Bool, '/mppi_status', qos)
        self.local_plan_pub = self.create_publisher(Path, '/local_plan', qos)

        # —— 初始化 MPPI 本体 —— 
        self.initialize_mppi()

        # —— 控制定时器 —— 
        control_frequency = float(self.get_parameter('control_frequency').value)
        self.timer = self.create_timer(1.0 / control_frequency, self.control_loop)

        self.get_logger().info('log-MPPI Controller Node 启动成功')
        self.get_logger().info('优先使用 /plan；无 /plan 时，用 RViz 目标自动生成直线路径')

    # ---------------------------- 初始化 MPPI ----------------------------
    def initialize_mppi(self):
        robot_params = {
            'max_linear_velocity': float(self.get_parameter('max_linear_velocity').value),
            'max_angular_velocity': float(self.get_parameter('max_angular_velocity').value),
            'robot_radius': float(self.get_parameter('robot_radius').value),
            'collision_cost': float(self.get_parameter('collision_cost').value)
        }

        if bool(self.get_parameter('use_log_normal').value):
            dist_type = 1
            self.get_logger().info('使用 log-MPPI (Normal + LogNormal)')
        else:
            dist_type = 0
            self.get_logger().info('使用标准 MPPI (Normal)')

        LogN_info = [
            dist_type,
            float(self.get_parameter('log_normal_mu').value),
            float(self.get_parameter('log_normal_std').value)
        ]

        self.mppi = TurtleBot3MPPI(
            state_dim=3,
            control_dim=2,
            num_samples=int(self.get_parameter('num_samples').value),
            time_horizon=float(self.get_parameter('time_horizon').value),
            control_freq=float(self.get_parameter('control_frequency').value),
            exploration_variance=np.array(self.get_parameter('exploration_variance').value, dtype=np.float32),
            robot_params=robot_params,
            LogN_info=LogN_info,
            num_optimization_iterations=int(self.get_parameter('num_iterations').value),
            lambda_=float(self.get_parameter('lambda_temperature').value)
        )

    # ---------------------------- 回调：里程计 ----------------------------
    def odom_callback(self, msg: Odometry):
        # 位姿
        self.current_pose = msg.pose.pose
        # 速度：拆成线 & 角，后续好用
        self.current_velocity = (msg.twist.twist.linear.x,
                                 msg.twist.twist.angular.z)
        self.has_odom = True

    # ---------------------------- 回调：全局路径 /plan ----------------------------
    def path_callback(self, msg: Path):
        if len(msg.poses) == 0:
            return
        self.global_path = msg
        self.target_reached = False
        self.get_logger().info(f'收到 /plan，{len(msg.poses)} 点')

    # ---------------------------- 回调：局部代价图 ----------------------------
    def costmap_callback(self, msg: OccupancyGrid):
        try:
            costmap_data = np.array(msg.data, dtype=np.float32)
            map_info = {
                'width': msg.info.width,
                'height': msg.info.height,
                'resolution': msg.info.resolution,
                'origin_x': msg.info.origin.position.x,
                'origin_y': msg.info.origin.position.y
            }
            # 传给 MPPI（你的 mppi_controller 里应有 update_costmap）
            self.mppi.update_costmap(costmap_data, map_info)
        except Exception as e:
            self.get_logger().error(f'处理代价地图失败: {e}')

    # ---------------------------- 回调：RViz 目标（无 /plan 时自建直线插点路径） ----------------------------
    def goal_callback(self, msg: PoseStamped):
        if not self.has_odom or self.current_pose is None:
            self.get_logger().warn('尚未收到 /odom，忽略 RViz 目标')
            return

        # 用 odom 坐标系
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = self.get_clock().now().to_msg()

        # 起点
        start_pose = PoseStamped()
        start_pose.header = path.header
        start_pose.pose = self.current_pose
        path.poses.append(start_pose)

        # 直线插点
        sx = float(self.current_pose.position.x)
        sy = float(self.current_pose.position.y)
        gx = float(msg.pose.position.x)
        gy = float(msg.pose.position.y)

        dist = float(np.hypot(gx - sx, gy - sy))
        # 0.3m一个点
        num_middle = max(1, int(dist / 0.3))
        for i in range(1, num_middle):
            r = float(i) / float(num_middle)
            ip = PoseStamped()
            ip.header = path.header
            ip.pose.position.x = float(sx + r * (gx - sx))
            ip.pose.position.y = float(sy + r * (gy - sy))
            ip.pose.position.z = 0.0
            ip.pose.orientation.w = 1.0
            path.poses.append(ip)

        # 终点
        goal = PoseStamped()
        goal.header = path.header
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        path.poses.append(goal)

        self.global_path = path
        self.target_reached = False
        self.get_logger().info(f'收到 RViz 目标: ({gx:.2f}, {gy:.2f})，生成直线插点路径 {len(path.poses)} 点')

    # ---------------------------- 工具：读当前状态(x,y,yaw) ----------------------------
    def get_current_state(self):
        if self.current_pose is None:
            return None
        q = self.current_pose.orientation
        # 四元数 -> yaw（ROS解算）
        siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return np.array([
            float(self.current_pose.position.x),
            float(self.current_pose.position.y),
            float(yaw)
        ], dtype=np.float32)

    # ---------------------------- 工具：从路径取“前瞻目标点” ----------------------------
    def get_target_state(self, current_state):
        if self.global_path is None or len(self.global_path.poses) == 0:
            return None

        lookahead = float(self.get_parameter('lookahead_distance').value)
        cx, cy = current_state[0], current_state[1]

        # 找最近点
        closest_idx, min_d = 0, float('inf')
        for i, ps in enumerate(self.global_path.poses):
            dx = ps.pose.position.x - cx
            dy = ps.pose.position.y - cy
            d = dx*dx + dy*dy
            if d < min_d:
                min_d = d
                closest_idx = i

        # 从最近点往前找 >= lookahead 的点
        target_pose = None
        for i in range(closest_idx, len(self.global_path.poses)):
            dx = self.global_path.poses[i].pose.position.x - cx
            dy = self.global_path.poses[i].pose.position.y - cy
            if math.hypot(dx, dy) >= lookahead:
                target_pose = self.global_path.poses[i]
                break

        # 没找到就用终点
        if target_pose is None:
            target_pose = self.global_path.poses[-1]

        # 目标 yaw（若无意义，用当前 yaw）
        tq = target_pose.pose.orientation
        siny_cosp = 2.0 * (tq.w*tq.z + tq.x*tq.y)
        cosy_cosp = 1.0 - 2.0 * (tq.y*tq.y + tq.z*tq.z)
        tyaw = math.atan2(siny_cosp, cosy_cosp)

        return np.array([
            float(target_pose.pose.position.x),
            float(target_pose.pose.position.y),
            float(tyaw)
        ], dtype=np.float32)

    # ---------------------------- 备用控制（MPPI失败兜底） ----------------------------
    def simple_backup_control(self, current_state, target_state):
        dx = target_state[0] - current_state[0]
        dy = target_state[1] - current_state[1]
        angle_to_target = math.atan2(dy, dx)
        angle_err = angle_to_target - current_state[2]
        # 归一化
        angle_err = (angle_err + math.pi) % (2*math.pi) - math.pi
        dist = math.hypot(dx, dy)

        # 规则：角度误差大 -> 先转向；小 -> 前进微调
        if abs(angle_err) > 0.5:
            v = 0.0
            w = 0.8 * np.sign(angle_err)
        else:
            v = min(0.3, 0.5 * dist)
            w = 1.0 * angle_err

        v = float(np.clip(v, 0.0, self.get_parameter('max_linear_velocity').value))
        w = float(np.clip(w, -self.get_parameter('max_angular_velocity').value,
                               self.get_parameter('max_angular_velocity').value))
        return np.array([v, w], dtype=np.float32)

    # ---------------------------- 发布 local_plan 供 RViz 看 ----------------------------
    def publish_local_plan(self, current_state, target_state):
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = self.get_clock().now().to_msg()

        a = PoseStamped()
        a.header = path.header
        a.pose.position.x = float(current_state[0])
        a.pose.position.y = float(current_state[1])
        a.pose.orientation.w = 1.0

        b = PoseStamped()
        b.header = path.header
        b.pose.position.x = float(target_state[0])
        b.pose.position.y = float(target_state[1])
        b.pose.orientation.w = 1.0

        path.poses = [a, b]
        self.local_plan_pub.publish(path)

    # ---------------------------- 主循环 ----------------------------
    def control_loop(self):
        # 必须先有 /odom
        if not self.has_odom:
            return

        current_state = self.get_current_state()
        if current_state is None:
            self.publish_zero_velocity()
            return

        target_state = self.get_target_state(current_state)
        if target_state is None:
            # 没有路径就停住（等 /plan 或 RViz 目标）
            self.publish_zero_velocity()
            return

        # 是否到达最终目标？
        goal_tol = float(self.get_parameter('goal_tolerance').value)
        if self.global_path and len(self.global_path.poses) > 0:
            goal = self.global_path.poses[-1].pose.position
            if math.hypot(current_state[0] - goal.x, current_state[1] - goal.y) < goal_tol:
                if not self.target_reached:
                    self.get_logger().info('到达目标')
                    self.target_reached = True
                    s = Bool(); s.data = True
                    self.status_pub.publish(s)
                self.publish_zero_velocity()
                return

        # MPPI 计算
        weights = np.array(self.get_parameter('state_weights').value, dtype=np.float32)
        try:
            t0 = time.time()
            control, normalizer, min_cost = self.mppi.compute_control(current_state, target_state, weights)
            dt = time.time() - t0

            # MPPI 失败条件：NaN / 明显无效
            invalid = (control is None
                       or np.any(np.isnan(control))
                       or (control[0] == 0.0 and control[1] == 0.0 and (min_cost == float('inf') or normalizer <= 0)))
            if invalid:
                self.get_logger().warn('MPPI返回无效控制，使用备用控制器')
                control = self.simple_backup_control(current_state, target_state)

            # 发布控制 & 局部线段
            self.publish_control(control, current_state, target_state)
            self.publish_local_plan(current_state, target_state)

            # 性能日志（>50ms提示）
            if dt > 0.05:
                self.get_logger().warn(f'控制计算耗时: {dt*1000:.1f}ms')

        except Exception as e:
            self.get_logger().error(f'MPPI控制异常: {e}，使用备用控制器')
            control = self.simple_backup_control(current_state, target_state)
            self.publish_control(control, current_state, target_state)

    # ---------------------------- 发布速度（带外层增益&最小前进） ----------------------------
    def publish_control(self, control, current_state, target_state):
        v = float(control[0]) * float(self.get_parameter('speed_gain').value)
        w = float(control[1]) * float(self.get_parameter('turn_gain').value)

        # 禁止倒车
        if v < 0.0:
            v = 0.0

        # “角度误差不大”时给最小前进速度
        dx = target_state[0] - current_state[0]
        dy = target_state[1] - current_state[1]
        angle_to_target = math.atan2(dy, dx)
        ang_err = (angle_to_target - current_state[2] + math.pi) % (2*math.pi) - math.pi
        if abs(ang_err) < float(self.get_parameter('heading_ok_threshold').value):
            v = max(v, float(self.get_parameter('min_forward_speed').value))

        v = float(np.clip(v, 0.0, self.get_parameter('max_linear_velocity').value))
        w = float(np.clip(w, -self.get_parameter('max_angular_velocity').value,
                              self.get_parameter('max_angular_velocity').value))

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_vel_pub.publish(cmd)

    def publish_zero_velocity(self):
        self.cmd_vel_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    try:
        node = LogMPPIControllerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'节点错误: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
