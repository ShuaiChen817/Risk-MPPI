#!/usr/bin/env python3
"""
@brief: log-MPPI控制器ROS2节点，适配TurtleBot3
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import time

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nodes"))
from mppi_controller import TurtleBot3MPPI

class LogMPPIControllerNode(Node):
    def __init__(self):
        super().__init__('log_mppi_controller')
        
        # 声明参数
        self.declare_parameters()
        
        # 状态变量
        self.current_pose = None
        self.current_velocity = None
        self.global_path = None
        self.local_costmap = None
        self.target_reached = False
        
        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 订阅器
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos)
        self.path_sub = self.create_subscription(
            Path, '/plan', self.path_callback, qos)
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/local_costmap/costmap', self.costmap_callback, qos)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, qos)
        
        # 发布器
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos)
        self.status_pub = self.create_publisher(Bool, '/mppi_status', qos)
        
        # 初始化MPPI控制器
        self.initialize_mppi()
        
        # 控制定时器
        control_frequency = self.get_parameter('control_frequency').value
        self.timer = self.create_timer(1.0/control_frequency, self.control_loop)
        
        self.get_logger().info('log-MPPI Controller Node启动成功')

    def declare_parameters(self):
        """声明ROS参数"""
        self.declare_parameter('control_frequency', 20.0)
        self.declare_parameter('max_linear_velocity', 0.3)
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

    def initialize_mppi(self):
        """初始化MPPI控制器"""
        # 机器人参数
        robot_params = {
            'max_linear_velocity': self.get_parameter('max_linear_velocity').value,
            'max_angular_velocity': self.get_parameter('max_angular_velocity').value,
            'robot_radius': self.get_parameter('robot_radius').value,
            'collision_cost': self.get_parameter('collision_cost').value
        }
        
        # log-MPPI分布参数
        if self.get_parameter('use_log_normal').value:
            dist_type = 1  # 使用log-MPPI
            self.get_logger().info('使用log-MPPI (Normal + LogNormal分布)')
        else:
            dist_type = 0  # 使用标准MPPI
            self.get_logger().info('使用标准MPPI (Normal分布)')
        
        LogN_info = [
            dist_type,
            self.get_parameter('log_normal_mu').value,
            self.get_parameter('log_normal_std').value
        ]
        
        # 创建MPPI控制器
        self.mppi = TurtleBot3MPPI(
            state_dim=3,
            control_dim=2,
            num_samples=self.get_parameter('num_samples').value,
            time_horizon=self.get_parameter('time_horizon').value,
            control_freq=self.get_parameter('control_frequency').value,
            exploration_variance=self.get_parameter('exploration_variance').value,
            robot_params=robot_params,
            LogN_info=LogN_info,
            num_optimization_iterations=self.get_parameter('num_iterations').value,
            lambda_=self.get_parameter('lambda_temperature').value
        )

    def odom_callback(self, msg):
        """里程计回调"""
        self.current_pose = msg.pose.pose
        self.current_velocity = msg.twist.twist

    def path_callback(self, msg):
        """路径回调"""
        if len(msg.poses) > 0:
            self.global_path = msg
            self.target_reached = False
            self.get_logger().info(f'收到新路径，包含{len(msg.poses)}个点')

    def costmap_callback(self, msg):
        """代价地图回调"""
        try:
            # 处理代价地图数据
            width, height = msg.info.width, msg.info.height
            resolution = msg.info.resolution
            origin_x = msg.info.origin.position.x
            origin_y = msg.info.origin.position.y
            
            # 转换为numpy数组
            costmap_data = np.array(msg.data, dtype=np.float32)
            
            # 地图信息
            map_info = {
                'width': width,
                'height': height,
                'resolution': resolution,
                'origin_x': origin_x,
                'origin_y': origin_y
            }
            
            # 更新MPPI控制器的地图
            self.mppi.update_costmap(costmap_data, map_info)
            self.local_costmap = msg
            
        except Exception as e:
            self.get_logger().error(f'处理代价地图失败: {e}')

    def goal_callback(self, msg):
        """目标回调"""
        self.get_logger().info(f'收到新目标: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')

    def get_current_state(self):
        """获取当前状态"""
        if self.current_pose is None:
            return None
        
        # 从四元数转欧拉角
        orientation = self.current_pose.orientation
        import math
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return np.array([
            self.current_pose.position.x,
            self.current_pose.position.y,
            yaw
        ], dtype=np.float32)

    def get_target_state(self):
        """从路径获取目标状态"""
        if self.global_path is None or len(self.global_path.poses) == 0:
            return None
        
        current_state = self.get_current_state()
        if current_state is None:
            return None
        
        # 寻找前瞻目标点
        lookahead_dist = 1.0  # 1米前瞻距离
        current_x, current_y = current_state[0], current_state[1]
        
        # 找到最近的路径点
        min_dist = float('inf')
        closest_idx = 0
        for i, pose in enumerate(self.global_path.poses):
            dist = np.sqrt((pose.pose.position.x - current_x)**2 + 
                          (pose.pose.position.y - current_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # 从最近点开始寻找前瞻点
        target_pose = None
        for i in range(closest_idx, len(self.global_path.poses)):
            pose = self.global_path.poses[i]
            dist = np.sqrt((pose.pose.position.x - current_x)**2 + 
                          (pose.pose.position.y - current_y)**2)
            if dist >= lookahead_dist:
                target_pose = pose
                break
        
        # 如果没有找到前瞻点，使用终点
        if target_pose is None:
            target_pose = self.global_path.poses[-1]
        
        # 转换为状态
        orientation = target_pose.pose.orientation
        import math
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        target_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return np.array([
            target_pose.pose.position.x,
            target_pose.pose.position.y,
            target_yaw
        ], dtype=np.float32)

    def control_loop(self):
        """主控制循环"""
        current_state = self.get_current_state()
        target_state = self.get_target_state()
        
        if current_state is None or target_state is None:
            self.publish_zero_velocity()
            return
        
        # 检查是否到达目标
        goal_tolerance = self.get_parameter('goal_tolerance').value
        if self.global_path and len(self.global_path.poses) > 0:
            final_goal = self.global_path.poses[-1]
            final_x = final_goal.pose.position.x
            final_y = final_goal.pose.position.y
            
            dist_to_goal = np.sqrt((current_state[0] - final_x)**2 + 
                                   (current_state[1] - final_y)**2)
            
            if dist_to_goal < goal_tolerance:
                if not self.target_reached:
                    self.get_logger().info('到达目标!')
                    self.target_reached = True
                    status_msg = Bool()
                    status_msg.data = True
                    self.status_pub.publish(status_msg)
                self.publish_zero_velocity()
                return
        
        # 执行MPPI控制
        weights = np.array(self.get_parameter('state_weights').value, dtype=np.float32)
        
        try:
            start_time = time.time()
            control, normalizer, min_cost = self.mppi.compute_control(
                current_state, target_state, weights)
            computation_time = time.time() - start_time
            
            # 发布控制指令
            self.publish_control(control)
            
            # 性能监控
            if computation_time > 0.05:
                self.get_logger().warn(f'控制计算耗时: {computation_time*1000:.1f}ms')
            
        except Exception as e:
            self.get_logger().error(f'MPPI控制失败: {e}')
            self.publish_zero_velocity()

    def publish_control(self, control):
        """发布控制指令"""
        cmd_vel = Twist()
        cmd_vel.linear.x = float(control[0])
        cmd_vel.angular.z = float(control[1])
        
        # 安全限制
        max_linear = self.get_parameter('max_linear_velocity').value
        max_angular = self.get_parameter('max_angular_velocity').value
        
        cmd_vel.linear.x = np.clip(cmd_vel.linear.x, -max_linear, max_linear)
        cmd_vel.angular.z = np.clip(cmd_vel.angular.z, -max_angular, max_angular)
        
        self.cmd_vel_pub.publish(cmd_vel)

    def publish_zero_velocity(self):
        """发布零速度"""
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)

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