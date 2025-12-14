#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

# 测试import
print("测试import...")
from mppi_controller import TurtleBot3MPPI
print("mppi_controller import成功")

class LogMPPIControllerNode(Node):
    def __init__(self):
        super().__init__('log_mppi_controller')
        
        # 声明所有参数（复制您原文件的参数声明）
        print("声明所有参数...")
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
        print("所有参数声明成功")
        
        self.get_logger().info('所有测试通过')

def main(args=None):
    rclpy.init(args=args)
    try:
        node = LogMPPIControllerNode()
        rclpy.spin(node)
    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()