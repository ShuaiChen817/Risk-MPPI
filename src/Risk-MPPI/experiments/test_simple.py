#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class SimpleTestNode(Node):
    def __init__(self):
        super().__init__('simple_test')
        
        # 简单声明参数
        self.declare_parameter('test_param', 1.0)
        
        # 发布器
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 定时器
        self.timer = self.create_timer(1.0, self.test_callback)
        
        self.get_logger().info('Simple test node started')

    def test_callback(self):
        self.get_logger().info('Test callback running')

def main():
    rclpy.init()
    node = SimpleTestNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()