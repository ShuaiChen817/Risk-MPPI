#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

print("开始导入...")

class LogMPPIControllerNode(Node):
    def __init__(self):
        super().__init__('log_mppi_controller')
        print("Node创建成功")
        
        # 测试参数声明
        print("开始声明参数...")
        self.declare_parameter('control_frequency', 20.0)
        print("第一个参数声明成功")
        
        self.declare_parameter('max_linear_velocity', 0.3)
        print("第二个参数声明成功")
        
        # 只声明几个参数测试
        self.get_logger().info('调试版本启动成功')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        print("创建节点...")
        node = LogMPPIControllerNode()
        print("节点创建完成")
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