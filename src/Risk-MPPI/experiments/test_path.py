#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class SimplePathPublisher(Node):
    def __init__(self):
        super().__init__('simple_path_publisher')
        self.publisher = self.create_publisher(Path, '/plan', 10)
        self.timer = self.create_timer(3.0, self.publish_simple_path)
        self.get_logger().info('路径发布器启动')
        
    def publish_simple_path(self):
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = self.get_clock().now().to_msg()
        
        # 创建简单的前进路径
        for i in range(10):
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(i * 0.2)  # 每步0.2米
            pose.pose.position.y = 0.0
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        
        self.publisher.publish(path)
        self.get_logger().info(f'发布路径，包含{len(path.poses)}个点')

def main():
    rclpy.init()
    node = SimplePathPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
