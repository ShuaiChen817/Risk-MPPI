#!/usr/bin/env python3
"""
Minimal test controller - just moves forward when path is received
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
import time

class MinimalTestController(Node):
    def __init__(self):
        super().__init__('minimal_test_controller')
        
        # State
        self.has_path = False
        self.start_time = None
        
        # Subscribers
        self.path_sub = self.create_subscription(
            Path, '/plan', self.path_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timer
        self.timer = self.create_timer(0.1, self.control_loop)  # 10Hz
        
        self.get_logger().info('Minimal Test Controller started')
        self.get_logger().info('Waiting for path on /plan topic...')

    def path_callback(self, msg):
        """Path callback"""
        if len(msg.poses) > 0 and not self.has_path:
            self.has_path = True
            self.start_time = time.time()
            self.get_logger().info(f'Received path with {len(msg.poses)} points - starting to move!')

    def control_loop(self):
        """Simple control loop"""
        cmd_vel = Twist()
        
        if self.has_path and self.start_time is not None:
            elapsed = time.time() - self.start_time
            
            if elapsed < 5.0:  # Move forward for 5 seconds
                cmd_vel.linear.x = 0.2  # 0.2 m/s forward
                cmd_vel.angular.z = 0.0
                self.get_logger().info(f'Moving forward... ({elapsed:.1f}s)')
            else:
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
                if elapsed < 6.0:  # Only log once
                    self.get_logger().info('Movement test completed - stopping')
        else:
            # No path received yet
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
        
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = MinimalTestController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()