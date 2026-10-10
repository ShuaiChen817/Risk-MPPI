#!/usr/bin/env python3
"""
Direct robot movement test - no path required
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class DirectMovementTest(Node):
    def __init__(self):
        super().__init__('direct_movement_test')
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timer
        self.timer = self.create_timer(0.1, self.move_robot)
        self.start_time = time.time()
        
        self.get_logger().info('Direct Movement Test started')
        self.get_logger().info('Robot will move forward for 5 seconds...')

    def move_robot(self):
        """Move robot forward for 5 seconds"""
        cmd_vel = Twist()
        
        elapsed = time.time() - self.start_time
        
        if elapsed < 5.0:
            # Move forward
            cmd_vel.linear.x = 0.3
            cmd_vel.angular.z = 0.0
            
            if int(elapsed) != getattr(self, 'last_logged_second', -1):
                self.get_logger().info(f'Moving forward... {elapsed:.1f}s')
                self.last_logged_second = int(elapsed)
                
        elif elapsed < 6.0:
            # Stop
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            
            if not getattr(self, 'stop_logged', False):
                self.get_logger().info('Test completed - robot should have moved')
                self.stop_logged = True
        else:
            # Keep stopped
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
        
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = DirectMovementTest()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()