#!/usr/bin/env python3
"""
Simple path follower without costmap dependency
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path, Odometry
import numpy as np
import math

class SimplePathFollower(Node):
    def __init__(self):
        super().__init__('simple_path_follower')
        
        # State
        self.current_pose = None
        self.path = None
        self.target_reached = False
        self.current_target_idx = 0
        
        # Parameters
        self.lookahead_distance = 0.5  # meters
        self.linear_speed = 0.2        # m/s
        self.angular_speed = 0.8       # rad/s
        self.goal_tolerance = 0.15     # meters
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.path_sub = self.create_subscription(
            Path, '/plan', self.path_callback, 10)
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timer
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Simple Path Follower started')
        self.get_logger().info('- No costmap required')
        self.get_logger().info('- Uses pure pursuit algorithm')

    def odom_callback(self, msg):
        """Odometry callback"""
        self.current_pose = msg.pose.pose

    def path_callback(self, msg):
        """Path callback"""
        if len(msg.poses) > 0:
            self.path = msg.poses
            self.target_reached = False
            self.current_target_idx = 0
            self.get_logger().info(f'Received path with {len(msg.poses)} waypoints')

    def get_current_position(self):
        """Get current robot position"""
        if self.current_pose is None:
            return None
        
        return {
            'x': self.current_pose.position.x,
            'y': self.current_pose.position.y,
            'theta': self.quaternion_to_yaw(self.current_pose.orientation)
        }

    def quaternion_to_yaw(self, q):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def find_target_point(self, current_pos):
        """Find target point using pure pursuit"""
        if not self.path:
            return None
        
        # Start from current target index
        for i in range(self.current_target_idx, len(self.path)):
            waypoint = self.path[i].pose.position
            distance = math.sqrt(
                (waypoint.x - current_pos['x'])**2 + 
                (waypoint.y - current_pos['y'])**2
            )
            
            if distance >= self.lookahead_distance:
                self.current_target_idx = i
                return {'x': waypoint.x, 'y': waypoint.y}
        
        # If no point found, return the last waypoint
        if len(self.path) > 0:
            last_waypoint = self.path[-1].pose.position
            return {'x': last_waypoint.x, 'y': last_waypoint.y}
        
        return None

    def compute_control(self, current_pos, target_point):
        """Compute control commands using pure pursuit"""
        # Calculate distance and angle to target
        dx = target_point['x'] - current_pos['x']
        dy = target_point['y'] - current_pos['y']
        
        distance_to_target = math.sqrt(dx**2 + dy**2)
        angle_to_target = math.atan2(dy, dx)
        
        # Calculate steering angle
        angle_error = angle_to_target - current_pos['theta']
        
        # Normalize angle error
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi
        
        # Control logic
        cmd_vel = Twist()
        
        if abs(angle_error) > 0.3:  # Large angle error - rotate in place
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = self.angular_speed * (1.0 if angle_error > 0 else -1.0)
        else:  # Small angle error - move forward with steering
            cmd_vel.linear.x = self.linear_speed
            cmd_vel.angular.z = 2.0 * angle_error  # Proportional steering
        
        # Limit angular velocity
        cmd_vel.angular.z = max(-self.angular_speed, 
                               min(self.angular_speed, cmd_vel.angular.z))
        
        return cmd_vel

    def check_goal_reached(self, current_pos):
        """Check if final goal is reached"""
        if not self.path:
            return False
        
        final_goal = self.path[-1].pose.position
        distance_to_goal = math.sqrt(
            (final_goal.x - current_pos['x'])**2 + 
            (final_goal.y - current_pos['y'])**2
        )
        
        return distance_to_goal < self.goal_tolerance

    def control_loop(self):
        """Main control loop"""
        current_pos = self.get_current_position()
        
        if current_pos is None:
            self.get_logger().warn('No odometry data - stopping')
            self.publish_stop()
            return
        
        if not self.path:
            # No path - just stop
            self.publish_stop()
            return
        
        # Check if goal reached
        if self.check_goal_reached(current_pos):
            if not self.target_reached:
                self.get_logger().info('Goal reached!')
                self.target_reached = True
            self.publish_stop()
            return
        
        # Find target point
        target_point = self.find_target_point(current_pos)
        if target_point is None:
            self.get_logger().warn('No target point found')
            self.publish_stop()
            return
        
        # Compute and publish control
        cmd_vel = self.compute_control(current_pos, target_point)
        self.cmd_vel_pub.publish(cmd_vel)
        
        # Debug info (reduce frequency)
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
        
        if self._debug_counter % 20 == 0:  # Every 2 seconds
            self.get_logger().info(
                f'Pos: ({current_pos["x"]:.2f}, {current_pos["y"]:.2f}) '
                f'Target: ({target_point["x"]:.2f}, {target_point["y"]:.2f}) '
                f'Cmd: v={cmd_vel.linear.x:.2f}, w={cmd_vel.angular.z:.2f}'
            )

    def publish_stop(self):
        """Publish stop command"""
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = SimplePathFollower()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()