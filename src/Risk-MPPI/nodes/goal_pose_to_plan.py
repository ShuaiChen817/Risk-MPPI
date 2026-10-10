#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav2_msgs.action import ComputePathToPose

class GoalToPlan(Node):
    def __init__(self):
        super().__init__('goal_to_plan')

        # 可配参数
        self.goal_topic = self.declare_parameter('goal_topic', '/goal_pose') \
                              .get_parameter_value().string_value or '/goal_pose'
        self.global_frame = self.declare_parameter('global_frame', 'odom') \
                                .get_parameter_value().string_value or 'odom'
        self.replan_rate = self.declare_parameter('replan_rate', 1.0) \
                                .get_parameter_value().double_value or 1.0
        self.clear_on_new_goal = self.declare_parameter('clear_on_new_goal', True) \
                                      .get_parameter_value().bool_value

        # 订阅 RViz 目标
        self.sub = self.create_subscription(PoseStamped, self.goal_topic, self._goal_cb, 10)

        # 与 Nav2 规划器通讯
        self.cli = ActionClient(self, ComputePathToPose, '/compute_path_to_pose')

        # 发布 /plan（用来“清线”或将来需要自定义路径时复用）
        self.plan_pub = self.create_publisher(Path, '/plan_raw', 10)

        # 最近的目标
        self.last_goal: PoseStamped | None = None

        # 周期性重算路径
        if self.replan_rate > 0.0:
            self.timer = self.create_timer(1.0 / self.replan_rate, self._replan_tick)
        else:
            self.timer = None

        self.get_logger().info(
            f'Listening goals on {self.goal_topic}, global_frame={self.global_frame}, '
            f'replan_rate={self.replan_rate} Hz'
        )

    # 收到 RViz 的 2D Goal Pose
    def _goal_cb(self, msg: PoseStamped):
        # 统一目标坐标系
        if not msg.header.frame_id:
            msg.header.frame_id = self.global_frame
        elif msg.header.frame_id != self.global_frame:
            # 这里不做 TF 变换，直接改帧名（你的 Nav2 也跑在同一帧 odom）
            self.get_logger().warn(
                f'Goal frame "{msg.header.frame_id}" != global_frame "{self.global_frame}", '
                f'will force to {self.global_frame}')
            msg.header.frame_id = self.global_frame

        self.last_goal = msg

        # 先清掉旧绿线（发一条空 Path）
        if self.clear_on_new_goal:
            empty = Path()
            empty.header.frame_id = self.global_frame
            empty.header.stamp = self.get_clock().now().to_msg()
            self.plan_pub.publish(empty)

        self.get_logger().info(
            f'Received goal: frame={msg.header.frame_id}, '
            f'pos=({msg.pose.position.x:.2f},{msg.pose.position.y:.2f}), '
            f'cleared old /plan, computing path…'
        )
        self._call_planner_once()

    # 定时重算路径（让绿线从当前处开始、旧段自然消失）
    def _replan_tick(self):
        if self.last_goal is None:
            return
        self._call_planner_once()

    # 调一次 /compute_path_to_pose
    def _call_planner_once(self):
        if self.last_goal is None:
            return
        if not self.cli.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn('planner_server action not available yet')
            return

        goal_msg = ComputePathToPose.Goal()
        # 目标用最新 goal（时间戳刷新一下更稳妥）
        goal = PoseStamped()
        goal.header = self.last_goal.header
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose = self.last_goal.pose
        goal_msg.goal = goal
        goal_msg.use_start = False  # 让 planner 自己用机器人当前位置当起点

        future = self.cli.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Planner goal rejected')
            return
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        result = future.result().result
        n = len(result.path.poses)
        if n == 0:
            self.get_logger().warn('Planner returned EMPTY path')
        else:
            self.get_logger().info(f'Planner returned path with {n} poses')
            self.plan_pub.publish(result.path)

def main():
    rclpy.init()
    rclpy.spin(GoalToPlan())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
