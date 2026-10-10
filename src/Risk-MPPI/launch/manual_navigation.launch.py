"""Start planner and costmap services without a lifecycle manager.

The Python MPPI node owns /cmd_vel. Nav2's controller output is isolated
on /cmd_vel_nav; no FollowPath action is needed in this workflow.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    project_share = Path(get_package_share_directory('nav2_risk_mppi'))
    params = LaunchConfiguration('params_file')
    sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=str(project_share / 'nav2_params.yaml')),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(
            package='nav2_planner', executable='planner_server', name='planner_server',
            output='screen', parameters=[params, {'use_sim_time': sim_time}]),
        Node(
            package='nav2_controller', executable='controller_server', name='controller_server',
            output='screen', parameters=[params, {'use_sim_time': sim_time}],
            remappings=[('cmd_vel', 'cmd_vel_nav')]),
    ])
