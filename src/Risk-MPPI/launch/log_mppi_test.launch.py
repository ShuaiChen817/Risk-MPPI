"""Alternative controller entry point; lifecycle transitions remain manual.

Do not run this together with nodes/test_mppi.py: both publish /cmd_vel.
The path bridge and risk-map nodes still need to be started separately.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    project_share = Path(get_package_share_directory('nav2_risk_mppi'))
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            str(project_share / 'launch' / 'world.launch.py'))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            str(project_share / 'launch' / 'manual_navigation.launch.py'))),
        ExecuteProcess(
            cmd=['python3', str(project_share / 'nodes' / 'log_mppi_node.py'),
                 '--ros-args', '-p', 'use_sim_time:=true'], output='screen'),
    ])
