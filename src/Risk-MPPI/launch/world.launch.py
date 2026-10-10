"""Launch the Gazebo Classic world and TurtleBot3 Waffle Pi."""
import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    project_share = Path(get_package_share_directory('nav2_risk_mppi'))
    gazebo_share = Path(get_package_share_directory('gazebo_ros'))
    robot_share = Path(get_package_share_directory('turtlebot3_gazebo'))
    urdf = robot_share / 'urdf' / 'turtlebot3_waffle_pi.urdf'
    return LaunchDescription([
        DeclareLaunchArgument(
            'world', default_value=str(project_share / 'world' / 'three_walking_humans.world'),
            description='Absolute path to a Gazebo Classic world file.'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(gazebo_share / 'launch' / 'gazebo.launch.py')),
            launch_arguments={
                'world': LaunchConfiguration('world'), 'pause': 'false',
                'use_sim_time': 'true',
            }.items()),
        Node(
            package='gazebo_ros', executable='spawn_entity.py', output='screen',
            arguments=[
                '-entity', 'tb3', '-file',
                str(robot_share / 'models' / 'turtlebot3_waffle_pi' / 'model.sdf'),
                '-x', '0', '-y', '0', '-z', '0.01', '-R', '0', '-P', '0',
                '-Y', str(math.pi),
            ]),
        Node(
            package='robot_state_publisher', executable='robot_state_publisher',
            output='screen', parameters=[{
                'use_sim_time': True, 'robot_description': urdf.read_text(),
            }]),
    ])
