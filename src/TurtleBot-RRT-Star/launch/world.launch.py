from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import math

def generate_launch_description():
    #use_sim_time_env = SetEnvironmentVariable('RCL_ROS_TIME_SOURCE', 'SYSTEM')
   
   
    #加载 Gazebo 世界模型
    world_path = os.path.join(
        os.getenv('HOME'),
        'turtlebot_ws', 'src', 'TurtleBot-RRT-Star', 'world', 'three_walking_humans.world'
    )
    #启动 Gazebo和世界
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch', 'gazebo.launch.py'
            )
        ),
        launch_arguments={'world': world_path,
            'pause': 'false',  # 确保仿真立即开始
            'use_sim_time': 'true'
        }.items()
    )


     # === 用你自己的 SDF ===
    # tb3_sdf = os.path.join(
    #     os.getenv('HOME'),
    #     'turtlebot_ws', 'src', 'TurtleBot-RRT-Star', 'robot', 'model.sdf'
    # )

    # spawn_tb3 = Node(
    #     package='gazebo_ros',
    #     executable='spawn_entity.py',
    #     arguments=[
    #         '-entity', 'tb3',
    #         '-file', tb3_sdf,
    #         '-x', '0', '-y', '0', '-z', '0.01',
    #         '-R', '0', '-P', '0', '-Y', f'{math.pi}'
    #     ],
    #     output='screen'
    # )

    # # === 用你自己的 URDF 给 RViz/TF ===
    # urdf_file = os.path.join(
    #     os.getenv('HOME'),
    #     'turtlebot_ws', 'src', 'TurtleBot-RRT-Star', 'robot', 'turtlebot3_waffle.urdf'
    # )

    # robot_state_publisher = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     name='robot_state_publisher',
    #     output='screen',
    #     parameters=[{
    #         'use_sim_time': True,
    #         'robot_description': open(urdf_file, 'r').read()
    #     }]
    # )

    # return LaunchDescription([gazebo, spawn_tb3, robot_state_publisher])

#     #加载tb3小车模型
    spawn_tb3 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'tb3',
            '-file', os.path.join(
                get_package_share_directory('turtlebot3_gazebo'),
                'models', 'turtlebot3_waffle_pi', 'model.sdf'
            ),
            '-x', '0', '-y', '0', '-z', '0.01',
            '-R', '0', '-P', '0', '-Y', f'{math.pi}'   # 180°
        ],
        output='screen'
    )

    

    


    # 加载 tb3 的 URDF 文件 Rviz里的TF
    urdf_file = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'urdf', 'turtlebot3_waffle_pi.urdf')
    # rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
    #            parameters=[{'use_sim_time': True, 'robot_description': open(urdf_file).read()}])

# ============= 发布 TF (非常关键！) =============
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': open(urdf_file, 'r').read()
        }]
    )
    return LaunchDescription([gazebo, spawn_tb3, robot_state_publisher])
