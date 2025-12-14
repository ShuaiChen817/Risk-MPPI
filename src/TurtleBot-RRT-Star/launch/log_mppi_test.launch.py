import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 获取包路径
    pkg_share = get_package_share_directory('nav2_bringup')
    
    # 启动您现有的世界
    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(os.getenv('HOME'), 'turtlebot_ws/src/TurtleBot-RRT-Star/launch/world.launch.py')
        ])
    )
    
    # 启动Nav2 (不包括controller_server)
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_share, 'launch', 'navigation_launch.py')
        ]),
        launch_arguments={
            'params_file': os.path.join(os.getenv('HOME'), 'turtlebot_ws/src/TurtleBot-RRT-Star/nav2_params.yaml'),
            'use_sim_time': 'true',
            'autostart': 'true'
        }.items()
    )
    
    # log-MPPI控制器节点
    # log_mppi_node = Node(
    #     package='TurtleBot-RRT-Star',
    #     executable='log_mppi_node.py',
    #     name='log_mppi_controller',
    #     output='screen',
    #     parameters=[{
    #         'use_sim_time': True,
    #         'control_frequency': 20.0,
    #         'max_linear_velocity': 0.3,
    #         'max_angular_velocity': 1.0,
    #         'robot_radius': 0.22,
    #         'num_samples': 800,
    #         'time_horizon': 1.0,
    #         'exploration_variance': [0.1, 0.2],
    #         'lambda_temperature': 1.0,
    #         'num_iterations': 1,
    #         'use_log_normal': True,  # 设为False使用标准MPPI
    #         'log_normal_mu': 0.0,
    #         'log_normal_std': 0.3,
    #         'collision_cost': 1000.0,
    #         'state_weights': [1.0, 1.0, 0.5],
    #         'goal_tolerance': 0.2
    #     }]
    # )
    
    log_mppi_node = ExecuteProcess(
    cmd=[
        'python3', 
        os.path.join(os.getenv('HOME'), 'turtlebot_ws/src/TurtleBot-RRT-Star/scripts/log_mppi_node.py')
    ],
    output='screen',
    # 注意：ExecuteProcess不支持parameters参数
    # 需要在Python代码中设置默认参数值
)
    return LaunchDescription([
        world_launch,
        nav2_launch,
        log_mppi_node
    ])