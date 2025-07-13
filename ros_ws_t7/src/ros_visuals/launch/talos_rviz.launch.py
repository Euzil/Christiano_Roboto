from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
    urdf_path = os.path.join(
    get_package_share_directory("talos_description"),
    "robots/talos_reduced_no_hands.urdf"
    )
    
    with open(urdf_path, 'r') as infp:
        robot_description_content = infp.read()

    your_package_path = get_package_share_directory('ros_visuals')
    rviz_config_file = os.path.join(your_package_path, 'config', 'talos.rviz')

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description_content}]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file]
        )
    ])