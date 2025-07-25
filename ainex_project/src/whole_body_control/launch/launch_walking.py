import launch
from launch.substitutions import LaunchConfiguration, Command
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_ros
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable


def generate_launch_description():
    # Declare the launch description
    ld = LaunchDescription()

    ainex_description_pkg_share = FindPackageShare('ainex_description').find('ainex_description')
    ainex_urdf_file_path = os.path.join(ainex_description_pkg_share, 'robots', 'ainex.urdf')

    with open(ainex_urdf_file_path, 'r') as urdf_temp:
        robot_description = urdf_temp.read()

    params = {'robot_description': robot_description}

    # Set environment variables
    ld.add_action(SetEnvironmentVariable('DISPLAY', ':0'))
    ld.add_action(SetEnvironmentVariable('QT_X11_NO_MITSHM', '1'))
    
    # Add the RViz2 node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
    )
    
    robot_state_publisher = launch_ros.actions.Node(package='robot_state_publisher',
                                  executable='robot_state_publisher',
                                  output='both',
                                  parameters=[params])
    
    
    walking_node = Node(
        package='whole_body_control',
        executable='walking',
        name='walking',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    # Add the nodes to the launch description
    ld.add_action(robot_state_publisher)
    ld.add_action(rviz_node)
    ld.add_action(walking_node)

    return ld