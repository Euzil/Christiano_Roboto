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
    ld = LaunchDescription()

    ainex_description_pkg_share = FindPackageShare('ainex_description').find('ainex_description')
    ainex_urdf_file_path = os.path.join(ainex_description_pkg_share, 'robots', 'ainex.urdf')

    with open(ainex_urdf_file_path, 'r') as urdf_temp:
        robot_description = urdf_temp.read()

    params = {'robot_description': robot_description}

    ld.add_action(SetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE', '1'))
    ld.add_action(SetEnvironmentVariable('QT_X11_NO_MITSHM', '1'))

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
    )
    
    robot_state_publisher = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[params]
    )

    joint_state_publisher_gui = launch_ros.actions.Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui'
    )

    ld.add_action(robot_state_publisher)
    ld.add_action(rviz_node)
    ld.add_action(joint_state_publisher_gui)

    return ld