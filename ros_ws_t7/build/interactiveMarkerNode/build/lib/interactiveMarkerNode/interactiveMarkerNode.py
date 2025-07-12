import rclpy
from rclpy.node import Node
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from geometry_msgs.msg import PoseStamped, Pose
import tf2_ros

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import threading
import time


class InteractiveMarkerNode(Node):
    def __init__(self):
        super().__init__('interactive_marker_node')

        self.server = InteractiveMarkerServer(self, "right_hand_marker_server")
        self.publisher = self.create_publisher(PoseStamped, "/right_hand_target_pose", 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(1.0, self.init_marker)
        self.marker_initialized = False
        self.current_pose = None 
        self.interp_thread = None

    def init_marker(self):
        if self.marker_initialized:
            return

        try:
            trans = self.tf_buffer.lookup_transform('base_link', 'arm_right_7_link', rclpy.time.Time())
            self.create_interactive_marker(trans.transform)
            self.marker_initialized = True
            self.timer.cancel()
            self.get_logger().info("Interactive Marker initialized.")
        except Exception as e:
            self.get_logger().info(f"Waiting for TF transform: {e}")

    def create_interactive_marker(self, transform):
        marker = Marker()
        marker.type = Marker.CUBE
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 1.0
        marker.color.a = 1.0

        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.name = "right_hand_marker"
        int_marker.description = "Right Hand Target"

        int_marker.pose.position.x = float(transform.translation.x)
        int_marker.pose.position.y = float(transform.translation.y)
        int_marker.pose.position.z = float(transform.translation.z)
        int_marker.pose.orientation.x = float(transform.rotation.x)
        int_marker.pose.orientation.y = float(transform.rotation.y)
        int_marker.pose.orientation.z = float(transform.rotation.z)
        int_marker.pose.orientation.w = float(transform.rotation.w)

        self.current_pose = int_marker.pose  # 初始化当前姿态

        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(marker)
        int_marker.controls.append(control)

        self.add_6dof_controls(int_marker)

        self.server.insert(int_marker, feedback_callback=self.marker_feedback)
        self.server.applyChanges()

    def add_6dof_controls(self, int_marker):
        def make_control(name, orientation, interaction_mode):
            control = InteractiveMarkerControl()
            control.orientation.w = float(orientation[0])
            control.orientation.x = float(orientation[1])
            control.orientation.y = float(orientation[2])
            control.orientation.z = float(orientation[3])
            control.name = name
            control.interaction_mode = interaction_mode
            int_marker.controls.append(control)

        make_control("move_x", [1, 1, 0, 0], InteractiveMarkerControl.MOVE_AXIS)
        make_control("move_y", [1, 0, 1, 0], InteractiveMarkerControl.MOVE_AXIS)
        make_control("move_z", [1, 0, 0, 1], InteractiveMarkerControl.MOVE_AXIS)
        make_control("rotate_x", [1, 1, 0, 0], InteractiveMarkerControl.ROTATE_AXIS)
        make_control("rotate_y", [1, 0, 1, 0], InteractiveMarkerControl.ROTATE_AXIS)
        make_control("rotate_z", [1, 0, 0, 1], InteractiveMarkerControl.ROTATE_AXIS)

    def marker_feedback(self, feedback):
        if feedback.event_type in [feedback.MOUSE_UP, feedback.POSE_UPDATE]:
            target_pose = feedback.pose

            if self.interp_thread and self.interp_thread.is_alive():
                self.get_logger().info("Waiting for previous interpolation to finish...")
                return

            try:
                trans = self.tf_buffer.lookup_transform('base_link', 'arm_right_7_link', rclpy.time.Time())
                start_pose = Pose()
                start_pose.position = trans.transform.translation
                start_pose.orientation = trans.transform.rotation
            except Exception as e:
                self.get_logger().warn(f"Failed to get current end-effector pose from TF: {e}")
                return

            self.interp_thread = threading.Thread(
                target=self.interpolate_poses,
                args=(start_pose, target_pose, feedback.header)
            )
            self.interp_thread.start()
            self.current_pose = target_pose  

    def interpolate_poses(self, start_pose, end_pose, header):
        duration = 1.0  
        steps = 50
        rate = duration / steps

        start_pos = np.array([start_pose.position.x, start_pose.position.y, start_pose.position.z])
        end_pos = np.array([end_pose.position.x, end_pose.position.y, end_pose.position.z])

        key_times = [0, 1]
        start_rot = R.from_quat([
            start_pose.orientation.x,
            start_pose.orientation.y,
            start_pose.orientation.z,
            start_pose.orientation.w
        ])
        end_rot = R.from_quat([
            end_pose.orientation.x,
            end_pose.orientation.y,
            end_pose.orientation.z,
            end_pose.orientation.w
        ])
        slerp = Slerp(key_times, R.concatenate([start_rot, end_rot]))

        for step in range(1, steps + 1):
            t = step / steps

            pos_interp = (1 - t) * start_pos + t * end_pos
            rot_interp = slerp([t])[0].as_quat()

            pose_stamped = PoseStamped()
            pose_stamped.header = header
            pose_stamped.pose.position.x = pos_interp[0]
            pose_stamped.pose.position.y = pos_interp[1]
            pose_stamped.pose.position.z = pos_interp[2]
            pose_stamped.pose.orientation.x = rot_interp[0]
            pose_stamped.pose.orientation.y = rot_interp[1]
            pose_stamped.pose.orientation.z = rot_interp[2]
            pose_stamped.pose.orientation.w = rot_interp[3]

            self.publisher.publish(pose_stamped)
            time.sleep(rate)

        self.current_pose = end_pose 

        
def main(args=None):
    rclpy.init(args=args)
    node = InteractiveMarkerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
