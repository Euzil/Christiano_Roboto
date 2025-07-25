"""
Fixed AiNEx Walking Control - Resolving COM reference issues
"""

import numpy as np
import pinocchio as pin
from whole_body_control.tsid_wrapper import TSIDWrapper
import whole_body_control.config as conf
import rclpy
import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import WrenchStamped
from visualization_msgs.msg import Marker, MarkerArray
from enum import Enum

################################################################################
# Enum Side for differenciating feet
################################################################################

class Side(Enum):
    """Side
    Describes which foot to use
    """
    LEFT=0
    RIGHT=1

################################################################################
# Enhanced AiNex Robot with Fixed COM Reference Methods
################################################################################

class Ainex():
    """
    Enhanced AiNex robot class with fixed COM reference handling
    """
    def __init__(self,q=None):
        
        # Store configuration
        self.conf = conf
        
        self.stack = TSIDWrapper(self.conf)
        
        ########################################################################
        # ROS2 Setup
        ########################################################################
        
        self.node = rclpy.create_node('ainex_robot_publisher')
        self.joint_state_pub = self.node.create_publisher(JointState, '/joint_states', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self.node)
        
        # Initialize joint state message
        self.joint_state_msg = JointState()
        self.joint_state_msg.header.frame_id = "world"

        # Set joint names
        self.joint_state_msg.name = ['head_pan', 'head_tilt',
                                    'l_hip_yaw', 'l_hip_roll', 'l_hip_pitch', 'l_knee', 'l_ank_pitch', 'l_ank_roll',
                                    'l_sho_pitch', 'l_sho_roll', 'l_el_pitch', 'l_el_yaw', 'l_gripper',
                                    'r_hip_yaw', 'r_hip_roll', 'r_hip_pitch', 'r_knee', 'r_ank_pitch', 'r_ank_roll',
                                    'r_sho_pitch', 'r_sho_roll', 'r_el_pitch', 'r_el_yaw', 'r_gripper']
        
        # Enhanced publishers
        self.marker_pub = self.node.create_publisher(MarkerArray, 'robot_markers', 10)
        self.left_wrench_pub = self.node.create_publisher(WrenchStamped, 'left_foot_wrench', 10)
        self.right_wrench_pub = self.node.create_publisher(WrenchStamped, 'right_foot_wrench', 10)
        
        # Last computed torques
        self.tau = np.zeros(self.conf.na)

        self.q = q
        self.v = np.zeros((conf.na + 6,))
        
        ########################################################################
        # State machine variables
        ########################################################################
        self.robot_state = "STANDING"
        self.standing_stable = False
        self.path_generated = False
        
        # Standing control parameters
        self.com_target = None
        self.standing_start_time = 0.0
        self.stabilization_threshold = 0.05
        self.required_stable_time = 2.0
        
        # Walking control parameters
        self.footstep_plan = []
        self.current_step_index = 0
        self.step_start_time = 0.0
        self.support_foot = Side.RIGHT
        self.swing_foot = Side.LEFT

    ############################################################################
    # Core methods
    ############################################################################

    def update(self, q, v):
        """Update base class and estimators"""
        self.q = q
        self.v = v

    def get_q(self):
        return self.q
    
    def get_v(self):
        return self.v

    def publish(self, T_b_w):
        """Publish robot state at 30 Hz"""
        current_time = self.node.get_clock().now()
        
        # Publish joint state
        self.joint_state_msg.header.stamp = current_time.to_msg()
        
        # Get current robot state
        q_current = np.array(self.q)
        v_current = np.array(self.v)
        
        # Extract joint positions and velocities
        if len(q_current) > self.conf.na:
            joint_positions = q_current[7:7+self.conf.na]
            joint_velocities = v_current[6:6+self.conf.na]
        else:
            joint_positions = q_current[:self.conf.na]
            joint_velocities = v_current[:self.conf.na]
        
        self.joint_state_msg.position = joint_positions.tolist()
        self.joint_state_msg.velocity = joint_velocities.tolist()
        
        if hasattr(self, 'tau') and self.tau is not None:
            tau_array = np.array(self.tau)
            if len(tau_array) >= self.conf.na:
                self.joint_state_msg.effort = tau_array[:self.conf.na].tolist()
        
        self.joint_state_pub.publish(self.joint_state_msg)
        
        # Broadcast transformation
        self._broadcast_base_transform(T_b_w, current_time)

    def _broadcast_base_transform(self, T_b_w, timestamp):
        """Broadcast base transformation"""
        if T_b_w is not None:
            transform_msg = TransformStamped()
            transform_msg.header.stamp = timestamp.to_msg()
            transform_msg.header.frame_id = "world"
            transform_msg.child_frame_id = "base_link"
            
            if hasattr(T_b_w, 'translation') and hasattr(T_b_w, 'rotation'):
                transform_msg.transform.translation.x = float(T_b_w.translation[0])
                transform_msg.transform.translation.y = float(T_b_w.translation[1])
                transform_msg.transform.translation.z = float(T_b_w.translation[2])
                
                quat = pin.Quaternion(T_b_w.rotation)
                transform_msg.transform.rotation.x = float(quat.x)
                transform_msg.transform.rotation.y = float(quat.y)
                transform_msg.transform.rotation.z = float(quat.z)
                transform_msg.transform.rotation.w = float(quat.w)
            else:
                transform_msg.transform.translation.z = 0.9
                transform_msg.transform.rotation.w = 1.0
            
            self.tf_broadcaster.sendTransform(transform_msg)