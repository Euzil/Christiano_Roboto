
"""
新的 talos.py 文件
包含Talos机器人类，支持站立控制和足步可视化
"""

import numpy as np
import pinocchio as pin
import pybullet as p

# simulator
from simulator.robot import Robot
# whole-body controller
from bullet_sims.tsid_wrapper import TSIDWrapper
# robot configs
import bullet_sims.talos_conf as conf
from bullet_sims.footstep_planner import Side

# ROS visualizations
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from visualization_msgs.msg import MarkerArray, Marker
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class Talos:
    """Talos robot
    combines wbc with pybullet, functions to read and set
    sensor values.
    """
    def __init__(self, simulator):
        self.conf = conf
        self.sim = simulator
        
        # Create the tsid wrapper for the whole body QP
        self.stack = TSIDWrapper(conf)   
        
        # spawn robot in simulation - Create the pybullet robot in the simulator
        self.robot = Robot(
            simulator=simulator,
            filename=conf.urdf,
            model=self.stack.model,  # 使用TSIDWrapper的model
            basePosition=np.array([0, 0, 1.1]),  # 根据需要调整初始高度
            q=conf.q_home,  # 使用配置的初始姿态
            useFixedBase=False  # Talos是floating base
        )
        
        ########################################################################
        # state
        ########################################################################
        self.support_foot = Side.RIGHT
        self.swing_foot = Side.LEFT
        
        ########################################################################
        # estimators
        ########################################################################
        self.zmp = np.zeros(3)
        self.dcm = np.zeros(3)
        
        ########################################################################
        # visualizations
        ########################################################################
        
        # Initialize ROS2 node if not already initialized
        try:
            rclpy.init()
        except:
            pass  # Already initialized
        
        self.ros_node = Node('talos_robot')
        
        # Joint state publisher
        self.joint_state_pub = self.ros_node.create_publisher(
            JointState, 
            'joint_states', 
            10
        )
        
        # Floating base broadcaster
        self.tf_broadcaster = TransformBroadcaster(self.ros_node)
        
        # ZMP and DCM point publisher - Use visualization_msgs::MarkerArray, SPHERE
        self.marker_pub = self.ros_node.create_publisher(
            MarkerArray,
            'robot_markers',
            10
        )
        
        # Wrench publisher for left and right foot - use geometry_msgs::Wrench
        self.left_wrench_pub = self.ros_node.create_publisher(
            WrenchStamped,
            'left_foot_wrench',
            10
        )
        
        self.right_wrench_pub = self.ros_node.create_publisher(
            WrenchStamped,
            'right_foot_wrench',
            10
        )
           
    def update(self):
        """updates the robot
        """
        t = self.sim.simTime()
        dt = 0.001  # 1ms 时间步长

        # Update the pybullet robot
        self.robot.update()
        
        # update the estimators
        self._update_zmp_estimate()
        self._update_dcm_estimate()
        
        # update wbc and send back to pybullet
        self._solve(t, dt)
        
    def setSupportFoot(self, side):
        """sets the the support foot of the robot on given side
        """
        
        # The support foot is in rigid contact with the ground and should 
        # hold the weight of the robot
        self.support_foot = side
        
        # Activate the foot contact on the support foot
        if side == Side.LEFT:
            self.stack.activateContact("left_foot_contact")
            # Deactivate the motion task on the support foot
            self.stack.deactivateTask("left_foot_motion")
        else:  # Side.RIGHT
            self.stack.activateContact("right_foot_contact")
            # Deactivate the motion task on the support foot
            self.stack.deactivateTask("right_foot_motion")
    
    def setSwingFoot(self, side):
        """sets the swing foot of the robot on given side
        """
        
        # The swing foot is not in contact and can move
        self.swing_foot = side
        
        # Deactivate the foot contact on the swing foot
        if side == Side.LEFT:
            self.stack.deactivateContact("left_foot_contact")
            # Turn on the motion task on the swing foot
            self.stack.activateTask("left_foot_motion")
        else:  # Side.RIGHT
            self.stack.deactivateContact("right_foot_contact")
            # Turn on the motion task on the swing foot
            self.stack.activateTask("right_foot_motion")
        
    def updateSwingFootRef(self, T_swing_w, V_swing_w, A_swing_w):
        """updates the swing foot motion reference
        """
        
        # Set the pose, velocity and acceleration on the swing foot's motion task
        if self.swing_foot == Side.LEFT:
            self.stack.setTaskReference("left_foot_motion", T_swing_w, V_swing_w, A_swing_w)
        else:  # Side.RIGHT
            self.stack.setTaskReference("right_foot_motion", T_swing_w, V_swing_w, A_swing_w)

    def swingFootPose(self):
        """return the pose of the current swing foot
        """
        # Return correct foot pose using TSIDWrapper
        if self.swing_foot == Side.LEFT:
            return self.stack.getFramePose(self.conf.lf_frame_name)
        else:  # Side.RIGHT
            return self.stack.getFramePose(self.conf.rf_frame_name)
    
    def supportFootPose(self):
        """return the pose of the current support foot
        """
        # Return correct foot pose using TSIDWrapper
        if self.support_foot == Side.LEFT:
            return self.stack.getFramePose(self.conf.lf_frame_name)
        else:  # Side.RIGHT
            return self.stack.getFramePose(self.conf.rf_frame_name)

    def publish(self):        
        # Publish the jointstate
        joint_msg = JointState()
        joint_msg.header.stamp = self.ros_node.get_clock().now().to_msg()
        
        # 获取关节名称 - 使用Robot类的方法
        joint_msg.name = self.robot.actuatedJointNames()
        
        # 获取关节状态 - 从Robot类获取
        joint_msg.position = self.robot.actuatedJointPosition().tolist()
        joint_msg.velocity = self.robot.actuatedJointVelocity().tolist()
        self.joint_state_pub.publish(joint_msg)
        
        # Broadcast odometry (floating base)
        transform = TransformStamped()
        transform.header.stamp = self.ros_node.get_clock().now().to_msg()
        transform.header.frame_id = "world"
        transform.child_frame_id = "base_link"
        
        # Get base pose from Robot class
        base_pos, base_quat = self.robot.baseWorldPose()
        
        transform.transform.translation.x = float(base_pos[0])
        transform.transform.translation.y = float(base_pos[1])
        transform.transform.translation.z = float(base_pos[2])
        transform.transform.rotation.x = float(base_quat[0])
        transform.transform.rotation.y = float(base_quat[1])
        transform.transform.rotation.z = float(base_quat[2])
        transform.transform.rotation.w = float(base_quat[3])
        
        self.tf_broadcaster.sendTransform(transform)
        
        # Publish feet wrenches - 使用TSIDWrapper获取接触力
        contact_forces = self.stack.getContactForces()
        
        # Left foot wrench
        left_wrench_msg = WrenchStamped()
        left_wrench_msg.header.stamp = self.ros_node.get_clock().now().to_msg()
        left_wrench_msg.header.frame_id = self.conf.lf_frame_name
        left_ft = contact_forces.get("left_foot", np.zeros(6))
        left_wrench_msg.wrench.force.x = float(left_ft[0])
        left_wrench_msg.wrench.force.y = float(left_ft[1])
        left_wrench_msg.wrench.force.z = float(left_ft[2])
        left_wrench_msg.wrench.torque.x = float(left_ft[3])
        left_wrench_msg.wrench.torque.y = float(left_ft[4])
        left_wrench_msg.wrench.torque.z = float(left_ft[5])
        self.left_wrench_pub.publish(left_wrench_msg)
        
        # Right foot wrench
        right_wrench_msg = WrenchStamped()
        right_wrench_msg.header.stamp = self.ros_node.get_clock().now().to_msg()
        right_wrench_msg.header.frame_id = self.conf.rf_frame_name
        right_ft = contact_forces.get("right_foot", np.zeros(6))
        right_wrench_msg.wrench.force.x = float(right_ft[0])
        right_wrench_msg.wrench.force.y = float(right_ft[1])
        right_wrench_msg.wrench.force.z = float(right_ft[2])
        right_wrench_msg.wrench.torque.x = float(right_ft[3])
        right_wrench_msg.wrench.torque.y = float(right_ft[4])
        right_wrench_msg.wrench.torque.z = float(right_ft[5])
        self.right_wrench_pub.publish(right_wrench_msg)
        
        # Publish DCM and ZMP marker
        marker_array = MarkerArray()
        
        # ZMP marker
        zmp_marker = Marker()
        zmp_marker.header.stamp = self.ros_node.get_clock().now().to_msg()
        zmp_marker.header.frame_id = "world"
        zmp_marker.ns = "robot_state"
        zmp_marker.id = 0
        zmp_marker.type = Marker.SPHERE
        zmp_marker.action = Marker.ADD
        zmp_marker.pose.position.x = float(self.zmp[0])
        zmp_marker.pose.position.y = float(self.zmp[1])
        zmp_marker.pose.position.z = float(self.zmp[2])
        zmp_marker.pose.orientation.w = 1.0
        zmp_marker.scale.x = 0.05
        zmp_marker.scale.y = 0.05
        zmp_marker.scale.z = 0.05
        zmp_marker.color.r = 1.0  # Red for ZMP
        zmp_marker.color.g = 0.0
        zmp_marker.color.b = 0.0
        zmp_marker.color.a = 1.0
        marker_array.markers.append(zmp_marker)
        
        # DCM marker
        dcm_marker = Marker()
        dcm_marker.header.stamp = self.ros_node.get_clock().now().to_msg()
        dcm_marker.header.frame_id = "world"
        dcm_marker.ns = "robot_state"
        dcm_marker.id = 1
        dcm_marker.type = Marker.SPHERE
        dcm_marker.action = Marker.ADD
        dcm_marker.pose.position.x = float(self.dcm[0])
        dcm_marker.pose.position.y = float(self.dcm[1])
        dcm_marker.pose.position.z = float(self.dcm[2])
        dcm_marker.pose.orientation.w = 1.0
        dcm_marker.scale.x = 0.05
        dcm_marker.scale.y = 0.05
        dcm_marker.scale.z = 0.05
        dcm_marker.color.r = 0.0
        dcm_marker.color.g = 1.0  # Green for DCM
        dcm_marker.color.b = 0.0
        dcm_marker.color.a = 1.0
        marker_array.markers.append(dcm_marker)
        
        self.marker_pub.publish(marker_array)

    ############################################################################
    # private functions
    ############################################################################

    def _solve(self, t, dt):
        # get the current state from Robot
        q = self.robot.q()
        v = self.robot.v()
        
        # solve the whole body qp using TSIDWrapper
        tau, dv = self.stack.update(q, v, t, do_sove=True)
        
        # command torques to robot using Robot class method
        self.robot.setActuatedJointTorques(tau)
    
    def _update_zmp_estimate(self):
        """update the estimated zmp position
        """
        # 使用TSIDWrapper获取接触力
        contact_forces = self.stack.getContactForces()
        left_ft = contact_forces.get("left_foot", np.zeros(6))
        right_ft = contact_forces.get("right_foot", np.zeros(6))
        
        # Get foot positions from TSIDWrapper
        left_pos = self.stack.getFramePose(self.conf.lf_frame_name)[:3, 3]
        right_pos = self.stack.getFramePose(self.conf.rf_frame_name)[:3, 3]
        
        # Extract forces (first 3 elements are forces)
        left_force = left_ft[:3]
        right_force = right_ft[:3]
        
        # Total force
        total_force = left_force + right_force
        
        if np.abs(total_force[2]) > 1e-6:  # Avoid division by zero
            # Weighted average position based on vertical forces
            weight_left = np.abs(left_force[2]) / np.abs(total_force[2])
            weight_right = np.abs(right_force[2]) / np.abs(total_force[2])
            
            self.zmp = weight_left * left_pos + weight_right * right_pos
        else:
            # If no vertical force, use center between feet
            self.zmp = 0.5 * (left_pos + right_pos)
        
    def _update_dcm_estimate(self):
        """update the estimated dcm position
        """
        # 使用TSIDWrapper获取质心状态
        com_pos = self.stack.getCenterOfMass()
        com_vel = self.stack.getCenterOfMassVelocity()
        
        # DCM calculation: ξ = c + c_dot / ω₀
        # where ω₀ = sqrt(g/z_com) for the linear inverted pendulum
        g = 9.81
        omega_0 = np.sqrt(g / max(com_pos[2], 0.1))  # Avoid division by zero
        
        self.dcm = com_pos + com_vel / omega_0