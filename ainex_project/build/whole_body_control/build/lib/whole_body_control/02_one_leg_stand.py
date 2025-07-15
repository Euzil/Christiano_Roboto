import numpy as np
from numpy import nan
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt

# pinocchio
import pinocchio as pin

# simulator
import pybullet as pb
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot

# robot and controller
from whole_body_control.tsid_wrapper import TSIDWrapper
import whole_body_control.config as conf

# ROS
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped

################################################################################
# settings
################################################################################

DO_PLOT = True

################################################################################
# Robot
################################################################################


class Talos(Robot):
    def __init__(self, simulator, urdf, model, node, q=None, verbose=True, useFixedBase=True):
        # call base class constructor

        # Initial condition for the simulator an model
        z_init = 1.15

        super().__init__(
            simulator,
            urdf,
            model,
            basePosition=[0, 0, z_init],
            baseQuationerion=[0, 0, 0, 1],
            q=q,
            useFixedBase=useFixedBase,
            verbose=verbose)

        self.node = node

        # add publisher
        self.pub_joint = self.node.create_publisher(
            JointState, "/joint_states", 10)

        self.joint_msg = JointState()
        self.joint_msg.name = self.actuatedJointNames()

        # add tf broadcaster
        self.br = tf2_ros.TransformBroadcaster(self.node)

    def update(self):
        # update base class
        super().update()

    def publish(self, T_b_w, tau):
        # publish jointstate
        self.joint_msg.header.stamp = self.node.get_clock().now().to_msg()
        self.joint_msg.position = self.actuatedJointPosition().tolist()
        self.joint_msg.velocity = self.actuatedJointVelocity().tolist()
        self.joint_msg.effort = tau.tolist()

        self.pub_joint.publish(self.joint_msg)

        # broadcast transformation T_b_w
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.node.get_clock().now().to_msg()
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = self.baseName()

        tf_msg.transform.translation.x = T_b_w.translation[0]
        tf_msg.transform.translation.y = T_b_w.translation[1]
        tf_msg.transform.translation.z = T_b_w.translation[2]

        q = pin.Quaternion(T_b_w.rotation)
        q.normalize()
        tf_msg.transform.rotation.x = q.x
        tf_msg.transform.rotation.y = q.y
        tf_msg.transform.rotation.z = q.z
        tf_msg.transform.rotation.w = q.w

        self.br.sendTransform(tf_msg)


################################################################################
# Application
################################################################################


class Environment(Node):
    def __init__(self):
        super().__init__('tutorial_4_standing_node')

        # init TSIDWrapper
        self.tsid_wrapper = TSIDWrapper(conf)

        # init Simulator
        self.simulator = PybulletWrapper()

        q_init = np.hstack([np.array([0, 0, 1.15, 0, 0, 0, 1]),
                           np.zeros_like(conf.q_actuated_home)])

        # init ROBOT
        self.robot = Talos(
            self.simulator,
            conf.urdf,
            self.tsid_wrapper.model,
            self,
            q=q_init,
            verbose=True,
            useFixedBase=False)

        self.t_publish = 0.0
        
        # One-leg stand phase tracking
        self.com_shift_started = False
        self.left_foot_lifted = False

    def update(self):
        # elapsed time
        t = self.simulator.simTime()
        
        # One-leg stand behavior
        self._update_one_leg_stand(t)

        # update the simulator and the robot
        self.simulator.step()
        self.simulator.debug()
        self.robot.update()

        # update TSID controller
        tau_sol, _ = self.tsid_wrapper.update(
            self.robot.q(), self.robot.v(), t)

        # command to the robot
        self.robot.setActuatedJointTorques(tau_sol)

        # publish to ros
        if t - self.t_publish > 1./30.:
            self.t_publish = t
            # get current BASE Pose
            T_b_w, _ = self.tsid_wrapper.baseState()
            self.robot.publish(T_b_w, tau_sol)
    
    def _update_one_leg_stand(self, t):
        # Phase 1: Shift COM to right foot position (at start)
        if not self.com_shift_started:
            # Get current COM position from TSID
            p_com_current = self.tsid_wrapper.comState().pos()
            
            # Get current right foot position
            rf_pos = self.tsid_wrapper.get_placement_RF().translation
            
            # Set new COM reference (keep same height, shift XY to right foot)
            p_com_new = np.array([rf_pos[0], rf_pos[1], p_com_current[2]])
            self.tsid_wrapper.setComRefState(p_com_new)
            
            self.com_shift_started = True
        
        # Phase 2: After 2 seconds, remove left foot contact and lift it
        if t >= 2.0 and not self.left_foot_lifted:
            # Remove left foot contact
            self.tsid_wrapper.remove_contact_LF()
            
            # Get current left foot placement and extract translation and rotation
            lf_current = self.tsid_wrapper.get_placement_LF()
            lf_current_pos = lf_current.translation
            lf_current_rot = lf_current.rotation
            
            print(f"Current left foot position: {lf_current_pos}")
            print(f"Current left foot rotation: {lf_current_rot}")
            
            # Create new position 0.3m higher and flatten rotation matrix
            lf_new_pos = np.concatenate([
                lf_current_pos + np.array([0, 0, 0.3]),  # New translation
                lf_current_rot.flatten()                # Flattened rotation matrix
            ])
            
            self.tsid_wrapper.set_LF_pose_ref(lf_new_pos)
            
            self.left_foot_lifted = True

################################################################################
# main
################################################################################


def main(args=None):
    rclpy.init(args=args)
    env = Environment()
    try:
        while rclpy.ok():
            env.update()

    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        env.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
