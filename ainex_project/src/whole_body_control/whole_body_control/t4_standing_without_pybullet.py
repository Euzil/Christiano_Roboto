import numpy as np
from numpy import nan
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt

# pinocchio
import pinocchio as pin

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


class Ainex():
    def __init__(self, node, q=None):

        self.node = node

        # add publisher
        self.pub_joint = self.node.create_publisher(
            JointState, "/joint_states", 10)

        self.joint_msg = JointState()
        self.joint_msg.name = ['r_hip_yaw', 'r_hip_roll', 'r_hip_pitch', 'r_knee', 
                               'r_ank_pitch', 'r_ank_roll', 'l_hip_yaw', 'l_hip_roll', 
                               'l_hip_pitch', 'l_knee', 'l_ank_pitch', 'l_ank_roll', 
                               'head_pan', 'head_tilt', 'r_sho_pitch', 'r_sho_roll', 
                               'r_el_pitch', 'r_el_yaw', 'r_gripper', 'l_sho_pitch', 
                               'l_sho_roll', 'l_el_pitch', 'l_el_yaw', 'l_gripper']

        # add tf broadcaster
        self.br = tf2_ros.TransformBroadcaster(self.node)

        self.q = q

        self.v = np.zeros((conf.na + 6,))

        self.tau = np.zeros((conf.na,))

    def update(self, q, v, tau):
        # update base class
        self.q = q
        self.v = v
        self.tau = tau

    def get_q(self):
        return self.q

    def get_v(self):
        return self.v

    def publish(self, T_b_w):
        # publish jointstate
        self.joint_msg.header.stamp = self.node.get_clock().now().to_msg()
        self.joint_msg.position = self.q[7:].tolist()
        self.joint_msg.velocity = self.v[6:].tolist()
        self.joint_msg.effort = self.tau

        self.pub_joint.publish(self.joint_msg)

        # broadcast transformation T_b_w
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.node.get_clock().now().to_msg()
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = "base_link"

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


class T4StandingRobotSimNode(Node):
    def __init__(self):
        super().__init__('tutorial_4_standing_node')

        z_init = 0.23

        # init TSIDWrapper
        self.tsid_wrapper = TSIDWrapper(conf)

        # init Simulator
        q_init = np.hstack([np.array([0, 0, z_init, 0, 0, 0, 1]), np.zeros(conf.na)])

        # init ROBOT - change class name
        self.robot = Ainex(self, q_init)

        # init simulation time
        self.t = 0.0

        # init q_tsid, v_tsid
        self.q_tsid = self.robot.get_q()
        self.v_tsid = self.robot.get_v()
        self.tau = np.zeros((conf.na,))

        # Set a timer to run periodically 
        self.timer_frequenz = 30 
        self.timer = self.create_timer(1/self.timer_frequenz, self.timer_callback)

    def timer_callback(self):
        
        # change KP and Kd of right hand 
        self.tsid_wrapper.rightHandTask.setKp(100*np.array([1,1,1,0,0,0]))
        self.tsid_wrapper.rightHandTask.setKd(2.0*np.sqrt(100)*np.array([1,1,1,0,0,0]))

        # activate hand motion
        self.tsid_wrapper.add_motion_RH()

        # get current right hand position
        rh_pos_current = self.tsid_wrapper.get_pose_RH().translation

        # define the desired right hand position such that it is at the starting position of the circle
        rh_pos_current[0] = 0.4
        rh_pos_current[1] = 0
        rh_pos_current[2] = 1.1

        # update reference position
        self.tsid_wrapper.set_RH_pos_ref(rh_pos_current, np.zeros((3,)), np.zeros((3,)))

        # update robot simulator
        self.robot.update(self.q_tsid, self.v_tsid, self.tau) 

        # TODO: update TSID controller
        tau_sol, dv_sol = self.tsid_wrapper.update(self.robot.get_q(), self.robot.get_v(), self.t)  
        self.tau = tau_sol

        # integrate dv_sol for position control
        self.q_tsid, self.v_tsid = self.tsid_wrapper.integrate_dv(self.q_tsid, self.v_tsid, dv_sol, 1/self.timer_frequenz)      

        self.get_logger().info(str(self.q_tsid.shape))

        # TODO:command to the hardware robot - should have reached q_tsid for next timer call

        # get current BASE Pose
        T_b_w, _ = self.tsid_wrapper.baseState()

        # publish transformation and joint states
        self.robot.publish(T_b_w)

        # update simulator time
        self.t = self.t + 1/self.timer_frequenz


################################################################################
# main
################################################################################


def main(args=None):
    rclpy.init(args=args)
    node = T4StandingRobotSimNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
