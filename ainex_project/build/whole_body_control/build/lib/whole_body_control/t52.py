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
            useFixedBase=False,
            verbose=verbose)

        self.node = node

        # add publisher
        self.pub_joint = self.node.create_publisher(
            JointState, "/joint_states", 10)

        self.joint_msg = JointState()
        self.joint_msg.name = self.actuatedJointNames()

        # add tf broadcaster
        self.br = tf2_ros.TransformBroadcaster(self.node)

        pb.enableJointForceTorqueSensor(self.id(), self.jointNameIndexMap()["leg_right_6_joint"], True)
        pb.enableJointForceTorqueSensor(self.id(), self.jointNameIndexMap()["leg_left_6_joint"], True)

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


class StateMachine:
    def __init__(self, robot, simulator):
        print("StateMachine initialized")
        self.robot = robot
        self.simulator = simulator
        self.state = "right"
        self.t_start = self.simulator.simTime()
        self.t_push = 0.5  # duration of force application
        self.t_period = 4.0  # interval between force applications
        self.force_magnitude = 45.0  # Start with a small force magnitude
        self.line_id = -1
        self.current_force = None  # Track current force being applied
        self.pushing = False  # Flag to track if currently pushing

        # Gains for ankle strategy
        self.Kx = 2.5  # Gain for CoM velocity adjustment 2.5
        self.Kp = 0.3  # Gain for ZMP position adjustment

        # Gains for hip strategy
        self.Kr = 8.2  # Gain for angular momentum adjustment

        # Reference positions
        self.x_ref = np.array([0, 0]) # Reference CoM velocity (assumed zero for standing)
        self.p_ref = np.zeros(2)  # Reference ZMP position (assumed zero for standing)
        self.r_ref = np.zeros(2)  # Reference CMP position (assumed zero for standing)

    def update(self):
        # Apply balance strategies first
        self.apply_ankle_strategy()
        self.apply_hip_strategy()

        t = self.simulator.simTime()
        elapsed = t - self.t_start

        if self.state == "right" and elapsed > self.t_period:
            self.start_pushing([self.force_magnitude, 0, 0])
            self.state = "right_pushing"
            self.t_start = t
        elif self.state == "right_pushing" and elapsed > self.t_push:
            self.stop_pushing()
            self.state = "left"
            self.t_start = t
        elif self.state == "left" and elapsed > self.t_period:
            self.start_pushing([-self.force_magnitude, 0, 0])
            self.state = "left_pushing"
            self.t_start = t
        elif self.state == "left_pushing" and elapsed > self.t_push:
            self.stop_pushing()
            self.state = "back"
            self.t_start = t
        elif self.state == "back" and elapsed > self.t_period:
            self.start_pushing([0, -self.force_magnitude, 0])
            self.state = "back_pushing"
            self.t_start = t
        elif self.state == "back_pushing" and elapsed > self.t_push:
            self.stop_pushing()
            self.state = "done"
            self.t_start = t

        # Apply force continuously while pushing
        if self.pushing and self.current_force is not None:
            self.apply_continuous_force()

    def apply_ankle_strategy(self):
        """Adjust CoM velocity based on ZMP feedback."""
        # Get current CoM position and velocity
        com_state = self.robot.node.tsid_wrapper.comState()
        com_pos = com_state.pos()[:2]  # Only x, y, and z components
        com_vel = com_state.vel()[:2]  # Only x, y, and z components

        # Get current ZMP position
        zmp_pos = self.robot.node.history_zmp[-1] if self.robot.node.history_zmp else np.zeros(2)

        # Calculate desired CoM velocity using ankle strategy
        x_d = self.x_ref - self.Kx * (com_pos - self.x_ref) + self.Kp * (zmp_pos - self.p_ref)

        # Update CoM reference state using tsid_wrapper's setComRefState method
        self.robot.node.tsid_wrapper.setComRefState(com_pos, x_d)

    def apply_hip_strategy(self):
        """Adjust angular momentum based on CMP feedback."""
        # Get current CMP position
        cmp_pos = self.robot.node.history_cmp[-1] if self.robot.node.history_cmp else np.zeros(2)

        # Calculate desired angular momentum using hip strategy
        # Γ_d = K_Γ * (r - r_ref)
        gamma_d = self.Kr * (cmp_pos - self.r_ref)

        # Update the angular momentum reference using the existing AM task
        # Create a sample with the desired angular momentum
        from whole_body_control.tsid_wrapper import create_sample
        am_ref = create_sample(gamma_d)
        self.robot.node.tsid_wrapper.amTask.setReference(am_ref)

    def start_pushing(self, force):
        """Start applying continuous force"""
        self.current_force = force
        self.pushing = True
        print(f"Starting continuous push with force: {force}")
        self.visualize_force(force, self.get_hip_position())

    def stop_pushing(self):
        """Stop applying force"""
        self.pushing = False
        self.current_force = None
        print("Stopping push")
        self.remove_force()

    def apply_continuous_force(self):
        """Apply force continuously every frame"""
        if self.current_force is not None:
            position = self.get_hip_position()
            self.robot.applyForce(self.current_force, position)

    def visualize_force(self, force, position):
        """Visualize the force direction using a debug line"""
        p1 = position
        p2 = p1 + np.array(force) / self.force_magnitude  # Scale for visualization
        self.line_id = self.simulator.addGlobalDebugLine(p1, p2, line_id=self.line_id, color=[1, 0, 0])

    def remove_force(self):
        """Remove the debug line"""
        if self.line_id != -1:
            self.simulator.removeDebugItem(self.line_id)
            self.line_id = -1

    def get_hip_position(self):
        """Get current hip/base position for visualization"""
        hip_position = self.robot.q()[:3]  
        return hip_position

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

        # init StateMachine
        self.state_machine = StateMachine(self.robot, self.simulator)

        # Data recording for plotting
        self.history_t = []
        self.history_com = []
        self.history_zmp = []
        self.history_cmp = []
        self.history_cp = []

        # Initialize virtual states for TSID integration
        self.q_tsid = q_init.copy()
        self.v_tsid = np.zeros(self.tsid_wrapper.robot.nv)
        self.n_update = 1  # Number of TSID updates per simulation step

    def update(self):
        # elapsed time
        t = self.simulator.simTime()

        # update the simulator and the robot
        self.simulator.step()
        self.simulator.debug()
        self.robot.update()

        # Get robot state
        q = self.robot.q()

        # update TSID controller
        tau_sol, dv_sol = self.tsid_wrapper.update(q, self.robot.v(), t)
        
        # Read Ankle Wrenches
        wren = pb.getJointState(self.robot.id(), self.robot.jointNameIndexMap()["leg_right_6_joint"])[2]
        wnp = np.array([-wren[0], -wren[1], -wren[2], -wren[3], -wren[4], -wren[5]])
        wr_rankle = pin.Force(wnp)
        wren = pb.getJointState(self.robot.id(), self.robot.jointNameIndexMap()["leg_left_6_joint"])[2]
        wnp = np.array([-wren[0], -wren[1], -wren[2], -wren[3], -wren[4], -wren[5]])
        wl_lankle = pin.Force(wnp)

        # Position of the ankles and the soles 
        data = self.robot._model.createData()
        pin.framesForwardKinematics(self.robot._model, data, q)
        H_w_lsole = data.oMf[self.robot._model.getFrameId("left_sole_link")]
        H_w_rsole = data.oMf[self.robot._model.getFrameId("right_sole_link")]
        H_w_lankle = data.oMf[self.robot._model.getFrameId("leg_left_6_joint")]
        H_w_rankle = data.oMf[self.robot._model.getFrameId("leg_right_6_joint")]

        # CoM position and velocity
        com_state = self.tsid_wrapper.comState()
        com_pos = com_state.pos()
        com_vel = com_state.vel()

        # Initialize d
        d_r = 0.1
        d_l = 0.1

        # Estimate ZMPs in ankle frames
        p_zmp_r_local = self.estimate_zmp(wr_rankle.angular, wr_rankle.linear, d_r)
        p_zmp_l_local = self.estimate_zmp(wl_lankle.angular, wl_lankle.linear, d_l)
        
        # Transform ZMPs to world frame
        p_zmp_r_world = H_w_rankle.act(p_zmp_r_local)
        p_zmp_l_world = H_w_lankle.act(p_zmp_l_local)

        # Estimate combined ZMP
        f_r_world = H_w_rankle.rotation @ wr_rankle.linear
        f_l_world = H_w_lankle.rotation @ wl_lankle.linear
        fz_r = f_r_world[2]
        fz_l = f_l_world[2]
        zmp_combined = self.estimate_zmp_combined(p_zmp_r_world, fz_r, p_zmp_l_world, fz_l)

        # Estimate CMP
        f_total_world = f_r_world + f_l_world
        cmp = self.estimate_cmp(com_pos, f_total_world)

        # Estimate CP
        cp = self.estimate_cp_dcm(com_pos, com_vel)

        # Store data for plotting
        if DO_PLOT:
            self.history_t.append(t)
            self.history_com.append(com_pos[:2])
            self.history_zmp.append(zmp_combined[:2])
            self.history_cmp.append(cmp[:2])
            self.history_cp.append(cp[:2])

        # TASK 2: Replace setActuatedJointTorques with position control
        # integrate sol in virtual state using TSID wrapper's integrate_dv method
        self.q_tsid, self.v_tsid = self.tsid_wrapper.integrate_dv(
            self.q_tsid, 
            self.v_tsid, 
            dv_sol, 
            self.n_update * self.simulator.stepTime()
        )
        self.robot.setActuatedJointPositions(self.q_tsid, self.v_tsid)

        # update state machine
        self.state_machine.update()

        # publish to ros
        if t - self.t_publish > 1./30.:
            self.t_publish = t
            # get current BASE Pose
            T_b_w, _ = self.tsid_wrapper.baseState()
            self.robot.publish(T_b_w, tau_sol)

    def estimate_zmp_combined(self, p_r, fz_r, p_l, fz_l):
        """Estimate the combined ZMP from both feet based on individual ZMPs and vertical forces."""
        fz_total = fz_r + fz_l
        if abs(fz_total) < 1e-6:  # Avoid division by zero
            # If total vertical force is negligible, ZMP is ill-defined.
            # A reasonable fallback is the midpoint of the individual ZMPs.
            return (p_r + p_l) / 2.0

        px = (p_r[0] * fz_r + p_l[0] * fz_l) / fz_total
        py = (p_r[1] * fz_r + p_l[1] * fz_l) / fz_total
        pz = 0.0
        return np.array([px, py, pz])

    def estimate_zmp(self, tau, f, d):
        """Estimate the Zero Moment Point (ZMP) based on the given parameters."""
        px_foot = (-tau[1] - f[0] * d) / f[2]
        py_foot = (tau[0] - f[1] * d) / f[2]
        pz_foot = 0
        return np.array([px_foot, py_foot, pz_foot])

    def estimate_cmp(self, com_pos, f):
        """Estimate the Centroidal Moment Pivot (CMP) based on the given parameters."""
        if abs(f[2]) < 1e-6:
            return np.array([com_pos[0], com_pos[1], 0.0])
        cmp_x = com_pos[0] - (f[0] / f[2]) * com_pos[2]
        cmp_y = com_pos[1] - (f[1] / f[2]) * com_pos[2]
        cmp_z = 0.0
        return np.array([cmp_x, cmp_y, cmp_z])

    def estimate_cp_dcm(self, com_pos, com_vel):
        """Estimate the Capture Point (CP) or Divergent Component of Motion (DCM) based on the given parameters."""
        g = 9.81  # magnitude of gravity
        xz = com_pos[2]


        omega = np.sqrt(g / xz)
        # CoM position and velocity projected to the ground
        xp = com_pos
        xp[2] = 0 # Ignore the z-component for 2D projection
        xp_dot = com_vel
        xp_dot[2] = 0 # Ignore the z-component for 2D projection
        
        cp = xp + xp_dot / omega
        
        return np.array([cp[0], cp[1], 0.0])
    

    @staticmethod
    def moving_average(data, window_size):
        """Apply a moving average filter to smooth the data."""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def plot_data(self):
        if not self.history_t:
            return

        # Convert lists to numpy arrays for easier indexing
        history_com = np.array(self.history_com)
        history_zmp = np.array(self.history_zmp)
        history_cmp = np.array(self.history_cmp)
        history_cp = np.array(self.history_cp)

        # Check if we have enough data points
        if len(self.history_t) < 20:  # Need at least 20 points for meaningful smoothing
            print(f"Not enough data points for plotting: {len(self.history_t)}")
            return

        # Apply moving average smoothing
        window_size = min(10, len(self.history_t) // 2)  # Adaptive window size
        smoothed_com_x = self.moving_average(history_com[:, 0], window_size)
        smoothed_zmp_x = self.moving_average(history_zmp[:, 0], window_size)
        smoothed_cmp_x = self.moving_average(history_cmp[:, 0], window_size)
        smoothed_cp_x = self.moving_average(history_cp[:, 0], window_size)

        smoothed_com_y = self.moving_average(history_com[:, 1], window_size)
        smoothed_zmp_y = self.moving_average(history_zmp[:, 1], window_size)
        smoothed_cmp_y = self.moving_average(history_cmp[:, 1], window_size)
        smoothed_cp_y = self.moving_average(history_cp[:, 1], window_size)

        # Adjust time array for smoothing - match the length of smoothed data
        # The moving average with mode='valid' returns an array of length (N - window_size + 1)
        time_array = np.array(self.history_t)
        start_idx = window_size - 1
        smoothed_time = time_array[start_idx:start_idx + len(smoothed_com_x)]

        # Ensure all arrays have the same length
        min_length = min(len(smoothed_time), len(smoothed_com_x))
        smoothed_time = smoothed_time[:min_length]
        smoothed_com_x = smoothed_com_x[:min_length]
        smoothed_zmp_x = smoothed_zmp_x[:min_length]
        smoothed_cmp_x = smoothed_cmp_x[:min_length]
        smoothed_cp_x = smoothed_cp_x[:min_length]
        smoothed_com_y = smoothed_com_y[:min_length]
        smoothed_zmp_y = smoothed_zmp_y[:min_length]
        smoothed_cmp_y = smoothed_cmp_y[:min_length]
        smoothed_cp_y = smoothed_cp_y[:min_length]

        plt.figure(figsize=(10, 8))

        plt.subplot(2, 1, 1)
        plt.plot(smoothed_time, smoothed_com_x, label='Smoothed CoM x')
        plt.plot(smoothed_time, smoothed_zmp_x, label='Smoothed ZMP x')
        plt.plot(smoothed_time, smoothed_cmp_x, label='Smoothed CMP x')
        plt.plot(smoothed_time, smoothed_cp_x, label='Smoothed CP x')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('x position (m)')
        plt.title('Smoothed X and Y Components of Ground Reference Points and CoM over Time')
        plt.ylim(-0.8, 0.8)  # Adjust y-limits for better visibility
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(smoothed_time, smoothed_com_y, label='Smoothed CoM y')
        plt.plot(smoothed_time, smoothed_zmp_y, label='Smoothed ZMP y')
        plt.plot(smoothed_time, smoothed_cmp_y, label='Smoothed CMP y')
        plt.plot(smoothed_time, smoothed_cp_y, label='Smoothed CP y')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('y position (m)')
        plt.ylim(-0.3, 0.3)  # Adjust y-limits for better visibility
        plt.grid(True)

        plt.tight_layout()
        plt.show()


################################################################################
# main
################################################################################


def main(args=None):
    rclpy.init(args=args)
    env = Environment()
    try:
        stop_time = -1.0
        while rclpy.ok():
            env.update()
            # check if state machine is done
            if env.state_machine.state == "done":
                if stop_time < 0:
                    stop_time = env.simulator.simTime() + 2.0
                if env.simulator.simTime() > stop_time:
                    break

    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if DO_PLOT:
            env.plot_data()
        env.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
