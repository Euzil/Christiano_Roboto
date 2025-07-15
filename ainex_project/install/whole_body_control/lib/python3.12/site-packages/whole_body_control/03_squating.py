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
        
        # Phase tracking
        self.com_shift_started = False
        self.left_foot_lifted = False
        self.squatting_started = False
        self.right_hand_started = False
        
        # Squatting parameters
        self.amplitude = 0.05  # 0.05 m amplitude
        self.frequency = 0.5   # 0.5 Hz frequency
        self.omega = 2 * np.pi * self.frequency  # angular frequency
        self.initial_com_height = None
        
        # Right hand circular motion parameters
        self.circle_center = np.array([0.4, -0.2, 1.1])  # center position
        self.circle_radius = 0.2  # 0.2 m radius
        self.circle_frequency = 0.1  # 0.1 Hz frequency
        self.circle_omega = 2 * np.pi * self.circle_frequency  # angular frequency

        # Simulation time limit
        self.max_simulation_time = 15.0  # Stop after 15 seconds

        # Data collection for plotting
        self.time_data = []
        self.com_ref_pos = []
        self.com_ref_vel = []
        self.com_ref_acc = []
        self.com_tsid_pos = []
        self.com_tsid_vel = []
        self.com_tsid_acc = []
        self.com_pybullet_pos = []
        self.com_pybullet_vel = []
        self.com_pybullet_acc = []

    def update(self):
        # elapsed time
        t = self.simulator.simTime()
        
        # Check if simulation should stop
        if t >= self.max_simulation_time:
            print(f"Simulation completed after {t:.2f} seconds. Stopping...")
            return False  # Signal to stop simulation
        
        # One-leg stand behavior
        self._update_sequence(t)

        # update the simulator and the robot
        self.simulator.step()
        self.simulator.debug()
        self.robot.update()

        # update TSID controller
        tau_sol, _ = self.tsid_wrapper.update(
            self.robot.q(), self.robot.v(), t)

        # command to the robot
        self.robot.setActuatedJointTorques(tau_sol)

        # Collect COM data for plotting
        self._collect_com_data(t)

        # publish to ros
        if t - self.t_publish > 1./30.:
            self.t_publish = t
            # get current BASE Pose
            T_b_w, _ = self.tsid_wrapper.baseState()
            self.robot.publish(T_b_w, tau_sol)

        return True  # Continue simulation

    def _collect_com_data(self, t):
        """Collect COM data from all three sources for plotting"""
        self.time_data.append(t)
        
        # Get COM reference from TSID
        com_ref = self.tsid_wrapper.comReference()
        self.com_ref_pos.append(com_ref.pos().copy())
        self.com_ref_vel.append(com_ref.vel().copy())
        self.com_ref_acc.append(com_ref.acc().copy())
        
        # Get COM computed by TSID
        com_tsid = self.tsid_wrapper.comState()
        self.com_tsid_pos.append(com_tsid.pos().copy())
        self.com_tsid_vel.append(com_tsid.vel().copy())
        self.com_tsid_acc.append(com_tsid.acc().copy())
        
        # Get COM from PyBullet (simulator)
        com_pybullet_pos = self.robot.baseCoMPosition()
        com_pybullet_vel = self.robot.baseCoMVelocity()
        # For acceleration, we can compute it from velocity changes
        if len(self.com_pybullet_vel) > 0:
            dt = t - self.time_data[-2] if len(self.time_data) > 1 else 0.001
            com_pybullet_acc = (com_pybullet_vel - self.com_pybullet_vel[-1]) / dt
        else:
            com_pybullet_acc = np.zeros(3)
            
        self.com_pybullet_pos.append(com_pybullet_pos.copy())
        self.com_pybullet_vel.append(com_pybullet_vel.copy())
        self.com_pybullet_acc.append(com_pybullet_acc.copy())

    def plot_com_comparison(self):
        """Create 3x3 plots comparing COM reference, TSID, and PyBullet"""
        if not DO_PLOT or len(self.time_data) == 0:
            return
            
        # Convert lists to numpy arrays for easier indexing
        time = np.array(self.time_data)
        com_ref_pos = np.array(self.com_ref_pos)
        com_ref_vel = np.array(self.com_ref_vel)
        com_ref_acc = np.array(self.com_ref_acc)
        com_tsid_pos = np.array(self.com_tsid_pos)
        com_tsid_vel = np.array(self.com_tsid_vel)
        com_tsid_acc = np.array(self.com_tsid_acc)
        com_pybullet_pos = np.array(self.com_pybullet_pos)
        com_pybullet_vel = np.array(self.com_pybullet_vel)
        com_pybullet_acc = np.array(self.com_pybullet_acc)
        
        # Create 3x3 subplot
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('COM Comparison: Reference vs TSID vs PyBullet', fontsize=16)
        
        # Labels for axes
        coord_labels = ['X', 'Y', 'Z']
        plot_types = ['Position', 'Velocity', 'Acceleration']
        
        # Plot each combination
        for row, plot_type in enumerate(plot_types):
            for col, coord in enumerate(coord_labels):
                ax = axes[row, col]
                
                if plot_type == 'Position':
                    ax.plot(time, com_ref_pos[:, col], 'r-', label='Reference', linewidth=2)
                    ax.plot(time, com_tsid_pos[:, col], 'b--', label='TSID', linewidth=2)
                    ax.plot(time, com_pybullet_pos[:, col], 'g:', label='PyBullet', linewidth=2)
                    ax.set_ylabel(f'{coord} Position [m]')
                elif plot_type == 'Velocity':
                    ax.plot(time, com_ref_vel[:, col], 'r-', label='Reference', linewidth=2)
                    ax.plot(time, com_tsid_vel[:, col], 'b--', label='TSID', linewidth=2)
                    ax.plot(time, com_pybullet_vel[:, col], 'g:', label='PyBullet', linewidth=2)
                    ax.set_ylabel(f'{coord} Velocity [m/s]')
                else:  # Acceleration
                    ax.plot(time, com_ref_acc[:, col], 'r-', label='Reference', linewidth=2)
                    ax.plot(time, com_tsid_acc[:, col], 'b--', label='TSID', linewidth=2)
                    ax.plot(time, com_pybullet_acc[:, col], 'g:', label='PyBullet', linewidth=2)
                    ax.set_ylim(-0.8, 0.8)  # Set y-limits for acceleration
                    ax.set_ylabel(f'{coord} Acceleration [m/s²]')
                
                ax.set_xlabel('Time [s]')
                ax.set_title(f'{plot_type} - {coord} Component')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        plt.tight_layout()
        plt.show()

    def _update_sequence(self, t):
        # Phase 1: Shift COM to right foot position (at start)
        if not self.com_shift_started:
            # Get current COM position from TSID
            p_com_current = self.tsid_wrapper.comState().pos()
            
            # Store initial COM height for squatting reference
            self.initial_com_height = p_com_current[2]
            
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
            
            # Create new position 0.3m higher and flatten rotation matrix
            lf_new_pos = np.concatenate([
                lf_current_pos + np.array([0, 0, 0.3]),  # New translation
                lf_current_rot.flatten()                # Flattened rotation matrix
            ])
            
            self.tsid_wrapper.set_LF_pose_ref(lf_new_pos)
            
            self.left_foot_lifted = True
        
        # Phase 3: After 4 seconds, start squatting with sinusoidal motion
        if t >= 4.0:
            if not self.squatting_started:
                self.squatting_started = True
                print("Starting squatting motion")
            
            # Get current COM reference position
            p_com_current = self.tsid_wrapper.comState().pos()
            
            # Compute sinusoidal height variation
            t_squat = t - 4.0  # Time since squatting started
            
            # Position: z = z0 + A * sin(ωt)
            z_offset = self.amplitude * np.sin(self.omega * t_squat)
            
            # Velocity: dz/dt = A * ω * cos(ωt)
            z_vel = self.amplitude * self.omega * np.cos(self.omega * t_squat)
            
            # Acceleration: d²z/dt² = -A * ω² * sin(ωt)
            z_acc = -self.amplitude * self.omega**2 * np.sin(self.omega * t_squat)
            
            # Set new COM reference with derivatives
            p_com_new = np.array([p_com_current[0], p_com_current[1], self.initial_com_height + z_offset])
            v_com_new = np.array([0, 0, z_vel])
            a_com_new = np.array([0, 0, z_acc])
            
            self.tsid_wrapper.setComRefState(p_com_new, v_com_new, a_com_new)
        
        # Phase 4: After 8 seconds, start moving right hand in a circle
        if t >= 8.0:
            if not self.right_hand_started:
                # Add right hand motion task
                self.tsid_wrapper.add_motion_RH()
                
                # Set orientation task gains to zero before activating the right-hand task
                self.tsid_wrapper.rightHandTask.setKp(100*np.array([1,1,1,0,0,0]))
                self.tsid_wrapper.rightHandTask.setKd(2.0*np.sqrt(100)*np.array([1,1,1,0,0,0]))
                
                self.right_hand_started = True
                print("Starting right hand circular motion")
                
                # Optional: Add trajectory visualization
                # Generate circle points for visualization
                theta_points = np.linspace(0, 2*np.pi, 50)
                circle_x = np.full_like(theta_points, self.circle_center[0])
                circle_y = self.circle_center[1] + self.circle_radius * np.cos(theta_points)
                circle_z = self.circle_center[2] + self.circle_radius * np.sin(theta_points)
                self.simulator.addGlobalDebugTrajectory(circle_x, circle_y, circle_z)
            
            # Compute circular motion in Y-Z plane
            t_circle = t - 8.0  # Time since circular motion started
            theta = self.circle_omega * t_circle
            
            # Position: circular motion in Y-Z plane (X constant)
            rh_pos = np.array([
                self.circle_center[0],  # X constant
                self.circle_center[1] + self.circle_radius * np.cos(theta),  # Y varies
                self.circle_center[2] + self.circle_radius * np.sin(theta)   # Z varies
            ])
            
            # Velocity: derivatives of position
            rh_vel = np.array([
                0,  # dX/dt = 0
                -self.circle_radius * self.circle_omega * np.sin(theta),  # dY/dt
                self.circle_radius * self.circle_omega * np.cos(theta)    # dZ/dt
            ])
            
            # Acceleration: derivatives of velocity  
            rh_acc = np.array([
                0,  # d²X/dt² = 0
                -self.circle_radius * self.circle_omega**2 * np.cos(theta),  # d²Y/dt²
                -self.circle_radius * self.circle_omega**2 * np.sin(theta)   # d²Z/dt²
            ])
            
            # Set right hand position reference with derivatives
            self.tsid_wrapper.set_RH_pos_ref(rh_pos, rh_vel, rh_acc)

################################################################################
# main
################################################################################


def main(args=None):
    rclpy.init(args=args)
    env = Environment()
    try:
        while rclpy.ok():
            if not env.update():  # Check return value
                break  # Exit loop when simulation is complete

    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Generate plots before shutting down
        if DO_PLOT:
            print("Generating COM comparison plots...")
            env.plot_com_comparison()
        env.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
