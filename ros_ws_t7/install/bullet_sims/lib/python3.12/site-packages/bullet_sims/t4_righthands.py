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
from bullet_sims.tsid_wrapper import TSIDWrapper
import bullet_sims.config as conf

# ROS
import rclpy
from rclpy.node import Node
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
    """
    Talos robot class that derives from Robot
    - Loads robot with floating base (useFixedBase=False)
    - No pin.RobotWrapper needed since TSID has its own robot wrapper
    - Includes ROS publisher for joint state
    - Includes TF-Broadcaster for T_b_w transformation
    """
    def __init__(self, simulator, urdf, model, q=None, verbose=True, useFixedBase=False):
        # Call base class constructor with floating base
        super().__init__(
            simulator=simulator,
            filename=urdf,
            model=model,
            basePosition=np.array([0, 0, 1.09]),
            baseQuationerion=np.array([0, 0, 0, 1]),
            q=q,
            useFixedBase=useFixedBase,  # False for floating base as required
            verbose=verbose
        )
        
        # Create ROS2 node for publishing
        self.node = rclpy.create_node('talos_robot_publisher')
        
        # Add ROS publisher for robot's joint state
        self.joint_state_pub = self.node.create_publisher(JointState, '/joint_states', 10)
        
        # Add TF-Broadcaster for T_b_w transformation between "base_link" and "world"
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self.node)
        
        # Initialize joint state message
        self.joint_state_msg = JointState()
        self.joint_state_msg.header.frame_id = "world"
        
        # Set joint names from TSID model (skip universe and root_joint)
        if hasattr(model, 'names') and len(model.names) > 2:
            joint_names = [name for name in model.names[2:]]
            self.joint_state_msg.name = joint_names[:conf.na]
        else:
            self.joint_state_msg.name = [f"joint_{i}" for i in range(conf.na)]

    def update(self):
        # Update base class only - no pinocchio since TSID has its own wrapper
        super().update()

    def publish(self, T_b_w):
        """
        Publish function called at 30 Hz
        - Publishes robot's joint state
        - Broadcasts T_b_w transformation from TSID baseState()
        """
        current_time = self.node.get_clock().now()
        
        # Publish joint state
        self.joint_state_msg.header.stamp = current_time.to_msg()
        
        # Get current robot state
        q_current = np.array(self.q())
        v_current = np.array(self.v())
        
        # Extract joint positions and velocities (skip floating base DOFs)
        if len(q_current) > conf.na:
            joint_positions = q_current[7:7+conf.na]  # Skip 7 base DOFs
            joint_velocities = v_current[6:6+conf.na]  # Skip 6 base DOFs
        else:
            joint_positions = q_current[:conf.na]
            joint_velocities = v_current[:conf.na]
        
        self.joint_state_msg.position = joint_positions.tolist()
        self.joint_state_msg.velocity = joint_velocities.tolist()
        
        # Add effort if available
        if hasattr(self, 'tau') and self.tau is not None:
            tau_array = np.array(self.tau)
            if len(tau_array) >= conf.na:
                self.joint_state_msg.effort = tau_array[:conf.na].tolist()
        
        self.joint_state_pub.publish(self.joint_state_msg)
        
        # Broadcast T_b_w transformation between "base_link" and "world"
        self._broadcast_base_transform(T_b_w, current_time)

    def _broadcast_base_transform(self, T_b_w, timestamp):
        """Broadcast T_b_w transformation from TSID baseState()"""
        if T_b_w is not None:
            transform_msg = TransformStamped()
            transform_msg.header.stamp = timestamp.to_msg()
            transform_msg.header.frame_id = "world"
            transform_msg.child_frame_id = "base_link"
            
            # Extract translation and rotation from SE3 object
            if hasattr(T_b_w, 'translation') and hasattr(T_b_w, 'rotation'):
                # SE3 object from TSID
                transform_msg.transform.translation.x = float(T_b_w.translation[0])
                transform_msg.transform.translation.y = float(T_b_w.translation[1])
                transform_msg.transform.translation.z = float(T_b_w.translation[2])
                
                # Convert rotation matrix to quaternion
                quat = pin.Quaternion(T_b_w.rotation)
                transform_msg.transform.rotation.x = float(quat.x)
                transform_msg.transform.rotation.y = float(quat.y)
                transform_msg.transform.rotation.z = float(quat.z)
                transform_msg.transform.rotation.w = float(quat.w)
            else:
                # Fallback: identity transform
                transform_msg.transform.translation.z = 0.9
                transform_msg.transform.rotation.w = 1.0
            
            self.tf_broadcaster.sendTransform(transform_msg)

################################################################################
# COM Data Logger
################################################################################

class COMDataLogger:
    """
    Class to log COM data for plotting
    """
    def __init__(self):
        # Time arrays
        self.times = []
        
        # COM Reference data (from TSIDWrapper.comReference())
        self.com_ref_pos = []
        self.com_ref_vel = []
        self.com_ref_acc = []
        
        # COM TSID data (from TSIDWrapper.comState())
        self.com_tsid_pos = []
        self.com_tsid_vel = []
        self.com_tsid_acc = []
        
        # COM PyBullet data (computed from robot base position/velocity)
        self.com_pybullet_pos = []
        self.com_pybullet_vel = []
        self.com_pybullet_acc = []
        
        # Previous velocity for acceleration calculation
        self.prev_pybullet_vel = None
        self.prev_time = None
    
    def log_data(self, t, tsid_wrapper, robot):
        """
        Log COM data at current time step
        """
        self.times.append(t)
        
        try:
            # Get COM reference from TSID
            com_ref = tsid_wrapper.comReference()
            if hasattr(com_ref, 'pos'):
                self.com_ref_pos.append(com_ref.pos().copy())
            else:
                self.com_ref_pos.append(np.array([0, 0, 0]))
            
            if hasattr(com_ref, 'vel'):
                self.com_ref_vel.append(com_ref.vel().copy())
            else:
                self.com_ref_vel.append(np.array([0, 0, 0]))
            
            if hasattr(com_ref, 'acc'):
                self.com_ref_acc.append(com_ref.acc().copy())
            else:
                self.com_ref_acc.append(np.array([0, 0, 0]))
        except:
            # Fallback if reference not available
            self.com_ref_pos.append(np.array([0, 0, 0]))
            self.com_ref_vel.append(np.array([0, 0, 0]))
            self.com_ref_acc.append(np.array([0, 0, 0]))
        
        try:
            # Get COM state from TSID
            com_state = tsid_wrapper.comState()
            if hasattr(com_state, 'pos'):
                self.com_tsid_pos.append(com_state.pos().copy())
            else:
                self.com_tsid_pos.append(np.array([0, 0, 0]))
            
            if hasattr(com_state, 'vel'):
                self.com_tsid_vel.append(com_state.vel().copy())
            else:
                self.com_tsid_vel.append(np.array([0, 0, 0]))
            
            if hasattr(com_state, 'acc'):
                self.com_tsid_acc.append(com_state.acc().copy())
            else:
                self.com_tsid_acc.append(np.array([0, 0, 0]))
        except:
            # Fallback if TSID state not available
            self.com_tsid_pos.append(np.array([0, 0, 0]))
            self.com_tsid_vel.append(np.array([0, 0, 0]))
            self.com_tsid_acc.append(np.array([0, 0, 0]))
        
        try:
            # Get COM from PyBullet (using base position as approximation)
            base_pos = np.array(robot.baseWorldPosition())
            base_vel = np.array(robot.baseWorldLinearVelocity())
            
            self.com_pybullet_pos.append(base_pos.copy())
            self.com_pybullet_vel.append(base_vel.copy())
            
            # Calculate acceleration from velocity difference
            if self.prev_pybullet_vel is not None and self.prev_time is not None:
                dt = t - self.prev_time
                if dt > 1e-6:  # Avoid division by very small numbers
                    acc = (base_vel - self.prev_pybullet_vel) / dt
                    self.com_pybullet_acc.append(acc.copy())
                else:
                    self.com_pybullet_acc.append(np.array([0, 0, 0]))
            else:
                self.com_pybullet_acc.append(np.array([0, 0, 0]))
            
            # Store for next iteration
            self.prev_pybullet_vel = base_vel.copy()
            self.prev_time = t
            
        except Exception as e:
            print(f"Warning: Could not get PyBullet COM data: {e}")
            self.com_pybullet_pos.append(np.array([0, 0, 0]))
            self.com_pybullet_vel.append(np.array([0, 0, 0]))
            self.com_pybullet_acc.append(np.array([0, 0, 0]))
    
    def plot_results(self):
        """
        Plot COM position, velocity, and acceleration comparison
        """
        if len(self.times) == 0:
            print("No data to plot!")
            return
        
        # Convert lists to numpy arrays for easier plotting
        times = np.array(self.times)
        
        # Convert position, velocity, acceleration lists to arrays
        com_ref_pos = np.array(self.com_ref_pos)
        com_ref_vel = np.array(self.com_ref_vel)
        com_ref_acc = np.array(self.com_ref_acc)
        
        com_tsid_pos = np.array(self.com_tsid_pos)
        com_tsid_vel = np.array(self.com_tsid_vel)
        com_tsid_acc = np.array(self.com_tsid_acc)
        
        com_pybullet_pos = np.array(self.com_pybullet_pos)
        com_pybullet_vel = np.array(self.com_pybullet_vel)
        com_pybullet_acc = np.array(self.com_pybullet_acc)
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('COM Tracking Comparison: Reference vs TSID vs PyBullet', fontsize=16)
        
        # Position plots
        axes[0, 0].plot(times, com_ref_pos[:, 0], 'r-', label='Reference', linewidth=2)
        axes[0, 0].plot(times, com_tsid_pos[:, 0], 'g--', label='TSID', linewidth=2)
        axes[0, 0].plot(times, com_pybullet_pos[:, 0], 'b:', label='PyBullet', linewidth=2)
        axes[0, 0].set_title('COM Position X')
        axes[0, 0].set_ylabel('Position [m]')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(times, com_ref_pos[:, 1], 'r-', label='Reference', linewidth=2)
        axes[0, 1].plot(times, com_tsid_pos[:, 1], 'g--', label='TSID', linewidth=2)
        axes[0, 1].plot(times, com_pybullet_pos[:, 1], 'b:', label='PyBullet', linewidth=2)
        axes[0, 1].set_title('COM Position Y')
        axes[0, 1].set_ylabel('Position [m]')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[0, 2].plot(times, com_ref_pos[:, 2], 'r-', label='Reference', linewidth=2)
        axes[0, 2].plot(times, com_tsid_pos[:, 2], 'g--', label='TSID', linewidth=2)
        axes[0, 2].plot(times, com_pybullet_pos[:, 2], 'b:', label='PyBullet', linewidth=2)
        axes[0, 2].set_title('COM Position Z')
        axes[0, 2].set_ylabel('Position [m]')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Velocity plots
        axes[1, 0].plot(times, com_ref_vel[:, 0], 'r-', label='Reference', linewidth=2)
        axes[1, 0].plot(times, com_tsid_vel[:, 0], 'g--', label='TSID', linewidth=2)
        axes[1, 0].plot(times, com_pybullet_vel[:, 0], 'b:', label='PyBullet', linewidth=2)
        axes[1, 0].set_title('COM Velocity X')
        axes[1, 0].set_ylabel('Velocity [m/s]')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(times, com_ref_vel[:, 1], 'r-', label='Reference', linewidth=2)
        axes[1, 1].plot(times, com_tsid_vel[:, 1], 'g--', label='TSID', linewidth=2)
        axes[1, 1].plot(times, com_pybullet_vel[:, 1], 'b:', label='PyBullet', linewidth=2)
        axes[1, 1].set_title('COM Velocity Y')
        axes[1, 1].set_ylabel('Velocity [m/s]')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        axes[1, 2].plot(times, com_ref_vel[:, 2], 'r-', label='Reference', linewidth=2)
        axes[1, 2].plot(times, com_tsid_vel[:, 2], 'g--', label='TSID', linewidth=2)
        axes[1, 2].plot(times, com_pybullet_vel[:, 2], 'b:', label='PyBullet', linewidth=2)
        axes[1, 2].set_title('COM Velocity Z')
        axes[1, 2].set_ylabel('Velocity [m/s]')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        # Acceleration plots
        axes[2, 0].plot(times, com_ref_acc[:, 0], 'r-', label='Reference', linewidth=2)
        axes[2, 0].plot(times, com_tsid_acc[:, 0], 'g--', label='TSID', linewidth=2)
        axes[2, 0].plot(times, com_pybullet_acc[:, 0], 'b:', label='PyBullet', linewidth=2)
        axes[2, 0].set_title('COM Acceleration X')
        axes[2, 0].set_ylabel('Acceleration [m/s²]')
        axes[2, 0].set_xlabel('Time [s]')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        axes[2, 1].plot(times, com_ref_acc[:, 1], 'r-', label='Reference', linewidth=2)
        axes[2, 1].plot(times, com_tsid_acc[:, 1], 'g--', label='TSID', linewidth=2)
        axes[2, 1].plot(times, com_pybullet_acc[:, 1], 'b:', label='PyBullet', linewidth=2)
        axes[2, 1].set_title('COM Acceleration Y')
        axes[2, 1].set_ylabel('Acceleration [m/s²]')
        axes[2, 1].set_xlabel('Time [s]')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        axes[2, 2].plot(times, com_ref_acc[:, 2], 'r-', label='Reference', linewidth=2)
        axes[2, 2].plot(times, com_tsid_acc[:, 2], 'g--', label='TSID', linewidth=2)
        axes[2, 2].plot(times, com_pybullet_acc[:, 2], 'b:', label='PyBullet', linewidth=2)
        axes[2, 2].set_title('COM Acceleration Z')
        axes[2, 2].set_ylabel('Acceleration [m/s²]')
        axes[2, 2].set_xlabel('Time [s]')
        axes[2, 2].legend()
        axes[2, 2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print some statistics
        print("\n=== COM Tracking Statistics ===")
        print(f"Simulation duration: {times[-1]:.2f} seconds")
        print(f"Data points: {len(times)}")
        
        # Calculate RMS errors between TSID and Reference
        pos_error_tsid = np.sqrt(np.mean((com_tsid_pos - com_ref_pos)**2, axis=0))
        vel_error_tsid = np.sqrt(np.mean((com_tsid_vel - com_ref_vel)**2, axis=0))
        acc_error_tsid = np.sqrt(np.mean((com_tsid_acc - com_ref_acc)**2, axis=0))
        
        print(f"\nTSID vs Reference RMS Errors:")
        print(f"Position [m]: X={pos_error_tsid[0]:.4f}, Y={pos_error_tsid[1]:.4f}, Z={pos_error_tsid[2]:.4f}")
        print(f"Velocity [m/s]: X={vel_error_tsid[0]:.4f}, Y={vel_error_tsid[1]:.4f}, Z={vel_error_tsid[2]:.4f}")
        print(f"Acceleration [m/s²]: X={acc_error_tsid[0]:.4f}, Y={acc_error_tsid[1]:.4f}, Z={acc_error_tsid[2]:.4f}")
        
        # Calculate RMS errors between PyBullet and TSID
        pos_error_pybullet = np.sqrt(np.mean((com_pybullet_pos - com_tsid_pos)**2, axis=0))
        vel_error_pybullet = np.sqrt(np.mean((com_pybullet_vel - com_tsid_vel)**2, axis=0))
        acc_error_pybullet = np.sqrt(np.mean((com_pybullet_acc - com_tsid_acc)**2, axis=0))
        
        print(f"\nPyBullet vs TSID RMS Errors:")
        print(f"Position [m]: X={pos_error_pybullet[0]:.4f}, Y={pos_error_pybullet[1]:.4f}, Z={pos_error_pybullet[2]:.4f}")
        print(f"Velocity [m/s]: X={vel_error_pybullet[0]:.4f}, Y={vel_error_pybullet[1]:.4f}, Z={vel_error_pybullet[2]:.4f}")
        print(f"Acceleration [m/s²]: X={acc_error_pybullet[0]:.4f}, Y={acc_error_pybullet[1]:.4f}, Z={acc_error_pybullet[2]:.4f}")

################################################################################
# main
################################################################################

def main():
    rclpy.init()
    node = rclpy.create_node('tutorial_4_squatting_node')
    
    # Initialize COM data logger
    com_logger = COMDataLogger()
    
    try:
        print("=== Tutorial 4: Single Leg Squatting with Hand Motion ===")
        
        # Step 1: Instantiate the TSIDWrapper
        print("Step 1: Instantiating TSIDWrapper...")
        tsid_wrapper = TSIDWrapper(conf)
        
        # Hand tasks and methods should already be implemented in TSIDWrapper
        print("Using TSIDWrapper with hand motion capabilities...")
        
        # Step 2: Instantiate the simulator PybulletWrapper
        print("Step 2: Instantiating PybulletWrapper...")
        simulator = PybulletWrapper()
        
        # Step 3: Instantiate Talos and give it the model from TSIDWrapper and conf.q_home
        print("Step 3: Instantiating Talos with model from TSIDWrapper and conf.q_home...")
        model = tsid_wrapper.robot.model()  # Get model from TSIDWrapper
        
        robot = Talos(
            simulator=simulator,
            urdf=conf.urdf,
            model=model,                    # Model from TSIDWrapper
            q=conf.q_home,                  # Joint configuration from conf.q_home
            verbose=True,
            useFixedBase=False              # Floating base
        )
        
        print("Robot initialization complete.")
        
        # Brief initial settling
        robot.enablePositionControl()
        for i in range(30):
            robot.setActuatedJointPositions(conf.q_home)
            simulator.step()
            robot.update()
        
        # Switch to torque control for TSID
        robot.enableTorqueControl()
        
        # Hand motion parameters (using original specification values)
        circle_center = np.array([0.4, -0.2, 1.1])  # Original center as specified
        circle_radius = 0.2       # Original radius: 0.2 m 
        hand_frequency = 0.1      # Original frequency: 0.1 Hz
        hand_omega = 2 * np.pi * hand_frequency  # Angular frequency for hand
        
        # Add debug trajectory visualization for the hand circle
        print("Adding debug trajectory visualization...")
        circle_points_x = []
        circle_points_y = []
        circle_points_z = []
        num_points = 50
        for i in range(num_points + 1):
            angle = 2 * np.pi * i / num_points
            x = circle_center[0]
            y = circle_center[1] + circle_radius * np.cos(angle)
            z = circle_center[2] + circle_radius * np.sin(angle)
            circle_points_x.append(x)
            circle_points_y.append(y) 
            circle_points_z.append(z)
        
        # Visualize the circular trajectory in pybullet
        simulator.addGlobalDebugTrajectory(circle_points_x, circle_points_y, circle_points_z)
        
        # Variables for 30 Hz publishing and behavior control
        t_publish = 0.0
        publish_rate = 30.0  # 30 Hz
        
        # Data logging rate (can be different from publish rate)
        t_log = 0.0
        log_rate = 100.0  # 100 Hz for detailed analysis
        
        # Motion parameters
        stabilization_time = 3.0  # 3 seconds for initial stabilization
        com_shift_time = 5.0      # 2 seconds for COM shifting (3+2=5)
        foot_lift_time = 7.0      # 2 seconds for foot lifting (5+2=7)
        squatting_start_time = 7.0  # Start squatting after single leg stance
        hand_motion_start_time = 8.0  # Start hand motion after 8 seconds
        amplitude = 0.05          # 0.05 m amplitude for squatting
        frequency = 0.5           # 0.5 Hz frequency for squatting
        omega = 2 * np.pi * frequency  # Angular frequency for squatting
        
        # Simulation duration for analysis
        simulation_duration = 20.0  # Run for 20 seconds to get good data
        
        # Behavior control flags
        stabilized = False
        com_shifted = False
        left_foot_lifted = False
        squatting_started = False
        hand_motion_started = False
        initial_com_height = None
        
        print("Starting simulation...")
        print("Phase 1: Initial stabilization (3 seconds)...")
        
        while rclpy.ok() and simulator.simTime() < simulation_duration:
            # Get simulation time
            t = simulator.simTime()
            
            # Update the simulator
            simulator.step()
            
            # Update the robot
            robot.update()
            
            # Get current state
            q_current = np.ascontiguousarray(robot.q(), dtype=np.float64)
            v_current = np.ascontiguousarray(robot.v(), dtype=np.float64)
            
            # Behavior logic: Five phases
            # Phase 1: Initial stabilization (0-3 seconds)
            if not stabilized and t < stabilization_time:
                # Let robot stabilize in home position
                pass
            
            # Phase 2: Shift COM to right foot (3-5 seconds)
            elif not com_shifted and t >= stabilization_time and t < com_shift_time:
                if not stabilized:
                    print("Phase 2: COM shifting to right foot (2 seconds)...")
                    stabilized = True
                
                # Get current COM position
                com_current = tsid_wrapper.comState().pos()
                
                # Get right foot position
                rf_placement = tsid_wrapper.get_placement_RF()
                rf_position = rf_placement.translation
                
                # Create new COM reference: XY from right foot, keep current Z
                p_com_new = np.array([rf_position[0], rf_position[1], com_current[2]])
                
                # Set new COM reference
                tsid_wrapper.setComRefState(p_com_new)
                
            # Phase 3: Remove left foot contact and lift it (5-7 seconds)
            elif not left_foot_lifted and t >= com_shift_time and t < foot_lift_time:
                if not com_shifted:
                    print("Phase 3: COM shifted, removing left foot contact and lifting...")
                    com_shifted = True
                
                # Remove left foot contact (only do this once)
                if not left_foot_lifted:
                    tsid_wrapper.remove_contact_LF()
                    
                    # Get current left foot position
                    lf_placement = tsid_wrapper.get_placement_LF()
                    
                    # Create new left foot reference: 0.3m higher
                    lf_position_new = lf_placement.translation.copy()
                    lf_position_new[2] += 0.3  # Lift 0.3m off the ground
                    
                    # Set new left foot pose reference
                    lf_pose_new = pin.SE3(lf_placement.rotation, lf_position_new)
                    tsid_wrapper.set_LF_pose_ref(lf_pose_new)
                    
                    left_foot_lifted = True
                    print("Left foot lifted! Robot now standing on right leg.")
                    
                    # Store initial COM height for squatting reference
                    com_current = tsid_wrapper.comState().pos()
                    initial_com_height = com_current[2]
                    print(f"Initial COM height for squatting: {initial_com_height:.3f} m")
            
            # Phase 4: Single leg squatting motion (7-8 seconds, before hand motion starts)
            elif t >= squatting_start_time and t < hand_motion_start_time and left_foot_lifted:
                if not squatting_started:
                    print("Phase 4: Starting single leg squatting motion...")
                    squatting_started = True
                    
                    # Ensure we have initial COM height
                    if initial_com_height is None:
                        com_current = tsid_wrapper.comState().pos()
                        initial_com_height = com_current[2]
                
                if squatting_started and initial_com_height is not None:
                    # Time since squatting started
                    t_squat = t - squatting_start_time
                    
                    # Sinusoidal height variation
                    # Position: z(t) = z0 + A * sin(ωt)
                    height_offset = amplitude * np.sin(omega * t_squat)
                    
                    # Velocity: v_z(t) = A * ω * cos(ωt)
                    height_velocity = amplitude * omega * np.cos(omega * t_squat)
                    
                    # Acceleration: a_z(t) = -A * ω² * sin(ωt)
                    height_acceleration = -amplitude * omega * omega * np.sin(omega * t_squat)
                    
                    # Get right foot position to maintain COM over support foot
                    rf_placement = tsid_wrapper.get_placement_RF()
                    rf_position = rf_placement.translation
                    
                    # Create COM reference with sinusoidal height, maintaining XY over right foot
                    p_com = np.array([
                        rf_position[0],  # Keep over right foot X
                        rf_position[1],  # Keep over right foot Y
                        initial_com_height + height_offset  # Sinusoidal Z
                    ])
                    
                    # COM velocity (only Z component varies)
                    v_com = np.array([0.0, 0.0, height_velocity])
                    
                    # COM acceleration (only Z component varies)
                    a_com = np.array([0.0, 0.0, height_acceleration])
                    
                    # Set COM reference with position, velocity, and acceleration
                    tsid_wrapper.setComRefState(p_com, v_com, a_com)
                    
                    # Debug output every second
                    if int(t_squat) != int(t_squat - 0.001):
                        print(f"Single leg squatting: t={t_squat:.1f}s, height_offset={height_offset:.3f}m, "
                              f"vel={height_velocity:.3f}m/s, acc={height_acceleration:.3f}m/s²")
            
            # Phase 5: Combined squatting + right hand circular motion (after 8 seconds)
            elif t >= hand_motion_start_time and left_foot_lifted:
                # Continue squatting motion
                if squatting_started and initial_com_height is not None:
                    # Time since squatting started
                    t_squat = t - squatting_start_time
                    
                    # Sinusoidal height variation
                    height_offset = amplitude * np.sin(omega * t_squat)
                    height_velocity = amplitude * omega * np.cos(omega * t_squat)
                    height_acceleration = -amplitude * omega * omega * np.sin(omega * t_squat)
                    
                    # Get right foot position to maintain COM over support foot
                    rf_placement = tsid_wrapper.get_placement_RF()
                    rf_position = rf_placement.translation
                    
                    # Create COM reference with sinusoidal height, maintaining XY over right foot
                    p_com = np.array([
                        rf_position[0],  # Keep over right foot X
                        rf_position[1],  # Keep over right foot Y
                        initial_com_height + height_offset  # Sinusoidal Z
                    ])
                    
                    # COM velocity and acceleration
                    v_com = np.array([0.0, 0.0, height_velocity])
                    a_com = np.array([0.0, 0.0, height_acceleration])
                    
                    # Set COM reference
                    tsid_wrapper.setComRefState(p_com, v_com, a_com)
                
                # Hand motion logic
                if not hand_motion_started:
                    print("Phase 5: Starting right hand circular motion (while continuing squatting)...")
                    
                    # Set gains BEFORE activating the task (as per original instructions)
                    tsid_wrapper.rightHandTask.setKp(100 * np.array([1, 1, 1, 0, 0, 0]))
                    tsid_wrapper.rightHandTask.setKd(2.0 * np.sqrt(100) * np.array([1, 1, 1, 0, 0, 0]))
                    
                    # Use TSIDWrapper's built-in method to add right hand task
                    tsid_wrapper.add_motion_RH(transition_time=0.0)
                    
                    hand_motion_started = True
                    print("Right hand task activated!")
                
                if hand_motion_started and tsid_wrapper.motion_RH_active:
                    # Time since hand motion started
                    t_hand = t - hand_motion_start_time
                    
                    # Circular motion in Y-Z plane as specified:
                    # p = c + [0, r*cos(ωt), r*sin(ωt)]^T
                    hand_position = np.array([
                        circle_center[0],  # X stays constant (0.4)
                        circle_center[1] + circle_radius * np.cos(hand_omega * t_hand),  # Y: -0.2 + 0.2*cos(ωt)
                        circle_center[2] + circle_radius * np.sin(hand_omega * t_hand)   # Z: 1.1 + 0.2*sin(ωt)
                    ])
                    
                    # Calculate velocity for circular motion
                    hand_velocity = np.array([
                        0.0,  # X velocity = 0
                        -circle_radius * hand_omega * np.sin(hand_omega * t_hand),  # Y velocity
                        circle_radius * hand_omega * np.cos(hand_omega * t_hand)   # Z velocity
                    ])
                    
                    # Calculate acceleration for circular motion
                    hand_acceleration = np.array([
                        0.0,  # X acceleration = 0
                        -circle_radius * hand_omega * hand_omega * np.cos(hand_omega * t_hand),  # Y acceleration
                        -circle_radius * hand_omega * hand_omega * np.sin(hand_omega * t_hand)   # Z acceleration
                    ])
                    
                    # Use TSIDWrapper's set_RH_pos_ref method with position, velocity, and acceleration
                    tsid_wrapper.set_RH_pos_ref(hand_position, hand_velocity, hand_acceleration)
                    
                    # Debug output every 2 seconds for hand motion
                    if int(t_hand / 2.0) != int((t_hand - 0.001) / 2.0):
                        print(f"Right hand circular motion: t={t_hand:.1f}s, "
                              f"pos=[{hand_position[0]:.3f}, {hand_position[1]:.3f}, {hand_position[2]:.3f}]")
            
            # TSIDWrapper.update(...) solves problem and returns solution torque and acceleration
            tau_sol, acc_sol = tsid_wrapper.update(q_current, v_current, t)
            
            # Ensure tau_sol is proper array and correct size
            tau_sol = np.array(tau_sol)
            if len(tau_sol) > conf.na:
                tau_sol = tau_sol[:conf.na]
            elif len(tau_sol) < conf.na:
                tau_padded = np.zeros(conf.na)
                tau_padded[:len(tau_sol)] = tau_sol
                tau_sol = tau_padded
            
            # Store for publishing
            robot.tau = tau_sol
            
            # Feed the torque to our robot
            robot.setActuatedJointTorques(tau_sol)
            
            # Log COM data at specified rate
            if t - t_log >= 1.0 / log_rate:
                t_log = t
                com_logger.log_data(t, tsid_wrapper, robot)
            
            # Call robot's publish function at 30 Hz
            if t - t_publish >= 1.0 / publish_rate:
                t_publish = t
                
                # Get base-to-world transformation from TSID with baseState() function
                T_b_w = tsid_wrapper.baseState()
                
                # Call robot's publish function
                robot.publish(T_b_w)
            
            # ROS spin
            rclpy.spin_once(node, timeout_sec=0.001)
        
        print("Simulation completed!")
        print("Generating COM tracking plots...")
        
        # Generate plots if data was collected
        if DO_PLOT and len(com_logger.times) > 0:
            com_logger.plot_results()
        else:
            print("No plotting data available or plotting disabled.")
    
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
        if DO_PLOT and len(com_logger.times) > 0:
            print("Generating plots from collected data...")
            com_logger.plot_results()
    except Exception as e:
        print(f"Error in simulation: {e}")
        import traceback
        traceback.print_exc()
        if DO_PLOT and len(com_logger.times) > 0:
            print("Generating plots from collected data...")
            com_logger.plot_results()
    finally:
        # Cleanup
        if 'simulator' in locals():
            simulator.disconnect()
        rclpy.shutdown()
        print("Simulation ended")

if __name__ == '__main__':
    main()