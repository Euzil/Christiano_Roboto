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
            basePosition=np.array([0, 0, 1.1]),
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
# main
################################################################################

def main():
    rclpy.init()
    node = rclpy.create_node('tutorial_4_one_leg_stand_node')
    
    try:
        print("=== Tutorial 4: One Leg Stand Controller ===")
        
        # Step 1: Instantiate the TSIDWrapper
        print("Step 1: Instantiating TSIDWrapper...")
        tsid_wrapper = TSIDWrapper(conf)
        
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
        
        # Variables for 30 Hz publishing and behavior control
        t_publish = 0.0
        publish_rate = 30.0  # 30 Hz
        
        # Behavior control flags and timing
        stabilization_time = 3.0  # 3 seconds for initial stabilization
        com_shift_time = 5.0      # 2 seconds for COM shifting (3+2=5)
        stabilized = False
        com_shifted = False
        left_foot_lifted = False
        
        print("Starting simulation...")
        print("Phase 1: Initial stabilization (3 seconds)...")
        
        while rclpy.ok():
            # Get simulation time
            t = simulator.simTime()
            
            # Update the simulator
            simulator.step()
            
            # Update the robot
            robot.update()
            
            # Get current state
            q_current = np.ascontiguousarray(robot.q(), dtype=np.float64)
            v_current = np.ascontiguousarray(robot.v(), dtype=np.float64)
            
            # Behavior logic: Three phases
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
                
            # Phase 3: Remove left foot contact and lift it (after 5 seconds)
            elif not com_shifted and t >= com_shift_time:
                print("Phase 3: COM shifted, removing left foot contact and lifting...")
                
                # Remove left foot contact
                tsid_wrapper.remove_contact_LF()
                
                # Get current left foot position
                lf_placement = tsid_wrapper.get_placement_LF()
                
                # Create new left foot reference: 0.3m higher
                lf_position_new = lf_placement.translation.copy()
                lf_position_new[2] += 0.3  # Lift 0.3m off the ground
                
                # Set new left foot pose reference
                lf_pose_new = pin.SE3(lf_placement.rotation, lf_position_new)
                tsid_wrapper.set_LF_pose_ref(lf_pose_new)
                
                com_shifted = True
                left_foot_lifted = True
                print("Left foot lifted! Robot now standing on right leg.")
            
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
            
            # Step 5: Call robot's publish function at 30 Hz
            if t - t_publish >= 1.0 / publish_rate:
                t_publish = t
                
                # Get base-to-world transformation from TSID with baseState() function
                T_b_w = tsid_wrapper.baseState()
                
                # Call robot's publish function
                robot.publish(T_b_w)
            
            # ROS spin
            rclpy.spin_once(node, timeout_sec=0.001)
    
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    except Exception as e:
        print(f"Error in simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'simulator' in locals():
            simulator.disconnect()
        rclpy.shutdown()
        print("Simulation ended")

if __name__ == '__main__':
    main()