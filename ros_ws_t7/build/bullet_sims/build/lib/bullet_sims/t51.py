import numpy as np
import pinocchio as pin
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot
from bullet_sims.tsid_wrapper import TSIDWrapper
import bullet_sims.config as conf
import rclpy
import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, WrenchStamped, Vector3
from std_msgs.msg import Header
import pybullet as pb
import enum

################################################################################
# State Machine for Push Forces
################################################################################

class PushState(enum.Enum):
    STANDING = 0
    PUSH_RIGHT = 1
    PUSH_LEFT = 2
    PUSH_BACK = 3
    FINISHED = 4

class PushStateMachine:
    """State machine for applying pushing forces to the robot"""
    
    def __init__(self, tperiod=5.0, tpush=1.0, fpush_magnitude=50.0):
        self.tperiod = tperiod  # Time before first push
        self.tpush = tpush      # Duration of each push
        self.fpush_magnitude = fpush_magnitude  # Force magnitude
        
        self.state = PushState.STANDING
        self.state_start_time = 0.0
        self.push_count = 0
        
        # Push force directions (in world frame)
        self.push_directions = {
            PushState.PUSH_RIGHT: np.array([0.0, -1.0, 0.0]),  # Push from right (force to left)
            PushState.PUSH_LEFT: np.array([0.0, 1.0, 0.0]),    # Push from left (force to right)
            PushState.PUSH_BACK: np.array([1.0, 0.0, 0.0])     # Push from back (force forward)
        }
        
        print(f"Push State Machine initialized:")
        print(f"  - Period before first push: {tperiod}s")
        print(f"  - Push duration: {tpush}s")
        print(f"  - Push force magnitude: {fpush_magnitude}N")
    
    def update(self, current_time):
        """Update state machine and return current push force"""
        time_in_state = current_time - self.state_start_time
        
        if self.state == PushState.STANDING:
            if time_in_state >= self.tperiod:
                self._transition_to_next_push()
            return np.zeros(3)
        
        elif self.state in [PushState.PUSH_RIGHT, PushState.PUSH_LEFT, PushState.PUSH_BACK]:
            if time_in_state >= self.tpush:
                self._transition_after_push()
                return np.zeros(3)
            else:
                # Apply push force
                direction = self.push_directions[self.state]
                return self.fpush_magnitude * direction
        
        else:  # FINISHED
            return np.zeros(3)
    
    def _transition_to_next_push(self):
        """Transition to the next push state"""
        self.push_count += 1
        self.state_start_time = self.state_start_time + (self.tperiod if self.state == PushState.STANDING else self.tpush)
        
        if self.push_count == 1:
            self.state = PushState.PUSH_RIGHT
            print(f"Starting RIGHT push at t={self.state_start_time:.2f}s")
        elif self.push_count == 2:
            self.state = PushState.PUSH_LEFT
            print(f"Starting LEFT push at t={self.state_start_time:.2f}s")
        elif self.push_count == 3:
            self.state = PushState.PUSH_BACK
            print(f"Starting BACK push at t={self.state_start_time:.2f}s")
        else:
            self.state = PushState.FINISHED
            print(f"All pushes completed at t={self.state_start_time:.2f}s")
    
    def _transition_after_push(self):
        """Transition back to standing after a push"""
        self.state_start_time = self.state_start_time + self.tpush
        
        if self.push_count < 3:
            self.state = PushState.STANDING
            print(f"Push {self.push_count} completed, standing at t={self.state_start_time:.2f}s")
        else:
            self.state = PushState.FINISHED
            print(f"All pushes completed at t={self.state_start_time:.2f}s")
    
    def get_state_name(self):
        """Get current state name for debugging"""
        return self.state.name

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
    - Includes ankle force-torque sensor publishers
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
        
        # Add publishers for ankle force-torque sensors
        self.left_ankle_wrench_pub = self.node.create_publisher(WrenchStamped, '/left_ankle_wrench', 10)
        self.right_ankle_wrench_pub = self.node.create_publisher(WrenchStamped, '/right_ankle_wrench', 10)
        
        # Add publisher for push force visualization
        self.push_force_pub = self.node.create_publisher(Vector3, '/push_force', 10)
        
        # Initialize joint state message
        self.joint_state_msg = JointState()
        self.joint_state_msg.header.frame_id = "world"
        
        # Set joint names from TSID model (skip universe and root_joint)
        if hasattr(model, 'names') and len(model.names) > 2:
            joint_names = [name for name in model.names[2:]]
            self.joint_state_msg.name = joint_names[:conf.na]
        else:
            self.joint_state_msg.name = [f"joint_{i}" for i in range(conf.na)]
        
        # Enable force-torque sensors on ankles
        self._enable_ankle_sensors()
        
        # Initialize pinocchio data for frame calculations
        self._pin_data = model.createData()
        
        # Debug line ID for force visualization
        self.force_line_id = -1
    
    def _enable_ankle_sensors(self):
        """Enable force-torque sensors on ankle joints"""
        try:
            # Enable sensors on 6th joint of each leg (10cm above sole)
            pb.enableJointForceTorqueSensor(self.id(), self.jointNameIndexMap()['leg_right_6_joint'], True)
            pb.enableJointForceTorqueSensor(self.id(), self.jointNameIndexMap()['leg_left_6_joint'], True)
            print("Ankle force-torque sensors enabled successfully")
        except KeyError as e:
            print(f"Warning: Could not enable ankle sensors - joint not found: {e}")
        except Exception as e:
            print(f"Warning: Could not enable ankle sensors: {e}")

    def update(self):
        # Update base class only - no pinocchio since TSID has its own wrapper
        super().update()
    
    def applyForce(self, force):
        """Apply external force to the robot's hip"""
        if np.linalg.norm(force) > 0:
            # Apply force to the base link (hip)
            pb.applyExternalForce(
                objectUniqueId=self.id(),
                linkIndex=-1,  # Base link
                forceObj=force,
                posObj=[0, 0, 0],  # Force applied at center of base link
                flags=pb.LINK_FRAME
            )

    def read_ankle_wrenches(self, q_current):
        """Read ankle force-torque sensor data"""
        wrenches = {'left': None, 'right': None}
        
        try:
            # Read right ankle wrench
            wren = pb.getJointState(self.id(), self.jointNameIndexMap()['leg_right_6_joint'])[2]
            wnp = np.array([-wren[0], -wren[1], -wren[2], -wren[3], -wren[4], -wren[5]])
            wrenches['right'] = pin.Force(wnp)
            
            # Read left ankle wrench
            wren = pb.getJointState(self.id(), self.jointNameIndexMap()['leg_left_6_joint'])[2]
            wnp = np.array([-wren[0], -wren[1], -wren[2], -wren[3], -wren[4], -wren[5]])
            wrenches['left'] = pin.Force(wnp)
            
        except Exception as e:
            print(f"Warning: Could not read ankle wrenches: {e}")
        
        return wrenches
    
    def get_frame_poses(self, q_current, model):
        """Get poses of ankle and sole frames"""
        try:
            # Update pinocchio kinematics
            pin.framesForwardKinematics(model, self._pin_data, q_current)
            
            # Get frame poses
            poses = {}
            poses['H_w_lsole'] = self._pin_data.oMf[model.getFrameId("left_sole_link")]
            poses['H_w_rsole'] = self._pin_data.oMf[model.getFrameId("right_sole_link")]
            poses['H_w_lankle'] = self._pin_data.oMf[model.getFrameId("leg_left_6_joint")]
            poses['H_w_rankle'] = self._pin_data.oMf[model.getFrameId("leg_right_6_joint")]
            
            return poses
        except Exception as e:
            print(f"Warning: Could not compute frame poses: {e}")
            return {}

    def visualize_push_force(self, force, simulator):
        """Visualize push force as a debug line in PyBullet"""
        try:
            if np.linalg.norm(force) > 0:
                # Get base position
                base_pos = np.array(pb.getBasePositionAndOrientation(self.id())[0])
                
                # Create line from base to force direction
                force_scale = 0.01  # Scale factor for visualization
                p1 = base_pos
                p2 = base_pos + force * force_scale
                
                # Remove previous line if exists
                if self.force_line_id != -1:
                    try:
                        pb.removeUserDebugItem(self.force_line_id)
                    except:
                        pass
                
                # Add new line (use pb.addUserDebugLine instead)
                self.force_line_id = pb.addUserDebugLine(
                    lineFromXYZ=p1,
                    lineToXYZ=p2,
                    lineColorRGB=[1, 0, 0],
                    lineWidth=3.0
                )
            else:
                # Remove line when no force is applied
                if self.force_line_id != -1:
                    try:
                        pb.removeUserDebugItem(self.force_line_id)
                        self.force_line_id = -1
                    except:
                        pass
        except Exception as e:
            # Silently handle debug line errors
            pass

    def publish(self, T_b_w, ankle_wrenches, push_force):
        """
        Publish function called at 30 Hz
        - Publishes robot's joint state
        - Broadcasts T_b_w transformation from TSID baseState()
        - Publishes ankle force-torque sensor data
        - Publishes push force for visualization
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
        
        # Publish ankle wrenches
        self._publish_ankle_wrenches(ankle_wrenches, current_time)
        
        # Publish push force
        self._publish_push_force(push_force, current_time)

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
    
    def _publish_ankle_wrenches(self, ankle_wrenches, timestamp):
        """Publish ankle force-torque sensor data"""
        if ankle_wrenches['left'] is not None:
            left_msg = WrenchStamped()
            left_msg.header.stamp = timestamp.to_msg()
            left_msg.header.frame_id = "leg_left_6_joint"
            
            left_msg.wrench.force.x = float(ankle_wrenches['left'].linear[0])
            left_msg.wrench.force.y = float(ankle_wrenches['left'].linear[1])
            left_msg.wrench.force.z = float(ankle_wrenches['left'].linear[2])
            left_msg.wrench.torque.x = float(ankle_wrenches['left'].angular[0])
            left_msg.wrench.torque.y = float(ankle_wrenches['left'].angular[1])
            left_msg.wrench.torque.z = float(ankle_wrenches['left'].angular[2])
            
            self.left_ankle_wrench_pub.publish(left_msg)
        
        if ankle_wrenches['right'] is not None:
            right_msg = WrenchStamped()
            right_msg.header.stamp = timestamp.to_msg()
            right_msg.header.frame_id = "leg_right_6_joint"
            
            right_msg.wrench.force.x = float(ankle_wrenches['right'].linear[0])
            right_msg.wrench.force.y = float(ankle_wrenches['right'].linear[1])
            right_msg.wrench.force.z = float(ankle_wrenches['right'].linear[2])
            right_msg.wrench.torque.x = float(ankle_wrenches['right'].angular[0])
            right_msg.wrench.torque.y = float(ankle_wrenches['right'].angular[1])
            right_msg.wrench.torque.z = float(ankle_wrenches['right'].angular[2])
            
            self.right_ankle_wrench_pub.publish(right_msg)
    
    def _publish_push_force(self, push_force, timestamp):
        """Publish push force for RViz visualization"""
        force_msg = Vector3()
        force_msg.x = float(push_force[0])
        force_msg.y = float(push_force[1])
        force_msg.z = float(push_force[2])
        
        self.push_force_pub.publish(force_msg)

################################################################################
# main function - exactly following task requirements
################################################################################

def main():
    rclpy.init()
    node = rclpy.create_node('tutorial_4_standing_node')
    
    try:
        print("=== Tutorial 4: Standing Controller with Push Forces ===")
        
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
        
        # Initialize push state machine
        print("Step 4: Initializing Push State Machine...")
        push_sm = PushStateMachine(tperiod=5.0, tpush=1.0, fpush_magnitude=50.0)
        
        # Brief initial settling
        robot.enablePositionControl()
        for i in range(30):
            robot.setActuatedJointPositions(conf.q_home)
            simulator.step()
            robot.update()
        
        # Switch to torque control for TSID
        robot.enableTorqueControl()
        
        # Variables for 30 Hz publishing
        t_publish = 0.0
        publish_rate = 30.0  # 30 Hz
        
        print("Starting simulation...")
        print("Push sequence: Wait 5s -> Right push (1s) -> Wait 5s -> Left push (1s) -> Wait 5s -> Back push (1s)")
        
        # Step 5: Create while loop to run the simulation
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
            
            # Update push state machine and get current push force
            push_force = push_sm.update(t)
            
            # Apply push force to robot
            robot.applyForce(push_force)
            
            # Visualize push force
            robot.visualize_push_force(push_force, simulator)
            
            # Read ankle force-torque sensors
            ankle_wrenches = robot.read_ankle_wrenches(q_current)
            
            # Get frame poses (for debugging/analysis)
            frame_poses = robot.get_frame_poses(q_current, model)
            
            # TSIDWrapper.update(...) solves problem in Eq. (10) and returns solution torque and acceleration
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
            
            # Step 6: Call robot's publish function at 30 Hz
            if t - t_publish >= 1.0 / publish_rate:
                t_publish = t
                
                # Get base-to-world transformation from TSID with baseState() function
                T_b_w = tsid_wrapper.baseState()
                
                # Call robot's publish function with ankle data and push force
                robot.publish(T_b_w, ankle_wrenches, push_force)
                
                # Print status every 2 seconds
                if int(t) % 2 == 0 and abs(t - int(t)) < 0.05:
                    state_name = push_sm.get_state_name()
                    force_mag = np.linalg.norm(push_force)
                    print(f"t={t:.1f}s - State: {state_name}, Force: {force_mag:.1f}N")
                    
                    # Print ankle forces if available
                    if ankle_wrenches['left'] is not None and ankle_wrenches['right'] is not None:
                        left_fz = ankle_wrenches['left'].linear[2]
                        right_fz = ankle_wrenches['right'].linear[2]
                        print(f"  Ankle forces - Left: {left_fz:.1f}N, Right: {right_fz:.1f}N")
            
            # ROS spin
            rclpy.spin_once(node, timeout_sec=0.001)
            
            # Exit condition for finite simulation
            if push_sm.state == PushState.FINISHED and t > push_sm.state_start_time + 2.0:
                print("Push sequence completed. Continuing standing...")
                # Continue running or break here if desired
                # break
    
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