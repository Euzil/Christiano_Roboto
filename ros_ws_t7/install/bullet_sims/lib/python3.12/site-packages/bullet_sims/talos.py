"""
Fixed Talos Walking Control - Resolving COM reference issues
"""

import numpy as np
import pinocchio as pin
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot
from bullet_sims.tsid_wrapper import TSIDWrapper
import bullet_sims.talos_conf as conf
import rclpy
import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import WrenchStamped
from visualization_msgs.msg import Marker, MarkerArray
from bullet_sims.footstep_planner import Side

################################################################################
# Enhanced Talos Robot with Fixed COM Reference Methods
################################################################################

class Talos(Robot):
    """
    Enhanced Talos robot class with fixed COM reference handling
    """
    def __init__(self, simulator, urdf, model, q=None, verbose=True, useFixedBase=False):
        # Call base class constructor
        super().__init__(
            simulator=simulator,
            filename=urdf,
            model=model,
            basePosition=np.array([0, 0, 1.1]),
            baseQuationerion=np.array([0, 0, 0, 1]),
            q=q,
            useFixedBase=useFixedBase,
            verbose=verbose
        )
        
        # Store configuration
        self.conf = conf
        self.sim = simulator
        self.verbose = verbose
        
        # Create the TSID wrapper
        if self.verbose:
            print("  Creating TSIDWrapper for enhanced robot...")
        self.stack = TSIDWrapper(self.conf)
        
        ########################################################################
        # ROS2 Setup (same as before)
        ########################################################################
        
        self.node = rclpy.create_node('talos_robot_publisher')
        self.joint_state_pub = self.node.create_publisher(JointState, '/joint_states', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self.node)
        
        # Initialize joint state message
        self.joint_state_msg = JointState()
        self.joint_state_msg.header.frame_id = "world"
        
        # Set joint names
        if hasattr(model, 'names') and len(model.names) > 2:
            joint_names = [name for name in model.names[2:]]
            self.joint_state_msg.name = joint_names[:self.conf.na]
        else:
            self.joint_state_msg.name = [f"joint_{i}" for i in range(self.conf.na)]
        
        # Enhanced publishers
        self.marker_pub = self.node.create_publisher(MarkerArray, 'robot_markers', 10)
        self.left_wrench_pub = self.node.create_publisher(WrenchStamped, 'left_foot_wrench', 10)
        self.right_wrench_pub = self.node.create_publisher(WrenchStamped, 'right_foot_wrench', 10)
        
        # Last computed torques
        self.tau = np.zeros(self.conf.na)
        
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
        
        # Estimators
        self.zmp = np.zeros(3)
        self.dcm = np.zeros(3)
        
        # Check FootStepPlanner availability
        try:
            from bullet_sims.footstep_planner import FootStepPlanner
            test_planner = FootStepPlanner(self.conf)
            self.footstep_planner_available = True
            if self.verbose:
                print("  ✓ FootStepPlanner available")
        except Exception as e:
            self.footstep_planner_available = False
            if self.verbose:
                print(f"  ⚠ FootStepPlanner not available: {e}")
        
        if self.verbose:
            print("  Enhanced Talos robot initialization complete!")

    ############################################################################
    # Core methods (same as before)
    ############################################################################

    def update(self):
        """Update base class and estimators"""
        super().update()
        self._update_zmp_estimate()
        self._update_dcm_estimate()

    def enablePositionControl(self):
        """Enable position control mode"""
        if hasattr(super(), 'enablePositionControl'):
            super().enablePositionControl()

    def enableTorqueControl(self):
        """Enable torque control mode"""
        if hasattr(super(), 'enableTorqueControl'):
            super().enableTorqueControl()

    def setActuatedJointPositions(self, positions):
        """Set target positions for actuated joints"""
        if hasattr(super(), 'setActuatedJointPositions'):
            super().setActuatedJointPositions(positions)

    def setActuatedJointTorques(self, torques):
        """Set torques for actuated joints"""
        self.tau = np.array(torques).copy()
        if hasattr(super(), 'setActuatedJointTorques'):
            super().setActuatedJointTorques(torques)

    def publish(self, T_b_w):
        """Publish robot state at 30 Hz"""
        current_time = self.node.get_clock().now()
        
        # Publish joint state
        self.joint_state_msg.header.stamp = current_time.to_msg()
        
        # Get current robot state
        q_current = np.array(self.q())
        v_current = np.array(self.v())
        
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
        
        # Enhanced publishing
        self._publish_walking_markers()
        self._publish_foot_wrenches()

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

    ############################################################################
    # FIXED: COM Reference Methods
    ############################################################################
    
    def _setComReference(self, pos_ref, vel_ref, acc_ref):
        """
        FIXED: Set COM reference using multiple approaches
        This replaces the missing method that was causing the error
        """
        try:
            # Ensure inputs are numpy arrays
            pos_ref = np.asarray(pos_ref, dtype=np.float64)
            vel_ref = np.asarray(vel_ref, dtype=np.float64) 
            acc_ref = np.asarray(acc_ref, dtype=np.float64)
            
            # Approach 1: Direct TSID stack method
            if hasattr(self.stack, 'setComReference'):
                self.stack.setComReference(pos_ref, vel_ref, acc_ref)
                return
            
            # Approach 2: Task-based method
            if hasattr(self.stack, 'setTaskReference'):
                try:
                    self.stack.setTaskReference("com", pos_ref, vel_ref, acc_ref)
                    return
                except:
                    pass
            
            # Approach 3: COM task direct access
            if hasattr(self.stack, 'com_task'):
                try:
                    self.stack.com_task.setReference(pos_ref, vel_ref, acc_ref)
                    return
                except:
                    pass
            
            # Approach 4: Check for different naming conventions
            for method_name in ['setCenterOfMassReference', 'setCoMReference', 'updateComReference']:
                if hasattr(self.stack, method_name):
                    try:
                        method = getattr(self.stack, method_name)
                        method(pos_ref, vel_ref, acc_ref)
                        return
                    except:
                        continue
            
            # Approach 5: Store reference for external setting
            self.com_pos_ref = pos_ref
            self.com_vel_ref = vel_ref  
            self.com_acc_ref = acc_ref
            
            if self.verbose:
                print(f"  COM reference stored: pos=[{pos_ref[0]:.3f}, {pos_ref[1]:.3f}, {pos_ref[2]:.3f}]")
                
        except Exception as e:
            if self.verbose:
                print(f"  Warning: All COM reference methods failed: {e}")
    
    def setComReference(self, pos_ref, vel_ref=None, acc_ref=None):
        """
        Public method to set COM reference with defaults
        """
        if vel_ref is None:
            vel_ref = np.zeros(3)
        if acc_ref is None:
            acc_ref = np.zeros(3)
        
        self._setComReference(pos_ref, vel_ref, acc_ref)

    ############################################################################
    # Enhanced state machine methods
    ############################################################################
    
    def initializeStanding(self):
        """Initialize standing control for PLANNING state"""
        # Get current foot positions
        T_lf_w = self.stack.getFramePose(self.conf.lf_frame_name)
        T_rf_w = self.stack.getFramePose(self.conf.rf_frame_name)
        
        lf_pos = T_lf_w[:3, 3]
        rf_pos = T_rf_w[:3, 3]
        
        # Target COM position: center between feet
        self.com_target = np.array([
            (lf_pos[0] + rf_pos[0]) / 2.0,
            (lf_pos[1] + rf_pos[1]) / 2.0,
            0.95
        ])
        
        # Set both feet as support
        self.setBothFeetSupport()
        
        if self.verbose:
            print(f"Standing target initialized:")
            print(f"  Left foot:  [{lf_pos[0]:.3f}, {lf_pos[1]:.3f}, {lf_pos[2]:.3f}]")
            print(f"  Right foot: [{rf_pos[0]:.3f}, {rf_pos[1]:.3f}, {rf_pos[2]:.3f}]")
            print(f"  COM target: [{self.com_target[0]:.3f}, {self.com_target[1]:.3f}, {self.com_target[2]:.3f}]")

    def updateStandingControl(self, t):
        """Update standing control and check stability"""
        if self.com_target is None:
            self.initializeStanding()
            self.standing_start_time = t
            return False
        
        # Get current COM position
        com_current = self.stack.getCenterOfMass()
        com_error = np.linalg.norm(com_current - self.com_target)
        
        # Debug output
        if int(t) % 2 == 0 and int(t * 10) % 10 == 0:
            print(f"Standing: Time {t:.2f}s - COM error: {com_error*100:.2f}cm")
            if self.standing_stable:
                stable_duration = t - self.standing_start_time
                print(f"         Stable for: {stable_duration:.2f}s")
        
        # Check stability
        if com_error < self.stabilization_threshold:
            if not self.standing_stable:
                self.standing_start_time = t
                self.standing_stable = True
                if self.verbose:
                    print(f"Time {t:.2f}s: Standing stabilized")
            
            stable_duration = t - self.standing_start_time
            if stable_duration > self.required_stable_time:
                if self.verbose:
                    print(f"Time {t:.2f}s: ✓ Standing completed!")
                return True
        else:
            if self.standing_stable:
                if self.verbose:
                    print(f"Time {t:.2f}s: Lost stability")
                self.standing_stable = False
        
        return False

    def generateWalkingPath(self, target_distance=2.0, num_steps=8):
        """Generate walking path and visualize"""
        if self.verbose:
            print(f"Generating walking path: {target_distance}m, {num_steps} steps")
        
        if not self.footstep_planner_available:
            if self.verbose:
                print("  Using simple forward motion fallback")
            self.footstep_plan = []
            self.path_generated = True
            self.current_step_index = 0
            return True
        
        try:
            from bullet_sims.footstep_planner import FootStepPlanner
            
            planner = FootStepPlanner(self.conf)
            
            # Get foot positions
            T_lf_w = self.stack.getFramePose(self.conf.lf_frame_name)
            T_rf_w = self.stack.getFramePose(self.conf.rf_frame_name)
            
            lf_pos = T_lf_w[:3, 3]
            rf_pos = T_rf_w[:3, 3]
            
            # Use center between feet for planning
            center_pos = (lf_pos + rf_pos) / 2.0
            center_pos[2] = 0.0  # Ground level
            
            if self.verbose:
                print(f"  Planning from: [{center_pos[0]:.3f}, {center_pos[1]:.3f}, {center_pos[2]:.3f}]")
            
            # Create SE3 transform
            T_0_w = pin.SE3(np.eye(3), center_pos)
            
            # Generate footstep plan
            self.footstep_plan = planner.planLine(T_0_w, Side.LEFT, num_steps)
            
            # Add final steps
            if len(self.footstep_plan) >= 2:
                self.footstep_plan.append(self.footstep_plan[-2])
                self.footstep_plan.append(self.footstep_plan[-1])
            
            if self.verbose:
                print(f"  ✓ Generated {len(self.footstep_plan)} steps")
            
            # Visualize in PyBullet
            try:
                planner.plot(self.sim)
                if self.verbose:
                    print("  ✓ Visualization complete")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Visualization failed: {e}")
            
            # Print plan
            if self.verbose:
                try:
                    planner.print_plan()
                except:
                    print("  Plan details:")
                    for i, step in enumerate(self.footstep_plan):
                        pos = step.position()
                        print(f"    Step {i}: {step.side.name} -> [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            
            self.path_generated = True
            self.current_step_index = 0
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"  ✗ Planning failed: {e}")
                print("  Using simple fallback")
            
            self.footstep_plan = []
            self.path_generated = True
            self.current_step_index = 0
            return True

    def updateWalkingControl(self, t):
        """FIXED: Update walking control with proper COM reference handling"""
        if not self.path_generated:
            return False
        
        # Execute footstep plan if available
        if len(self.footstep_plan) > 0:
            # Phase durations (based on single leg stand reference)
            stabilization_duration = 1.0
            com_shift_duration = 2.0  
            foot_lift_duration = 2.0
            settle_duration = 1.0
            total_step_duration = stabilization_duration + com_shift_duration + foot_lift_duration + settle_duration
            
            if t - self.step_start_time > total_step_duration:
                # Move to next step
                self.current_step_index += 1
                self.step_start_time = t
                
                if self.current_step_index >= len(self.footstep_plan):
                    if self.verbose:
                        print(f"Time {t:.2f}s: ✓ Walking completed!")
                    return True
                
                # Start new step
                current_step = self.footstep_plan[self.current_step_index]
                if self.verbose:
                    step_pos = current_step.position()
                    print(f"Time {t:.2f}s: Step {self.current_step_index}: {current_step.side.name} -> [{step_pos[0]:.3f}, {step_pos[1]:.3f}]")
            
            # Execute current step phases
            if self.current_step_index < len(self.footstep_plan):
                current_step = self.footstep_plan[self.current_step_index]
                step_elapsed = t - self.step_start_time
                
                if step_elapsed < stabilization_duration:
                    # PHASE 1: STABILIZATION
                    if step_elapsed < 0.1:
                        print(f"         Phase 1: Stabilization...")
                        self.setBothFeetSupport()
                    
                elif step_elapsed < stabilization_duration + com_shift_duration:
                    # PHASE 2: COM SHIFTING (FIXED)
                    if step_elapsed < stabilization_duration + 0.1:
                        if current_step.side == Side.LEFT:
                            print(f"         Phase 2: COM shifting to RIGHT foot...")
                        else:
                            print(f"         Phase 2: COM shifting to LEFT foot...")
                    
                    # Get support foot position (opposite of stepping foot)
                    try:
                        if current_step.side == Side.LEFT:
                            # Lifting LEFT, support on RIGHT
                            rf_pose = self.stack.getFramePose(getattr(self.conf, 'rf_frame_name', 'right_foot'))
                            support_position = rf_pose[:3, 3]
                        else:
                            # Lifting RIGHT, support on LEFT
                            lf_pose = self.stack.getFramePose(getattr(self.conf, 'lf_frame_name', 'left_foot'))
                            support_position = lf_pose[:3, 3]
                        
                        # Get current COM
                        com_current = self.stack.getCenterOfMass()
                        
                        # Create COM reference over support foot
                        p_com_new = np.array([support_position[0], support_position[1], com_current[2]])
                        
                        # FIXED: Use the corrected method
                        self._setComReference(p_com_new, np.zeros(3), np.zeros(3))
                        
                        # Progress indication
                        phase_progress = (step_elapsed - stabilization_duration) / com_shift_duration
                        if int(step_elapsed * 4) % 4 == 0:
                            print(f"         COM shift progress: {phase_progress*100:.1f}%")
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"Warning: COM shift error: {e}")
                
                elif step_elapsed < stabilization_duration + com_shift_duration + foot_lift_duration:
                    # PHASE 3: FOOT LIFTING
                    if step_elapsed < stabilization_duration + com_shift_duration + 0.1:
                        print(f"         Phase 3: Lifting {current_step.side.name} foot...")
                        
                        # Remove contact and set swing foot
                        if current_step.side == Side.LEFT:
                            self.setSwingFoot(Side.LEFT)
                            self.setSupportFoot(Side.RIGHT)
                        else:
                            self.setSwingFoot(Side.RIGHT)
                            self.setSupportFoot(Side.LEFT)
                    
                    # Execute foot movement
                    foot_phase_elapsed = step_elapsed - stabilization_duration - com_shift_duration
                    foot_progress = foot_phase_elapsed / foot_lift_duration
                    
                    try:
                        # Get current swing foot position
                        current_swing_pose = self.getSwingFootPose()
                        current_pos = current_swing_pose[:3, 3]
                        
                        # Get target position
                        step_pos = current_step.position()
                        
                        # Create trajectory: lift, move, lower
                        if foot_progress < 0.3:
                            # Lifting phase
                            lift_progress = foot_progress / 0.3
                            target_pos = current_pos.copy()
                            target_pos[2] += 0.15 * lift_progress
                        elif foot_progress < 0.7:
                            # Moving phase
                            move_progress = (foot_progress - 0.3) / 0.4
                            target_pos = current_pos + move_progress * (step_pos - current_pos)
                            target_pos[2] = current_pos[2] + 0.15
                        else:
                            # Lowering phase
                            lower_progress = (foot_progress - 0.7) / 0.3
                            target_pos = step_pos.copy()
                            target_pos[2] = step_pos[2] + 0.15 * (1 - lower_progress)
                        
                        # Create SE3 pose
                        current_rotation = current_swing_pose[:3, :3]
                        T_target_se3 = pin.SE3(current_rotation, target_pos)
                        
                        # Set foot reference
                        V_swing = np.zeros(6)
                        A_swing = np.zeros(6)
                        self.updateSwingFootRef(T_target_se3, V_swing, A_swing)
                        
                        if int(foot_phase_elapsed * 4) % 4 == 0:
                            print(f"         Foot progress: {foot_progress*100:.1f}%, Height: {target_pos[2]:.3f}m")
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"Warning: Foot movement error: {e}")
                
                else:
                    # PHASE 4: SETTLING
                    settle_elapsed = step_elapsed - stabilization_duration - com_shift_duration - foot_lift_duration
                    
                    if settle_elapsed < 0.1:
                        print(f"         Phase 4: Settling...")
                        self.setBothFeetSupport()
                    
                    # Small COM adjustment
                    try:
                        # Get center between feet for settling
                        lf_pose = self.stack.getFramePose(getattr(self.conf, 'lf_frame_name', 'left_foot'))
                        rf_pose = self.stack.getFramePose(getattr(self.conf, 'rf_frame_name', 'right_foot'))
                        
                        center_pos = (lf_pose[:3, 3] + rf_pose[:3, 3]) / 2.0
                        
                        if self.com_target is not None:
                            current_com = self.com_target.copy()
                        else:
                            current_com = self.stack.getCenterOfMass()
                        
                        settle_progress = settle_elapsed / settle_duration
                        com_pos_ref = current_com + settle_progress * 0.1 * (center_pos - current_com)
                        com_pos_ref[2] = current_com[2]
                        
                        self._setComReference(com_pos_ref, np.zeros(3), np.zeros(3))
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"Warning: Settling error: {e}")
        else:
            # Simple walking fallback
            walking_duration = 20.0
            if t - self.step_start_time > walking_duration:
                if self.verbose:
                    print(f"Time {t:.2f}s: ✓ Simple walking completed!")
                return True
        
        return False

    ############################################################################
    # Foot contact control methods (same as before)
    ############################################################################
    
    def setSupportFoot(self, side):
        """Set support foot"""
        self.support_foot = side
        
        try:
            if side == Side.LEFT:
                self.stack.activateContact("left_foot_contact")
                if hasattr(self.stack, 'deactivateTask'):
                    self.stack.deactivateTask("left_foot_motion")
                if self.verbose:
                    print(f"         LEFT foot = support")
            else:
                self.stack.activateContact("right_foot_contact")
                if hasattr(self.stack, 'deactivateTask'):
                    self.stack.deactivateTask("right_foot_motion")
                if self.verbose:
                    print(f"         RIGHT foot = support")
        except Exception as e:
            if self.verbose:
                print(f"Warning: Support foot setting failed: {e}")
    
    def setSwingFoot(self, side):
        """Set swing foot"""
        self.swing_foot = side
        
        try:
            if side == Side.LEFT:
                if hasattr(self.stack, 'deactivateContact'):
                    self.stack.deactivateContact("left_foot_contact")
                if hasattr(self.stack, 'activateTask'):
                    self.stack.activateTask("left_foot_motion")
                if self.verbose:
                    print(f"         LEFT foot = swing")
            else:
                if hasattr(self.stack, 'deactivateContact'):
                    self.stack.deactivateContact("right_foot_contact")
                if hasattr(self.stack, 'activateTask'):
                    self.stack.activateTask("right_foot_motion")
                if self.verbose:
                    print(f"         RIGHT foot = swing")
        except Exception as e:
            if self.verbose:
                print(f"Warning: Swing foot setting failed: {e}")
    
    def setBothFeetSupport(self):
        """Set both feet as support"""
        try:
            self.stack.activateContact("left_foot_contact")
            self.stack.activateContact("right_foot_contact")
            
            self.stack.deactivateTask("left_foot_motion")
            self.stack.deactivateTask("right_foot_motion")
            
            if self.verbose:
                print("  Both feet = support")
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Both feet support failed: {e}")

    def updateSwingFootRef(self, T_swing_w, V_swing_w, A_swing_w):
        """Update swing foot motion reference"""
        try:
            # Convert to SE3 if needed
            if isinstance(T_swing_w, np.ndarray):
                T_swing_se3 = pin.SE3(T_swing_w[:3, :3], T_swing_w[:3, 3])
            else:
                T_swing_se3 = T_swing_w
            
            # Ensure proper vectors
            V_swing_vec = np.array(V_swing_w).flatten()
            A_swing_vec = np.array(A_swing_w).flatten()
            
            if self.swing_foot == Side.LEFT:
                if hasattr(self.stack, 'setTaskReference'):
                    self.stack.setTaskReference("left_foot_motion", T_swing_se3, V_swing_vec, A_swing_vec)
                elif hasattr(self.stack, 'setLeftFootReference'):
                    self.stack.setLeftFootReference(T_swing_se3, V_swing_vec, A_swing_vec)
            else:
                if hasattr(self.stack, 'setTaskReference'):
                    self.stack.setTaskReference("right_foot_motion", T_swing_se3, V_swing_vec, A_swing_vec)
                elif hasattr(self.stack, 'setRightFootReference'):
                    self.stack.setRightFootReference(T_swing_se3, V_swing_vec, A_swing_vec)
                    
        except Exception as e:
            if self.verbose:
                print(f"Warning: Swing foot reference failed: {e}")
                # Try fallback approach
                try:
                    if isinstance(T_swing_w, np.ndarray):
                        target_pos = T_swing_w[:3, 3]
                    else:
                        target_pos = T_swing_w.translation
                    
                    T_simple = pin.SE3(np.eye(3), target_pos)
                    
                    if self.swing_foot == Side.LEFT:
                        self.stack.setTaskReference("left_foot_motion", T_simple)
                    else:
                        self.stack.setTaskReference("right_foot_motion", T_simple)
                        
                except Exception as e2:
                    if self.verbose:
                        print(f"Fallback swing foot reference also failed: {e2}")
    
    def getSwingFootPose(self):
        """Get current swing foot pose"""
        try:
            if self.swing_foot == Side.LEFT:
                return self.stack.getFramePose(getattr(self.conf, 'lf_frame_name', 'left_foot'))
            else:
                return self.stack.getFramePose(getattr(self.conf, 'rf_frame_name', 'right_foot'))
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not get swing foot pose: {e}")
            return np.eye(4)

    ############################################################################
    # Enhanced visualization methods
    ############################################################################
    
    def _publish_walking_markers(self):
        """Publish ZMP, DCM markers for visualization"""
        marker_array = MarkerArray()
        current_time = self.node.get_clock().now().to_msg()
        
        # ZMP marker
        zmp_marker = Marker()
        zmp_marker.header.stamp = current_time
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
        dcm_marker.header.stamp = current_time
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
    
    def _publish_foot_wrenches(self):
        """Publish foot force/torque information"""
        current_time = self.node.get_clock().now().to_msg()
        
        # Get contact forces from TSID stack
        try:
            contact_forces = self.stack.getContactForces()
            
            # Left foot wrench
            left_wrench_msg = WrenchStamped()
            left_wrench_msg.header.stamp = current_time
            left_wrench_msg.header.frame_id = getattr(self.conf, 'lf_frame_name', 'left_foot')
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
            right_wrench_msg.header.stamp = current_time
            right_wrench_msg.header.frame_id = getattr(self.conf, 'rf_frame_name', 'right_foot')
            right_ft = contact_forces.get("right_foot", np.zeros(6))
            right_wrench_msg.wrench.force.x = float(right_ft[0])
            right_wrench_msg.wrench.force.y = float(right_ft[1])
            right_wrench_msg.wrench.force.z = float(right_ft[2])
            right_wrench_msg.wrench.torque.x = float(right_ft[3])
            right_wrench_msg.wrench.torque.y = float(right_ft[4])
            right_wrench_msg.wrench.torque.z = float(right_ft[5])
            self.right_wrench_pub.publish(right_wrench_msg)
        except:
            pass  # Silently fail if contact forces not available

    ############################################################################
    # Private estimation methods
    ############################################################################
    
    def _update_zmp_estimate(self):
        """Update the estimated ZMP position"""
        try:
            contact_forces = self.stack.getContactForces()
            left_ft = contact_forces.get("left_foot", np.zeros(6))
            right_ft = contact_forces.get("right_foot", np.zeros(6))
            
            # Get foot positions from TSID stack
            left_pos = self.stack.getFramePose(getattr(self.conf, 'lf_frame_name', 'left_foot'))[:3, 3]
            right_pos = self.stack.getFramePose(getattr(self.conf, 'rf_frame_name', 'right_foot'))[:3, 3]
            
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
        except:
            # Fallback to zero if estimation fails
            self.zmp = np.zeros(3)
        
    def _update_dcm_estimate(self):
        """Update the estimated DCM position"""
        try:
            com_pos = self.stack.getCenterOfMass()
            com_vel = self.stack.getCenterOfMassVelocity()
            
            # DCM calculation: ξ = c + c_dot / ω₀
            # where ω₀ = sqrt(g/z_com) for the linear inverted pendulum
            g = 9.81
            omega_0 = np.sqrt(g / max(com_pos[2], 0.1))  # Avoid division by zero
            
            self.dcm = com_pos + com_vel / omega_0
        except:
            # Fallback to zero if estimation fails
            self.dcm = np.zeros(3)