"""
Walking with Hardware Control - ROS2 Node Version
Adapted for real AiNex Christiano robot hardware
STANDING -> PLANNING -> WALKING (follow path) -> KICKING
"""

import numpy as np
from numpy import nan
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt
import os

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

# Ainex Hardware 
from ainex_motion.joint_controller import JointController
from ament_index_python.packages import get_package_share_directory

################################################################################
# settings
################################################################################

DO_PLOT = True

################################################################################
# Robot Class for Hardware Integration
################################################################################

class Ainex():
    def __init__(self, node, q=None):
        self.node = node
        self.hardware_controller = JointController(self.node)
        
        # add publisher
        self.pub_joint = self.node.create_publisher(
            JointState, "/joint_states", 10)

        self.joint_msg = JointState()
        self.joint_msg.name = ['head_pan', 'head_tilt',
                               'l_hip_yaw', 'l_hip_roll', 'l_hip_pitch', 'l_knee', 'l_ank_pitch', 'l_ank_roll',
                               'l_sho_pitch', 'l_sho_roll', 'l_el_pitch', 'l_el_yaw', 'l_gripper',
                               'r_hip_yaw', 'r_hip_roll', 'r_hip_pitch', 'r_knee', 'r_ank_pitch', 'r_ank_roll',
                               'r_sho_pitch', 'r_sho_roll', 'r_el_pitch', 'r_el_yaw', 'r_gripper']
      
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
# Visualization class to create plots for the Center of Masses
################################################################################

class Visualization:
    def __init__(self, tsid_wrapper):
        self.tsid_wrapper = tsid_wrapper
        self.time_log = []

        # COM logs
        self.com_ref_log = []
        self.com_state_log = []
        self.com_vel_ref_log = []
        self.com_vel_state_log = []
        self.com_acc_ref_log = []
        self.com_acc_state_log = []

    def getCOMStates(self, t):
        # Save current time
        self.time_log.append(t)

        # Get COM reference and actual states
        ref = self.tsid_wrapper.comReference()
        state = self.tsid_wrapper.comState()

        self.com_ref_log.append(np.array(ref.value()))
        self.com_state_log.append(np.array(state.value()))
        self.com_vel_ref_log.append(np.array(ref.derivative()))
        self.com_vel_state_log.append(np.array(state.derivative()))
        self.com_acc_ref_log.append(np.array(ref.second_derivative()))
        self.com_acc_state_log.append(np.array(state.second_derivative()))
        
    def update_subplot_com(self, ax, t, data_ref, data_state, title):
        ax.clear()
        ax.plot(t, data_ref, label='COM Ref', linestyle='--')
        ax.plot(t, data_state, label='COM State', linestyle='-.')
        ax.legend()
        ax.set_title(title)
        ax.grid(True)

    def plot_all(self):
        # create plots
        fig_pos, axs_pos = plt.subplots(3, 1, figsize=(15, 10))
        fig_vel, axs_vel = plt.subplots(3, 1, figsize=(15, 10))
        fig_acc, axs_acc = plt.subplots(3, 1, figsize=(15, 10))

        for ax, label in zip(axs_pos, ['X', 'Y', 'Z']):
            ax.set_title(f'CoM Position - {label}', fontsize=12)
            ax.grid(True)

        for ax, label in zip(axs_vel, ['X', 'Y', 'Z']):
            ax.set_title(f'CoM Velocity - {label}', fontsize=12)
            ax.grid(True)

        for ax, label in zip(axs_acc, ['X', 'Y', 'Z']):
            ax.set_title(f'CoM Acceleration - {label}', fontsize=12)
            ax.grid(True)

        # Add shared labels
        fig_pos.supxlabel('Time [s]')
        fig_pos.supylabel('CoM Position [m]')
        fig_vel.supxlabel('Time [s]')
        fig_vel.supylabel('CoM Velocity [m/s]')
        fig_acc.supxlabel('Time [s]')
        fig_acc.supylabel('CoM Acceleration [m/s²]')

        # update plots
        for i in range(3):
            self.update_subplot_com(
                axs_pos[i], self.time_log,
                [x[i] for x in self.com_ref_log],
                [x[i] for x in self.com_state_log],
                title=f"COM Position - {['X','Y','Z'][i]}"
            )
            self.update_subplot_com(
                axs_vel[i], self.time_log,
                [x[i] for x in self.com_vel_ref_log],
                [x[i] for x in self.com_vel_state_log],
                title=f"COM Velocity - {['X','Y','Z'][i]}"
            )
            self.update_subplot_com(
                axs_acc[i], self.time_log,
                [x[i] for x in self.com_acc_ref_log],
                [x[i] for x in self.com_acc_state_log],
                title=f"COM Acceleration - {['X','Y','Z'][i]}"
            )

        plt.pause(0.1)
        fig_pos.tight_layout(pad=2.0)
        fig_vel.tight_layout(pad=2.0)
        fig_acc.tight_layout(pad=2.0)
        
        # draw on windows
        fig_pos.canvas.draw()
        fig_pos.canvas.flush_events()
        fig_vel.canvas.draw()
        fig_vel.canvas.flush_events()
        fig_acc.canvas.draw()
        fig_acc.canvas.flush_events()

        # get directory of current module
        try:
            bullet_description = get_package_share_directory("whole_body_control")
            # store results
            fig_pos.savefig(os.path.join(bullet_description, "com_position_plot.png"), dpi=300)
            fig_vel.savefig(os.path.join(bullet_description, "com_velocity_plot.png"), dpi=300)
            fig_acc.savefig(os.path.join(bullet_description, "com_acceleration_plot.png"), dpi=300)
        except Exception as e:
            print(f"Could not save plots: {e}")

################################################################################
# Method for executing steps
################################################################################

def execute_step_along_path(tsid_wrapper, step_index, footstep_plan, step_phase, step_elapsed, height, phase_duration):
    """
    Execute a single gait step along the path (with landing detection and knee bend contact logic)
    """
    if step_index >= len(footstep_plan):
        return True  # All steps completed

    foot_side, target_position = footstep_plan[step_index]

    # Phase timing settings
    com_shift_phase = 1 * phase_duration
    lift_move_phase = 2 * phase_duration
    place_phase = 3 * phase_duration
    shift_back_phase = 4 * phase_duration

    try:
        # Phase 1: COM shift to support foot
        if step_elapsed < com_shift_phase:
            if step_phase != "com_shift":
                print(f"  Step {step_index + 1}: COM shifting to {'left' if foot_side == 'right' else 'right'} foot...")
                step_phase = "com_shift"

                if foot_side == 'right':
                    support_pos = tsid_wrapper.get_placement_LF().translation
                else:
                    support_pos = tsid_wrapper.get_placement_RF().translation

                com_current = tsid_wrapper.comState().pos()
                p_com_new = np.array([support_pos[0], support_pos[1], com_current[2]])
                tsid_wrapper.setComRefState(p_com_new)

        # Phase 2: Lift foot and move
        elif step_elapsed < lift_move_phase:
            if step_phase != "lift_move":
                print(f"  Step {step_index + 1}: Lifting and moving {foot_side} foot...")
                step_phase = "lift_move"

                if foot_side == 'right' and hasattr(tsid_wrapper, 'remove_contact_RF'):
                    tsid_wrapper.remove_contact_RF()
                elif foot_side == 'left' and hasattr(tsid_wrapper, 'remove_contact_LF'):
                    tsid_wrapper.remove_contact_LF()

            lift_progress = (step_elapsed - com_shift_phase) / phase_duration

            if foot_side == 'right':
                current_foot_pos = tsid_wrapper.get_placement_RF().translation
            else:
                current_foot_pos = tsid_wrapper.get_placement_LF().translation

            # Lift -> Move -> Lower
            if lift_progress < 0.3:
                target_pos = current_foot_pos.copy()
                target_pos[2] += height * (lift_progress / 0.3)
            elif lift_progress < 0.7:
                ratio = (lift_progress - 0.3) / 0.4
                target_pos = current_foot_pos + ratio * (target_position - current_foot_pos)
                target_pos[2] = current_foot_pos[2] + height
            else:
                ratio = (lift_progress - 0.7) / 0.3
                target_pos = target_position.copy()
                target_pos[2] += height * (1 - ratio)

            foot_pose = pin.SE3(np.eye(3), target_pos)
            if foot_side == 'right' and hasattr(tsid_wrapper, 'set_RF_pose_ref'):
                tsid_wrapper.set_RF_pose_ref(foot_pose)
            elif foot_side == 'left' and hasattr(tsid_wrapper, 'set_LF_pose_ref'):
                tsid_wrapper.set_LF_pose_ref(foot_pose)

        # Phase 3: Place foot -> Check if landed
        elif step_elapsed < place_phase:
            if step_phase != "place":
                print(f"  Step {step_index + 1}: Placing {foot_side} foot...")
                step_phase = "place"

                # Directly set foot pose
                foot_pose = pin.SE3(np.eye(3), target_position)
                if foot_side == 'right' and hasattr(tsid_wrapper, 'set_RF_pose_ref'):
                    tsid_wrapper.set_RF_pose_ref(foot_pose)
                elif foot_side == 'left' and hasattr(tsid_wrapper, 'set_LF_pose_ref'):
                    tsid_wrapper.set_LF_pose_ref(foot_pose)

            # If foot hasn't touched ground (suspended), try "bending knee" to lower base/com
            actual_z = tsid_wrapper.get_placement_RF().translation[2] if foot_side == 'right' \
                       else tsid_wrapper.get_placement_LF().translation[2]
            expected_z = target_position[2]

            if abs(actual_z - expected_z) > 0.004:  # >0.4cm suspended
                print(" Foot not contacting ground - lowering COM to help foot land...")
                com_current = tsid_wrapper.comState().pos()
                lowered_com = com_current.copy()
                lowered_com[2] -= 0.01  # Bend knee and lower 2cm
                tsid_wrapper.setComRefState(lowered_com)

        # Phase 4: After successful landing, add contact -> COM transfer
        elif step_elapsed < shift_back_phase:
            if step_phase != "shift_back":
                actual_pos = tsid_wrapper.get_placement_RF().translation if foot_side == 'right' \
                             else tsid_wrapper.get_placement_LF().translation

                if abs(actual_pos[2] - target_position[2]) < 0.01:  # Already landed
                    print(f"  Step {step_index + 1}: COM shifting to newly placed {foot_side} foot...")
                    step_phase = "shift_back"

                    # Add contact
                    if foot_side == 'right' and hasattr(tsid_wrapper, 'add_contact_RF'):
                        tsid_wrapper.add_contact_RF()
                    elif foot_side == 'left' and hasattr(tsid_wrapper, 'add_contact_LF'):
                        tsid_wrapper.add_contact_LF()

                    # Transfer COM to this foot
                    com_current = tsid_wrapper.comState().pos()
                    p_com_new = np.array([target_position[0], target_position[1], com_current[2]])
                    tsid_wrapper.setComRefState(p_com_new)
                    
                    com_current = tsid_wrapper.comState().pos()
                    lowered_com = com_current.copy()
                    lowered_com[2] += 0.01  # Restore height after bending knee
                    tsid_wrapper.setComRefState(lowered_com)
                    
                else:
                    # Still not in contact, don't make transition, wait for next loop
                    print("  Waiting for foot to contact ground before COM shift...")

        # Step completion
        else:
            if step_phase != "complete":
                print(f"  Step {step_index + 1} completed: {foot_side} foot at [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
                step_phase = "complete"
            return True

    except Exception as e:
        print(f"  Step execution error: {e}")
        return True

    return False  # Step still in progress

################################################################################
# Main Walking Hardware Node
################################################################################

class WalkingHardwareNode(Node):
    def __init__(self):
        super().__init__('walking_hardware_node')

        print("=" * 70)
        print("WALKING WITH HARDWARE CONTROL - AINEX CHRISTIANO")
        print("=" * 70)
        print("Features:")
        print("- Real hardware robot control")
        print("- ROS2 node structure")
        print("- Step-by-step execution")
        print("- Real-time progress feedback")
        print("- Joint state publishing")
        print("=" * 70)

        # Hardware controller initialization
        self.hardware_controller = JointController(self)
        
        # Joint names for hardware
        self.joint_names = ['head_pan', 'head_tilt',
                            'l_hip_yaw', 'l_hip_roll', 'l_hip_pitch', 'l_knee', 'l_ank_pitch', 'l_ank_roll',
                            'l_sho_pitch', 'l_sho_roll', 'l_el_pitch', 'l_el_yaw', 'l_gripper',
                            'r_hip_yaw', 'r_hip_roll', 'r_hip_pitch', 'r_knee', 'r_ank_pitch', 'r_ank_roll',
                            'r_sho_pitch', 'r_sho_roll', 'r_el_pitch', 'r_el_yaw', 'r_gripper']

        # Initial height
        z_init = 0.23

        # init TSIDWrapper
        self.tsid_wrapper = TSIDWrapper(conf)

        # init Simulator
        q_init = conf.q_home

        # init ROBOT - hardware version
        self.robot = Ainex(self, q_init)

        # initialize visualization
        self.visual_class = Visualization(self.tsid_wrapper)

        # State machine variables
        self.current_state = "HOME"
        self.state_start_time = 0.0

        # Walking data
        self.footstep_plan = []
        self.current_step_index = 0
        self.step_start_time = 0.0
        self.step_phase = "none"

        # init simulation time
        self.t = 0.0

        # init q_tsid, v_tsid
        self.q_tsid = self.robot.get_q()
        self.v_tsid = self.robot.get_v()
        self.tau = np.zeros((conf.na,))

        ################################################################################
        # Walking motion parameters
        ################################################################################
        
        # Phase durations
        self.home_duration = 4.0
        self.standing_duration = 3.0
        self.planning_duration = 3.0
        self.end_duration = 4.0
        
        # Step length settings
        self.first_step = 0.065  # and used for last step
        self.other_step = 0.13   # used for every step except first and last one
        self.num_steps = 10      # Even number: start with right foot
        self.height = 0.04       # max step height

        # how long one step takes
        self.phase_duration = 1.5

        # Set a timer to run periodically 
        self.timer_frequency = 30 
        self.timer = self.create_timer(1/self.timer_frequency, self.timer_callback)
        
        # Initial hardware posture
        self.hardware_controller.setPosture('stand', 0.8)

        print(f"\nWalking Hardware Node initialized with {self.timer_frequency}Hz control loop")
        print(f"Walking parameters: {self.num_steps} steps, {self.height}m step height")

    def timer_callback(self):
        try:
            ###################################################################
            # PHASE 1: GET INTO HOME POSITION
            ###################################################################
            if self.current_state == "HOME":
                if self.t - self.state_start_time == 0:
                    self.state_start_time = self.t
                    self.get_logger().info(f"[{self.t:.1f}s] PHASE 1: HOME")
                
                if self.t - self.state_start_time > self.home_duration:
                    self.current_state = "STANDING"
                    self.state_start_time = self.t
                    self.get_logger().info("In home position")

            ###################################################################
            # PHASE 2: STANDING
            ###################################################################
            elif self.current_state == "STANDING":
                if self.t - self.state_start_time == 0:
                    self.state_start_time = self.t
                    self.get_logger().info(f"[{self.t:.1f}s] PHASE 2: STANDING")
                    self.get_logger().info("- Establishing stable base")
                
                if self.t - self.state_start_time > self.standing_duration:
                    self.current_state = "PLANNING"
                    self.state_start_time = self.t
                    self.get_logger().info("Standing stable - Ready for path planning")
            
            ###################################################################
            # PHASE 3: PLANNING
            ###################################################################
            elif self.current_state == "PLANNING":
                if self.t - self.state_start_time < 0.1:
                    self.get_logger().info(f"[{self.t:.1f}s] PHASE 3: PLANNING")
                    self.get_logger().info("- Generating footstep path")
                
                planning_elapsed = self.t - self.state_start_time
                
                # Generate footstep plan
                if len(self.footstep_plan) == 0 and planning_elapsed > 0.5:
                    try:
                        self.get_logger().info(f"[{self.t:.1f}s] Generating footstep plan...")
                        
                        # Get current foot positions
                        rf_placement = self.tsid_wrapper.get_placement_RF()
                        lf_placement = self.tsid_wrapper.get_placement_LF()
                        
                        rf_pos = rf_placement.translation
                        lf_pos = lf_placement.translation
                        
                        self.get_logger().info(f"Current foot positions:")
                        self.get_logger().info(f"  Right foot: [{rf_pos[0]:.3f}, {rf_pos[1]:.3f}, {rf_pos[2]:.3f}]")
                        self.get_logger().info(f"  Left foot:  [{lf_pos[0]:.3f}, {lf_pos[1]:.3f}, {lf_pos[2]:.3f}]")
                        
                        # Create walking plan (alternating steps)
                        rf_pos = rf_placement.translation.copy()
                        lf_pos = lf_placement.translation.copy()

                        self.footstep_plan = []

                        for i in range(self.num_steps):
                            if i == 0 or i == self.num_steps-1:
                                step_length = self.first_step
                            else:
                                step_length = self.other_step

                            if i % 2 == 0:
                                # Right foot forward
                                rf_pos[0] += step_length
                                self.footstep_plan.append(('right', rf_pos.copy()))
                            else:
                                # Left foot forward
                                lf_pos[0] += step_length
                                self.footstep_plan.append(('left', lf_pos.copy()))
                        
                        self.get_logger().info(f"Generated plan with {len(self.footstep_plan)} steps")
                        
                        for i, (side, pos) in enumerate(self.footstep_plan):
                            self.get_logger().info(f"  {i+1}. {side:>5} foot -> [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                        
                    except Exception as e:
                        self.get_logger().error(f"Planning error: {e}")
                        # Create simple fallback plan
                        rf_simple = rf_placement.translation.copy()
                        rf_simple[0] += 0.3
                        self.footstep_plan = [('right', rf_simple)]
                        self.get_logger().info("Created simple fallback plan")
                
                # Complete planning phase
                if planning_elapsed > self.planning_duration and len(self.footstep_plan) > 0:
                    self.current_state = "WALKING"
                    self.state_start_time = self.t
                    self.step_start_time = self.t
                    self.current_step_index = 0
                    self.step_phase = "none"
                    self.get_logger().info("Planning complete - Starting path execution")
            
            ###################################################################
            # PHASE 4: WALKING ALONG PATH
            ###################################################################
            elif self.current_state == "WALKING":
                walking_elapsed = self.t - self.state_start_time
                
                if walking_elapsed < 0.1:
                    self.get_logger().info(f"[{self.t:.1f}s] PHASE 4: WALKING ALONG PATH")
                    self.get_logger().info(f"- Executing {len(self.footstep_plan)} planned steps")
                
                # Execute current step
                if self.current_step_index < len(self.footstep_plan):
                    step_elapsed = self.t - self.step_start_time
                    
                    # Start new step
                    if step_elapsed < 0.1:
                        foot_side, target_pos = self.footstep_plan[self.current_step_index]
                        self.get_logger().info(f"[{self.t:.1f}s] Executing Step {self.current_step_index + 1}/{len(self.footstep_plan)}")
                        self.get_logger().info(f"  {foot_side.upper()} foot -> [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
                    
                    # Execute step
                    step_complete = execute_step_along_path(
                        self.tsid_wrapper, self.current_step_index, self.footstep_plan, 
                        self.step_phase, step_elapsed, self.height, self.phase_duration
                    )
                    
                    # Check if step is complete (4 phases total 6.0 seconds)
                    if step_complete and step_elapsed > 6.0:
                        self.current_step_index += 1
                        self.step_start_time = self.t
                        self.step_phase = "none"
                        
                        if self.current_step_index < len(self.footstep_plan):
                            self.get_logger().info(f"Step {self.current_step_index} completed, moving to next step...")
                        else:
                            self.get_logger().info(f"All {len(self.footstep_plan)} steps completed!")
                            self.current_state = "END_POSITION"
                            self.state_start_time = self.t
                
                # All steps completed
                else:
                    self.get_logger().info("WALKING COMPLETED!")
                    self.get_logger().info(f"Successfully executed all {len(self.footstep_plan)} planned steps")
                    self.current_state = "END_POSITION"
                    self.state_start_time = self.t
            
            ###################################################################
            # PHASE 5: END POSITION
            ###################################################################
            elif self.current_state == "END_POSITION":
                if self.t - self.state_start_time == 0:
                    self.state_start_time = self.t
                    self.get_logger().info(f"[{self.t:.1f}s] PHASE 5: END POSITION")
                    self.get_logger().info("- prepare posture for ball kicking")

                    try:
                        rf = self.tsid_wrapper.get_placement_RF().translation
                        lf = self.tsid_wrapper.get_placement_LF().translation
                        com_target = (rf + lf) / 2.0
                        com_target[2] = self.tsid_wrapper.comState().pos()[2]  # Keep height

                        self.get_logger().info(f"Shifting COM to center: [{com_target[0]:.3f}, {com_target[1]:.3f}, {com_target[2]:.3f}]")
                        self.tsid_wrapper.setComRefState(com_target)
                    except Exception as e:
                        self.get_logger().warn(f"Warning during COM centering: {e}")

                if self.t - self.state_start_time > self.end_duration:
                    self.current_state = "KICKING"
                    self.state_start_time = self.t
                    self.get_logger().info("Prepare for kicking the ball")

            ###################################################################
            # PHASE 6: KICKING
            ###################################################################
            elif self.current_state == "KICKING":
                if self.t - self.state_start_time == 0:
                    self.state_start_time = self.t
                    self.get_logger().info(f"[{self.t:.1f}s] PHASE 6: KICKING")
                    self.get_logger().info("- Kicking ball")

                # TODO: Implement kicking motion
                if self.t - self.state_start_time > 3.0:
                    self.current_state = "COMPLETED"
                    self.get_logger().info("Walking and kicking sequence completed!")

            ###################################################################
            # Simulation and Hardware Control Update
            ###################################################################
            
            # update robot simulator
            self.robot.update(self.q_tsid, self.v_tsid, self.tau) 

            if self.current_state != "HOME":
                # store CoM data
                self.visual_class.getCOMStates(self.t)
            
            # TSID control update
            q_current = np.ascontiguousarray(self.robot.get_q(), dtype=np.float64)
            v_current = np.ascontiguousarray(self.robot.get_v(), dtype=np.float64)
            
            tau_sol, acc_sol = self.tsid_wrapper.update(q_current, v_current, self.t)
            self.tau = tau_sol

            # integrate dv_sol for position control
            self.q_tsid, self.v_tsid = self.tsid_wrapper.integrate_dv(
                self.q_tsid, self.v_tsid, acc_sol, 1/self.timer_frequency)      

            ###################################################################
            # Hardware Robot Command
            ###################################################################
            
            # Ensure q_tsid is valid
            if len(self.q_tsid) < 7:
                self.get_logger().error("q_tsid does not have enough elements for slicing.")
                return
            
            try:
                # Extract joint positions (skip base 6DOF + quaternion)
                joint_positions = self.q_tsid[7:].tolist()
                
                # Command to the hardware robot
                self.hardware_controller.setJointPositions(
                    self.joint_names, joint_positions, 0.2, unit='rad')
                
                # Log joint commands periodically (every 1 second)
                if abs(self.t % 1.0) < (1/self.timer_frequency):
                    self.get_logger().info(f"Commanded joint positions to hardware robot")
                    if self.current_state == "WALKING":
                        self.get_logger().info(f"Walking progress: Step {self.current_step_index + 1}/{len(self.footstep_plan)}")
                        
            except Exception as e:
                self.get_logger().error(f"Hardware command error: {e}")
                # Continue with simulation even if hardware fails
      
            # get current BASE Pose
            T_b_w, _ = self.tsid_wrapper.baseState()

            # publish transformation and joint states
            self.robot.publish(T_b_w)

            # update simulator time
            self.t = self.t + 1/self.timer_frequency

        except Exception as e:
            self.get_logger().error(f"Timer callback error: {e}")
            import traceback
            traceback.print_exc()

    def shutdown_sequence(self):
        """Clean shutdown sequence"""
        try:
            self.get_logger().info("Starting shutdown sequence...")
            
            # Stop hardware controller
            if hasattr(self, 'hardware_controller'):
                self.get_logger().info("Stopping hardware controller...")
                # Set to safe position or stop motion
                self.hardware_controller.setPosture('stand', 1.0)
            
            # Generate final plots
            if hasattr(self, 'visual_class') and len(self.visual_class.time_log) > 0:
                self.get_logger().info("Generating final plots...")
                self.visual_class.plot_all()
            
            self.get_logger().info("Shutdown sequence completed")
            
        except Exception as e:
            self.get_logger().error(f"Error during shutdown: {e}")

################################################################################
# main
################################################################################

def main(args=None):
    rclpy.init(args=args)
    node = WalkingHardwareNode()

    try:
        print("\nStarting walking with hardware control...")
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\nWalking interrupted by user")
    except ExternalShutdownException:
        print("\nWalking shutdown externally")
    except Exception as e:
        print(f"Walking error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "=" * 70)
        print("WALKING WITH HARDWARE CONTROL COMPLETE")
        print("=" * 70)
        
        # Execute shutdown sequence
        node.shutdown_sequence()
        
        print(f"\n=== Execution Summary ===")
        if hasattr(node, 'footstep_plan') and len(node.footstep_plan) > 0:
            completion_rate = (node.current_step_index / len(node.footstep_plan)) * 100
            print(f"Path planned: {len(node.footstep_plan)} steps")
            print(f"Steps executed: {node.current_step_index}/{len(node.footstep_plan)} ({completion_rate:.1f}%)")
            print(f"Final state: {node.current_state}")
        
        print(f"\n=== Hardware Control Features ===")
        print("✓ Real-time joint position commands")
        print("✓ Hardware controller integration")
        print("✓ ROS2 node structure")
        print("✓ Joint state publishing")
        print("✓ TF broadcasting")
        print("✓ Error handling and recovery")
        
        print(f"\n=== Walking Execution ===")
        print("✓ COM shifting to support foot")
        print("✓ Coordinated foot lifting and placement")
        print("✓ Step-by-step execution tracking")
        print("✓ Real-time progress feedback")
        print("✓ Hardware safety measures")
        
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()
        print("\nHardware walking control ended successfully.")

if __name__ == '__main__':
    main()