"""
Humanoid Robot Walking Controller with Ball Kicking Capability

This module implements a comprehensive walking controller for the AiNex humanoid robot
featuring a complete motion sequence from standing to ball kicking with celebration.

Features:
- Multi-phase state machine (Home → Standing → Planning → Walking → End → Kicking → Celebration)
- TSID-based trajectory optimization and whole-body control
- Real-time footstep planning and path execution
- Coordinated ball kicking motion with recovery
- Hardware integration with motion primitives
- Comprehensive logging and real-time feedback

Architecture:
- Uses Task Space Inverse Dynamics (TSID) for control
- Integrates with ROS2 for robot communication
- Employs Pinocchio for rigid body dynamics

Author: Theresa Gräbner, Djamal Halim, Youran Wang
Institution: ICS, TU Munich 
Project: Christiano Roboto - AiNex Humanoid Robot Walking Control
Date: 29.07.2025
Version: 1.0

Dependencies:
- ROS2 (Robot Operating System)
- Pinocchio (Rigid body dynamics library)
- TSID (Task Space Inverse Dynamics)
- AiNex motion controller interface
"""


import numpy as np
import pinocchio as pin
import rclpy
import matplotlib.pyplot as plt
import os
import time 

import whole_body_control.config as conf
from whole_body_control.ainex import Ainex
from whole_body_control.tsid_wrapper import TSIDWrapper
from ament_index_python.packages import get_package_share_directory

import matplotlib.pyplot as plt
from ainex_motion.joint_controller import JointController


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
                axs_pos[i],
                self.time_log,
                [x[i] for x in self.com_ref_log],
                [x[i] for x in self.com_state_log],
                title=f"COM Position - {['X','Y','Z'][i]}"
            )

            self.update_subplot_com(
                axs_vel[i],
                self.time_log,
                [x[i] for x in self.com_vel_ref_log],
                [x[i] for x in self.com_vel_state_log],
                title=f"COM Velocity - {['X','Y','Z'][i]}"
            )

            self.update_subplot_com(
                axs_acc[i],
                self.time_log,
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
        current_dir = os.path.dirname(os.path.realpath(__file__))

        # store results
        fig_pos.savefig(os.path.join(current_dir, "com_position_plot.png"), dpi=300)
        fig_vel.savefig(os.path.join(current_dir, "com_velocity_plot.png"), dpi=300)
        fig_acc.savefig(os.path.join(current_dir, "com_acceleration_plot.png"), dpi=300)

################################################################################
# Method for executing steps
################################################################################

def execute_step_along_path(tsid_wrapper, step_index, footstep_plan, step_phase, step_elapsed, height, phase_duration, shift_com_x, shift_com_y):
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

                sign = -1 if foot_side == 'left' else 1

                if foot_side == 'right':
                    support_pos = tsid_wrapper.get_placement_LF().translation
                else:
                    support_pos = tsid_wrapper.get_placement_RF().translation

                com_current = tsid_wrapper.comState().pos()
                p_com_new = np.array([support_pos[0] + shift_com_x, support_pos[1] + sign*shift_com_y, com_current[2]])
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

            if abs(actual_z - expected_z) > 0.015:  # >0.4cm suspended
                print(" Foot not contacting ground - lowering COM to help foot land...")
                #com_current = tsid_wrapper.comState().pos()
                #lowered_com = com_current.copy()
                #lowered_com[2] -= 0.02  # Bend knee and lower 2cm
                #tsid_wrapper.setComRefState(lowered_com)

        # Phase 4: After successful landing, add contact -> COM transfer
        elif step_elapsed < shift_back_phase:
            if step_phase != "shift_back":
                actual_pos = tsid_wrapper.get_placement_RF().translation if foot_side == 'right' \
                             else tsid_wrapper.get_placement_LF().translation

                if abs(actual_pos[2] - target_position[2]) < 0.01:  # Already landed
                    print(f"  Step {step_index + 1}: COM shifting to newly placed {foot_side} foot...")
                    step_phase = "shift_back"

                    sign = -1 if foot_side == 'left' else 1

                    if step_index == len(footstep_plan)-2:
                        sign = 0

                    # Add contact
                    if foot_side == 'right' and hasattr(tsid_wrapper, 'add_contact_RF'):
                        tsid_wrapper.add_contact_RF()
                    elif foot_side == 'left' and hasattr(tsid_wrapper, 'add_contact_LF'):
                        tsid_wrapper.add_contact_LF()

                    # Transfer COM to this foot
                    com_current = tsid_wrapper.comState().pos()
                    p_com_new = np.array([target_position[0] + shift_com_x, target_position[1] + sign*shift_com_y, com_current[2]])
                    tsid_wrapper.setComRefState(p_com_new)
                    
                    #com_current = tsid_wrapper.comState().pos()
                    #lowered_com = com_current.copy()
                    #lowered_com[2] += 0.0  # Restore height after bending knee
                    #tsid_wrapper.setComRefState(lowered_com)
                    
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
# main walking controller managing everything
################################################################################

def main(args=None):
    rclpy.init()
    node = rclpy.create_node('walking_with_path_visualization')
    
    joint_names = ['head_pan', 'head_tilt',
                   'l_hip_yaw', 'l_hip_roll', 'l_hip_pitch', 'l_knee', 'l_ank_pitch', 'l_ank_roll',
                   'l_sho_pitch', 'l_sho_roll', 'l_el_pitch', 'l_el_yaw', 'l_gripper',
                   'r_hip_yaw', 'r_hip_roll', 'r_hip_pitch', 'r_knee', 'r_ank_pitch', 'r_ank_roll',
                   'r_sho_pitch', 'r_sho_roll', 'r_el_pitch', 'r_el_yaw', 'r_gripper']
    
    q_celebration = np.zeros(24)
    q_celebration[:2] = np.array([0.0, 0.4])                                # head (head pan, head tilt)
    q_celebration[2:8] = np.array([0.0, -0.2, -0.2, 0.4, 0.20, -0.2])         # left leg (l_hip_yaw, l_hip_roll, l_hip_pitch, ,l_knee, l_ank_pitch, l_ank_roll)
    q_celebration[8:13] = np.array([0.5, 0.5, 0.0, 0, 0.0])                 # left arm (l_sho_pitch, l_sho_roll, l_el_pitch, l_el_yaw, l_gripper)
    q_celebration[13:19] = np.array([0.0, 0.2, 0.2, -0.4, -0.20, 0.2])      # right leg (r_hip_yaw, r_hip_roll, r_hip_pitch, r_knee, r_ank_pitch, r_ank_roll)
    q_celebration[19:24] = np.array([-0.5, -0.5, 0.0, 0, 0.0])              # right arm (r_sho_pitch, r_sho_roll, r_el_pitch, r_el_yaw, r_gripper)
    
    hardware_controller = JointController(node)

    starting_joint_config = conf.q_actuated_home
    # Hardware controller activation flag
    hardware_control_active = False
    hardware_start_time = 2  # 2 seconds before starting hardware control
    hardware_controller.setPosture('stand', 2)
    time.sleep(3)
    hardware_controller.setJointPositions(joint_names, starting_joint_config, 1 , unit='rad')

    try:
        print("=" * 70)
        print("WALKING WITH PATH VISUALIZATION IN PYBULLET")
        print("=" * 70)
        print("Features:")
        print("- Visual footstep path in PyBullet")
        print("- Color-coded foot markers (Red=Right, Blue=Left)")
        print("- Step-by-step execution following the path")
        print("- Real-time progress feedback")
        print("=" * 70)
        
        # Initialize system
        tsid_wrapper = TSIDWrapper(conf)
        
        # initial hight
        z_init = 0.23

        # init Simulator
        q_init = np.hstack([np.array([0, 0, z_init, 0, 0, 0, 1]), np.zeros((conf.na,))])

        robot = Ainex(q=q_init)

        # initialize visualization
        visual_class = Visualization(tsid_wrapper)
        
        # State machine
        current_state = "HOME"
        state_start_time = 0.0

        # Walking data
        footstep_plan = []
        visual_ids = []
        current_step_index = 0
        step_start_time = 0.0
        step_phase = "none"

        # get q and v
        q_tsid = robot.get_q()
        v_tsid = robot.get_v()

        # time
        t = 0.0

        ################################################################################
        # Parameter to adjust frequency in which the new joint positions are published 
        # and commanded to the real hardware robot AiNex Christiano Roboto
        ################################################################################

        # time frequency
        timer_frequency = 30

        ################################################################################
        # Parameter to adjust walking motion
        ################################################################################
        
        # Phase durations
        home_duration = 2.0
        standing_duration = 1.5
        planning_duration = 3.0
        end_duration = 4.0
        
        # Step length settings
        first_step = 0.065      # and used for last step
        other_step = 0.13       # used for every step except first and last one
        num_steps = 20          # Even number: start with right foot
        height = 0.03           # max step height

        # how long one step takes
        phase_duration = 1.5

        # shift CoM
        shift_com_x = 0.03
        shift_com_y = 0.005
        shift_com_z = -0.02

        print("\nStarting walking with path visualization...")
        # bool values for celebration phase
        celebration_active = False 
        move_1_finished = False 
        move_2_finished = False
        move_3_finished = False

        while rclpy.ok():
            
            ###################################################################
            # PHASE 1: GET INTO HOME POSITION
            ###################################################################
            if current_state == "HOME":
                if t - state_start_time == 0:
                    state_start_time = t
                    print(f"\n[{t:.1f}s] PHASE 1: HOME")
                
                if t - state_start_time > home_duration:
                    current_state = "STANDING"
                    state_start_time = t
                    print(f"\nIn home position")

            ###################################################################
            # PHASE 2: STANDING
            ###################################################################
            if current_state == "STANDING":
                if t - state_start_time == 0:
                    state_start_time = t
                    print(f"\n[{t:.1f}s] PHASE 2: STANDING")
                    print("- Establishing stable base")

                    try:
                        rf = tsid_wrapper.get_placement_RF().translation
                        lf = tsid_wrapper.get_placement_LF().translation
                        com_target = (rf + lf) / 2.0
                        com_target[0] = com_target[0] + shift_com_x  
                        com_target[2] = tsid_wrapper.comState().pos()[2] + shift_com_z 

                        print(f"Shifting COM to center: [{com_target[0]:.3f}, {com_target[1]:.3f}, {com_target[2]:.3f}]")
                        tsid_wrapper.setComRefState(com_target)
                    except Exception as e:
                        print(f"  Warning during COM centering: {e}")
                
                if t - state_start_time > standing_duration:
                    current_state = "PLANNING"
                    state_start_time = t
                    print(f"\nStanding stable - Ready for path planning")
            
            ###################################################################
            # PHASE 3: PLANNING WITH VISUALIZATION
            ###################################################################
            elif current_state == "PLANNING":
                if t - state_start_time < 0.1:
                    print(f"\n[{t:.1f}s] PHASE 3: PLANNING WITH VISUALIZATION")
                    print("- Generating footstep path")
                    print("- Visualizing in PyBullet")
                
                planning_elapsed = t - state_start_time
                
                # Generate footstep plan
                if len(footstep_plan) == 0 and planning_elapsed > 0.5:
                    try:
                        print(f"[{t:.1f}s] Generating footstep plan...")
                        
                        # Get current foot positions
                        rf_placement = tsid_wrapper.get_placement_RF()
                        lf_placement = tsid_wrapper.get_placement_LF()
                        
                        rf_pos = rf_placement.translation
                        lf_pos = lf_placement.translation
                        
                        print(f"Current foot positions:")
                        print(f"  Right foot: [{rf_pos[0]:.3f}, {rf_pos[1]:.3f}, {rf_pos[2]:.3f}]")
                        print(f"  Left foot:  [{lf_pos[0]:.3f}, {lf_pos[1]:.3f}, {lf_pos[2]:.3f}]")
                        
                        # Create walking plan (alternating steps)
                        rf_pos = rf_placement.translation.copy()
                        lf_pos = lf_placement.translation.copy()

                        rf_pos = rf_placement.translation.copy()
                        lf_pos = lf_placement.translation.copy()

                        footstep_plan = []

                        for i in range(num_steps):
                            if i == 0 or i == num_steps-1:
                                step_length = first_step
                            else:
                                step_length = other_step

                            if i % 2 == 0:
                                # Right foot forward
                                rf_pos[0] += step_length
                                footstep_plan.append(('right', rf_pos.copy()))
                            else:
                                # Left foot forward
                                lf_pos[0] += step_length
                                footstep_plan.append(('left', lf_pos.copy()))
                        
                        print(f"Generated plan with {len(footstep_plan)} steps")
                        
                        print(f"\nStep sequence:")
                        for i, (side, pos) in enumerate(footstep_plan):
                            print(f"  {i+1}. {side:>5} foot -> [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                        
                    except Exception as e:
                        print(f"Planning error: {e}")
                        # Create simple fallback plan
                        rf_simple = rf_placement.translation.copy()
                        rf_simple[0] += 0.3
                        footstep_plan = [('right', rf_simple)]
                        print("Created simple fallback plan")
                
                # Complete planning phase
                if planning_elapsed > planning_duration and len(footstep_plan) > 0:
                    current_state = "WALKING"
                    state_start_time = t
                    step_start_time = t
                    current_step_index = 0
                    step_phase = "none"
                    print(f"\nPlanning complete - Starting path execution")
                    print(f"Path visualized with {len(visual_ids)} markers")
            
            ###################################################################
            # PHASE 4: WALKING ALONG PATH
            ###################################################################
            elif current_state == "WALKING":
                walking_elapsed = t - state_start_time
                
                if walking_elapsed < 0.1:
                    print(f"\n[{t:.1f}s] PHASE 4: WALKING ALONG PATH")
                    print(f"- Executing {len(footstep_plan)} planned steps")
                    print("- Following visualized path")
                
                # Execute current step
                if current_step_index < len(footstep_plan):
                    step_elapsed = t - step_start_time
                    
                    # Start new step
                    if step_elapsed < 0.1:
                        foot_side, target_pos = footstep_plan[current_step_index]
                        print(f"\n[{t:.1f}s] Executing Step {current_step_index + 1}/{len(footstep_plan)}")
                        print(f"  {foot_side.upper()} foot -> [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
                    
                    # Execute step
                    step_complete = execute_step_along_path(
                        tsid_wrapper, current_step_index, footstep_plan, step_phase, step_elapsed, height, phase_duration, shift_com_x, shift_com_y
                    )
                    
                    # Check if step is complete
                    # Step completion criteria: 4 phases total 6.0 seconds
                    if step_complete and step_elapsed > 6.0:
                        current_step_index += 1
                        step_start_time = t
                        step_phase = "none"
                        
                        if current_step_index < len(footstep_plan):
                            print(f"\nStep {current_step_index} completed, moving to next step...")
                            
                        else:
                            print(f"\nAll {len(footstep_plan)} steps completed!")
                            current_state = "END POSITION"
                            state_start_time = t
                            print(f"\nGo into end position")
                
                # All steps completed
                else:
                    print(f"\nWALKING COMPLETED!")
                    print(f"Successfully executed all {len(footstep_plan)} planned steps")

                    current_state = "END POSITION"
                    state_start_time = t
                    print(f"\nGo into end position")
            
            ###################################################################
            # PHASE 5: END POSITION
            ###################################################################
            if current_state == "END POSITION":
                if t - state_start_time == 0:
                    state_start_time = t
                    print(f"\n[{t:.1f}s] PHASE 5: END POSITION")
                    print("- prepare posture for ball kicking")

                    try:
                        rf = tsid_wrapper.get_placement_RF().translation
                        lf = tsid_wrapper.get_placement_LF().translation
                        com_target = (rf + lf) / 2.0
                        com_target[0] = com_target[0] + shift_com_x  
                        com_target[2] = tsid_wrapper.comState().pos()[2] + (-1) * shift_com_z  # Höhe beibehalten

                        print(f"Shifting COM to center: [{com_target[0]:.3f}, {com_target[1]:.3f}, {com_target[2]:.3f}]")
                        tsid_wrapper.setComRefState(com_target)
                    except Exception as e:
                        print(f"  Warning during COM centering: {e}")

                if t - state_start_time > end_duration:
                    current_state = "KICKING"
                    state_start_time = t
                    print(f"\nPrepare for kicking the ball")

            ###################################################################
            # PHASE 6: KICKING
            ###################################################################
            if current_state == "KICKING":
                if t - state_start_time == 0:
                    state_start_time = t
                    print(f"\n[{t:.1f}s] PHASE 6: KICKING")
                    print("- Performing kicking motion")
                
                kicking_elapsed = t - state_start_time
                
                # Kicking motion parameters
                prepare_duration = 1.5      # Preparation phase
                swing_duration = 1.0        # Leg swing phase
                strike_duration = 0.6       # Ball strike phase
                recovery_duration = 1.5     # Recovery phase
                total_kick_duration = prepare_duration + swing_duration + strike_duration + recovery_duration
                
                try:
                    # Phase 6.1: Prepare for kick (shift COM to support foot)
                    if kicking_elapsed < prepare_duration:
                        if kicking_elapsed < 0.1:
                            print(f"  Preparing for kicking motion")
                        
                        # Shift COM to support foot (left foot)
                        lf_pos = tsid_wrapper.get_placement_LF().translation
                        com_current = tsid_wrapper.comState().pos()
                        
                        # Shift COM towards support foot
                        prepare_progress = kicking_elapsed / prepare_duration
                        
                        com_target = np.array([
                            lf_pos[0] + shift_com_x, 
                            lf_pos[1],  # Move COM above support foot
                            com_current[2]  # Maintain height
                        ])
                        tsid_wrapper.setComRefState(com_target)
                    
                    # Phase 6.2: Swing back (backswing)
                    elif kicking_elapsed < prepare_duration + swing_duration:
                        swing_elapsed = kicking_elapsed - prepare_duration
                        if swing_elapsed < 0.1:
                            print(f"  Swinging right foot back")
                            # Remove contact constraint from kicking foot
                            if hasattr(tsid_wrapper, 'remove_contact_RF'):
                                tsid_wrapper.remove_contact_RF()
                        
                        swing_progress = swing_elapsed / swing_duration
                        
                        # Get initial position of kicking foot
                        if swing_elapsed < 0.1:
                            # Record initial position for backswing calculation
                            initial_rf_pos = tsid_wrapper.get_placement_RF().translation.copy()
                        else:
                            # Use recorded initial position
                            if not hasattr(tsid_wrapper, '_initial_rf_pos'):
                                tsid_wrapper._initial_rf_pos = tsid_wrapper.get_placement_RF().translation.copy()
                            initial_rf_pos = tsid_wrapper._initial_rf_pos
                        
                        # Backswing trajectory
                        swing_back_distance = 0.12  # Swing back 12cm
                        swing_up_height = 0.06      # Lift up 6cm
                        
                        target_pos = initial_rf_pos.copy()
                        target_pos[0] -= swing_back_distance * swing_progress  # Swing backward
                        target_pos[2] += swing_up_height * swing_progress      # Lift up
                        
                        # Set kicking foot position
                        foot_pose = pin.SE3(np.eye(3), target_pos)
                        if hasattr(tsid_wrapper, 'set_RF_pose_ref'):
                            tsid_wrapper.set_RF_pose_ref(foot_pose)
                    
                    # Phase 6.3: Strike (forward kick)
                    elif kicking_elapsed < prepare_duration + swing_duration + strike_duration:
                        strike_elapsed = kicking_elapsed - prepare_duration - swing_duration
                        if strike_elapsed < 0.1:
                            print(f"  Executing forward kick")
                        
                        strike_progress = strike_elapsed / strike_duration
                        
                        # Get backswing end position
                        if not hasattr(tsid_wrapper, '_swing_end_pos'):
                            tsid_wrapper._swing_end_pos = tsid_wrapper.get_placement_RF().translation.copy()
                        swing_end_pos = tsid_wrapper._swing_end_pos
                        
                        # Forward kick trajectory
                        strike_forward_distance = 0.32  # Kick forward 32cm
                        
                        # Use smooth kicking trajectory
                        target_pos = swing_end_pos.copy()
                        target_pos[0] += strike_forward_distance * strike_progress  # Forward kick
                        target_pos[2] = swing_end_pos[2] - 0.04 * strike_progress  # Slightly descend
                        
                        # Set kicking foot position
                        foot_pose = pin.SE3(np.eye(3), target_pos)
                        if hasattr(tsid_wrapper, 'set_RF_pose_ref'):
                            tsid_wrapper.set_RF_pose_ref(foot_pose)
                        
                        # Output information during kicking motion
                        if strike_progress > 0.5 and strike_elapsed < 0.2:
                            print(f"  Kicking motion executed")
                    
                    # Phase 6.4: Recovery (recovery)
                    elif kicking_elapsed < total_kick_duration:
                        recovery_elapsed = kicking_elapsed - prepare_duration - swing_duration - strike_duration
                        if recovery_elapsed < 0.1:
                            print(f"  Recovering to stable stance")
                        
                        recovery_progress = recovery_elapsed / recovery_duration
                        
                        # Get current right foot position
                        rf_pos = tsid_wrapper.get_placement_RF().translation
                        lf_pos = tsid_wrapper.get_placement_LF().translation
                        
                        # Calculate recovery target position
                        recovery_target = lf_pos.copy()
                        recovery_target[0] = lf_pos[0]      # Same line as left foot
                        recovery_target[1] = lf_pos[1] - 0.18  # Right foot position (shoulder width)
                        recovery_target[2] = lf_pos[2]      # Same height
                        
                        # Smooth transition to recovery position
                        target_pos = rf_pos + recovery_progress * (recovery_target - rf_pos)
                        
                        foot_pose = pin.SE3(np.eye(3), target_pos)
                        if hasattr(tsid_wrapper, 'set_RF_pose_ref'):
                            tsid_wrapper.set_RF_pose_ref(foot_pose)
                        
                        # Restore foot contact
                        if recovery_progress > 0.6:
                            if hasattr(tsid_wrapper, 'add_contact_RF'):
                                tsid_wrapper.add_contact_RF()
                        
                        # Return COM to center between both feet
                        if recovery_progress > 0.4:
                            rf = tsid_wrapper.get_placement_RF().translation
                            lf = tsid_wrapper.get_placement_LF().translation
                            com_target = (rf + lf) / 2.0
                            com_target[0] = com_target[0] + shift_com_x
                            com_target[2] = tsid_wrapper.comState().pos()[2]
                            
                            tsid_wrapper.setComRefState(com_target)
                    
                    # Kicking completed
                    else:
                        if kicking_elapsed > total_kick_duration and kicking_elapsed < total_kick_duration + 0.1:
                            print(f"\nKicking motion completed")
                            current_state = "CELEBRATION"
                            state_start_time = t
                        
                        if kicking_elapsed > total_kick_duration + 2.0:
                            print(f"\nWalking and kicking sequence finished")
                            break
                
                except Exception as e:
                    print(f"Kicking error: {e}")
                    break
                
            ###################################################################
            # PHASE 7: CELEBRATION
            ###################################################################
            
            elif current_state == "CELEBRATION":
                if t - state_start_time == 0:
                    state_start_time = t
                    print(f"\n[{t:.1f}s] PHASE 7: CELEBRATION")
                    print("- Disabling TSID control for celebration")
                    hardware_control_active = False  # ❌ Disable TSID hardware control
                
                celebration_elapsed = t - state_start_time
                
                try:
                    # Celebration sequence with timed phases
                    crouch_duration = 2.0
                    stand_duration = 2.0  
                    victory_pose_duration = 5.0
                    total_celebration_duration = crouch_duration + stand_duration + victory_pose_duration
                    
                    # Phase 7.1: Crouch down
                    if celebration_elapsed < crouch_duration and not move_1_finished:
                        if celebration_elapsed < 0.1:
                            print(f"  Crouching down...")
                            hardware_controller.setPosture('crouch', 0.8)
                            move_1_finished = True
                            time.sleep(2)  # Wait for crouch to stabilize
                    
                    # Phase 7.2: Stand up
                    elif celebration_elapsed < crouch_duration + stand_duration and not move_2_finished:
                        stand_elapsed = celebration_elapsed - crouch_duration
                        if stand_elapsed < 0.1:
                            print(f"  Standing up...")
                            hardware_controller.setPosture('stand', 0.8)
                            move_2_finished = True
                            time.sleep(2)
                    
                    # Phase 7.3: Victory pose
                    elif celebration_elapsed < total_celebration_duration and not move_3_finished:
                        victory_elapsed = celebration_elapsed - crouch_duration - stand_duration
                        if victory_elapsed < 0.1:
                            print(f"  Victory pose - arms up!")
                            hardware_controller.setJointPositions(joint_names, q_celebration, 0.6, unit='rad')
                            celebration_active = True
                            move_3_finished = True
                            time.sleep(2)
                    
                    # Celebration completed
                    else:
                        if celebration_elapsed > total_celebration_duration and celebration_elapsed < total_celebration_duration + 0.1:
                            print(f"\n🎉 CELEBRATION COMPLETE! 🎉")
                            print("Walking and kicking sequence finished successfully!")
                            break
                
                except Exception as e:
                    print(f"Celebration error: {e}")
                    break

            ###################################################################
            # Simulation update
            ###################################################################
            
            robot.update(q_tsid, v_tsid)

            if current_state != "HOME" :

                # store CoM data
                visual_class.getCOMStates(t)
            
            # TSID control
            q_current = np.ascontiguousarray(robot.get_q(), dtype=np.float64)
            v_current = np.ascontiguousarray(robot.get_v(), dtype=np.float64)
            
            tau_sol, acc_sol = tsid_wrapper.update(q_current, v_current, t)

            # integrate dv_sol for position control
            q_tsid, v_tsid = tsid_wrapper.integrate_dv(q_tsid, v_tsid, acc_sol, 1/timer_frequency)
            
            # TODO:command to the hardware robot - should have reached q_tsid for next timer call
            if len(q_tsid) < 7:
                node.get_logger().error("q_tsid does not have enough elements for slicing.")
                continue  # Use continue instead of return
            
            # Hardware control activation check
            if not hardware_control_active and t >= hardware_start_time:
                hardware_control_active = True
                print(f"\n[{t:.1f}s] Hardware control activated - Starting joint position commands")
            
            # Only send TSID joint commands if NOT in celebration
            if hardware_control_active and current_state != "CELEBRATION":
                # Convert numpy array slice to list for setJointPositions
                joint_positions = q_tsid[7:].tolist()
                
                # Command to the hardware robot
                hardware_controller.setJointPositions(joint_names, joint_positions, 0.03, unit='rad')
            elif current_state == "CELEBRATION":
                # During celebration, don't send TSID commands
                pass
            else:
                # Optional: Log that hardware control is waiting
                if int(t * 10) % 10 == 0:  # Log every 1 second
                    remaining_time = hardware_start_time - t
                    print(f"[{t:.1f}s] Hardware control starts in {remaining_time:.1f}s...")
            
            # get current BASE Pose
            T_b_w, _ = tsid_wrapper.baseState()
            
            # Publishing
            robot.publish(T_b_w)

            # adjust time value
            t = t + 1/timer_frequency
            
            rclpy.spin_once(node, timeout_sec=1/timer_frequency)
    
    except KeyboardInterrupt:
        print("\nWalking interrupted")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "=" * 70)
        print("WALKING WITH PATH VISUALIZATION COMPLETE")
        print("=" * 70)
        
        print(f"\n=== Execution Summary ===")
        if len(footstep_plan) > 0:
            completion_rate = (current_step_index / len(footstep_plan)) * 100
            print(f"Path planned: {len(footstep_plan)} steps")
            print(f"Steps executed: {current_step_index}/{len(footstep_plan)} ({completion_rate:.1f}%)")
            print(f"Visualization markers: {len(visual_ids)}")
        
        print(f"\n=== Path Visualization Features ===")
        print("Real-time footstep markers in PyBullet")
        print("Color-coded foot identification")
        print("Path connection lines")
        print("Step-by-step execution tracking")
        
        print(f"\n=== Walking Execution ===")
        print("COM shifting to support foot")
        print("Coordinated foot lifting and placement")
        print("Following pre-planned path precisely")
        print("Real-time progress feedback")
        
        # visualize results
        # visual_class.plot_all()

        rclpy.shutdown()
        print("\nPath visualization walking ended successfully.")

if __name__ == '__main__':
    main()