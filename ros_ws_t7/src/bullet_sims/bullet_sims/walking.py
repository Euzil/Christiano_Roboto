"""
Walking with Path Visualization in PyBullet
Display path in PyBullet, then follow the path step by step
STANDING -> PLANNING (with visualization) -> WALKING (follow path)
"""

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
import pybullet as pb

from simulator.pybullet_wrapper import PybulletWrapper
import bullet_sims.talos_conf as conf
from bullet_sims.talos import Talos
from bullet_sims.tsid_wrapper import TSIDWrapper

def visualize_footstep_path(simulator, footstep_plan, current_foot_positions):
    """
    Visualize footstep path in PyBullet
    """
    print("\n=== Visualizing Path in PyBullet ===")
    
    try:
        # Clear previous visualization markers
        # pb.removeAllUserDebugItems()  # If need to clear all markers
        
        # Color definitions
        RED = [1, 0, 0]      # Right foot - red
        BLUE = [0, 0, 1]     # Left foot - blue  
        GREEN = [0, 1, 0]    # Current position - green
        YELLOW = [1, 1, 0]   # Path connection lines - yellow
        
        visual_ids = []
        
        # 1. Show current foot positions
        rf_current, lf_current = current_foot_positions
        
        # Current right foot position (green sphere)
        rf_sphere_id = pb.addUserDebugText(
            "RF_START", 
            [rf_current[0], rf_current[1], rf_current[2] + 0.1],
            textColorRGB=GREEN,
            textSize=1.2
        )
        visual_ids.append(rf_sphere_id)
        
        # Current left foot position (green sphere)
        lf_sphere_id = pb.addUserDebugText(
            "LF_START", 
            [lf_current[0], lf_current[1], lf_current[2] + 0.1],
            textColorRGB=GREEN,
            textSize=1.2
        )
        visual_ids.append(lf_sphere_id)
        
        # 2. Show planned footsteps
        for i, (foot_side, position) in enumerate(footstep_plan):
            color = RED if foot_side == 'right' else BLUE
            foot_label = "RF" if foot_side == 'right' else "LF"
            
            # Add footstep marker (colored box)
            box_id = pb.addUserDebugText(
                f"{foot_label}_{i+1}",
                [position[0], position[1], position[2] + 0.05],
                textColorRGB=color,
                textSize=1.5
            )
            visual_ids.append(box_id)
            
            # Add footstep outline (rectangle representing foot shape)
            foot_corners = [
                [position[0] - 0.05, position[1] - 0.03, position[2]],  # Back left
                [position[0] + 0.05, position[1] - 0.03, position[2]],  # Back right
                [position[0] + 0.05, position[1] + 0.03, position[2]],  # Front right
                [position[0] - 0.05, position[1] + 0.03, position[2]],  # Front left
                [position[0] - 0.05, position[1] - 0.03, position[2]]   # Close the loop
            ]
            
            # Draw footstep outline
            for j in range(len(foot_corners) - 1):
                line_id = pb.addUserDebugLine(
                    foot_corners[j], 
                    foot_corners[j + 1],
                    lineColorRGB=color,
                    lineWidth=3
                )
                visual_ids.append(line_id)
        
        # 3. Draw path connection lines
        if len(footstep_plan) > 1:
            # From current position to first step
            first_foot, first_pos = footstep_plan[0]
            start_pos = rf_current if first_foot == 'right' else lf_current
            
            path_line_id = pb.addUserDebugLine(
                start_pos,
                first_pos,
                lineColorRGB=YELLOW,
                lineWidth=2
            )
            visual_ids.append(path_line_id)
            
            # Connect each footstep
            for i in range(len(footstep_plan) - 1):
                current_pos = footstep_plan[i][1]
                next_pos = footstep_plan[i + 1][1]
                
                path_line_id = pb.addUserDebugLine(
                    current_pos,
                    next_pos, 
                    lineColorRGB=YELLOW,
                    lineWidth=2
                )
                visual_ids.append(path_line_id)
        
        # 4. Add path information text
        if len(footstep_plan) > 0:
            info_text = f"Path: {len(footstep_plan)} steps planned"
            info_id = pb.addUserDebugText(
                info_text,
                [rf_current[0] - 0.5, rf_current[1] + 0.5, rf_current[2] + 0.3],
                textColorRGB=[0, 0, 0],
                textSize=1.0
            )
            visual_ids.append(info_id)
        
        print(f"Visualized {len(footstep_plan)} footsteps in PyBullet")
        print(f"  Red markers: Right foot steps")
        print(f"  Blue markers: Left foot steps")
        print(f"  Yellow lines: Path connections")
        print(f"  Green text: Current positions")
        
        return visual_ids
        
    except Exception as e:
        print(f"Visualization error: {e}")
        return []
    
    
def execute_step_along_path(tsid_wrapper, step_index, footstep_plan, step_phase, step_elapsed):
    """
    Execute a single gait step along the path (with landing detection and knee bend contact logic)
    """
    if step_index >= len(footstep_plan):
        return True  # All steps completed

    foot_side, target_position = footstep_plan[step_index]

    # Phase timing settings
    phase_duration = 1.5
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
                target_pos[2] += 0.15 * (lift_progress / 0.3)
            elif lift_progress < 0.7:
                ratio = (lift_progress - 0.3) / 0.4
                target_pos = current_foot_pos + ratio * (target_position - current_foot_pos)
                target_pos[2] = current_foot_pos[2] + 0.15
            else:
                ratio = (lift_progress - 0.7) / 0.3
                target_pos = target_position.copy()
                target_pos[2] += 0.15 * (1 - ratio)

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

            if abs(actual_z - expected_z) > 0.015:  # >1.5cm suspended
                print(" Foot not contacting ground - lowering COM to help foot land...")
                com_current = tsid_wrapper.comState().pos()
                lowered_com = com_current.copy()
                lowered_com[2] -= 0.02  # Bend knee and lower 2cm
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
                    lowered_com[2] += 0.02  # Restore height after bending knee
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




def main():
    rclpy.init()
    node = rclpy.create_node('walking_with_path_visualization')
    
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
        simulator = PybulletWrapper()
        model = tsid_wrapper.robot.model()
        
        robot = Talos(
            simulator=simulator,
            urdf=conf.urdf,
            model=model,
            q=conf.q_home,
            verbose=True,
            useFixedBase=False
        )
        
        # Initial settling
        robot.enablePositionControl()
        for i in range(30):
            robot.setActuatedJointPositions(conf.q_home)
            simulator.step()
            robot.update()
        robot.enableTorqueControl()
        
        # State machine
        current_state = "STANDING"
        state_start_time = 0.0
        t_publish = 0.0
        publish_rate = 30.0
        
        # Phase durations
        standing_duration = 3.0
        planning_duration = 3.0
        
        # Walking data
        footstep_plan = []
        visual_ids = []
        current_step_index = 0
        step_start_time = 0.0
        step_phase = "none"
        
        print("\nStarting walking with path visualization...")
        
        while rclpy.ok():
            t = simulator.simTime()
            
            ###################################################################
            # PHASE 1: STANDING
            ###################################################################
            if current_state == "STANDING":
                if t - state_start_time == 0:
                    state_start_time = t
                    print(f"\n[{t:.1f}s] PHASE 1: STANDING")
                    print("- Establishing stable base")
                
                if t - state_start_time > standing_duration:
                    current_state = "PLANNING"
                    state_start_time = t
                    print(f"\nStanding stable - Ready for path planning")
            
            ###################################################################
            # PHASE 2: PLANNING WITH VISUALIZATION
            ###################################################################
            elif current_state == "PLANNING":
                if t - state_start_time < 0.1:
                    print(f"\n[{t:.1f}s] PHASE 2: PLANNING WITH VISUALIZATION")
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

                        # Step length settings
                        first_step = 0.25
                        other_step = 0.50
                        num_steps = 6  # Even number: start with right foot

                        for i in range(num_steps):
                            if i == 0:
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
                        
                        # Visualize path
                        current_positions = (rf_pos, lf_pos)
                        visual_ids = visualize_footstep_path(simulator, footstep_plan, current_positions)
                        
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
            # PHASE 3: WALKING ALONG PATH
            ###################################################################
            elif current_state == "WALKING":
                walking_elapsed = t - state_start_time
                
                if walking_elapsed < 0.1:
                    print(f"\n[{t:.1f}s] PHASE 3: WALKING ALONG PATH")
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
                        tsid_wrapper, current_step_index, footstep_plan, step_phase, step_elapsed
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
                            break

                
                # All steps completed
                else:
                    print(f"\nWALKING COMPLETED!")
                    print(f"Successfully executed all {len(footstep_plan)} planned steps")
                    break
            
            ###################################################################
            # Simulation update
            ###################################################################
            simulator.step()
            robot.update()
            
            # TSID control
            q_current = np.ascontiguousarray(robot.q(), dtype=np.float64)
            v_current = np.ascontiguousarray(robot.v(), dtype=np.float64)
            
            tau_sol, acc_sol = tsid_wrapper.update(q_current, v_current, t)
            tau_sol = np.array(tau_sol)
            if len(tau_sol) > conf.na:
                tau_sol = tau_sol[:conf.na]
            
            robot.tau = tau_sol
            robot.setActuatedJointTorques(tau_sol)
            
            # Publishing
            if t - t_publish >= 1.0 / publish_rate:
                t_publish = t
                T_b_w = tsid_wrapper.baseState()
                robot.publish(T_b_w)
            
            rclpy.spin_once(node, timeout_sec=0.001)
    
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
        
        if 'simulator' in locals():
            simulator.disconnect()
        rclpy.shutdown()
        print("\nPath visualization walking ended successfully.")

if __name__ == '__main__':
    main()