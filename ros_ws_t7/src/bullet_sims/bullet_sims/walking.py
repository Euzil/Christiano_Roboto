"""
talos walking simulation
"""

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node

# simulator
from simulator.pybullet_wrapper import PybulletWrapper

# robot configs
import bullet_sims.talos_conf as conf

# modules
from bullet_sims.talos import Talos
from bullet_sims.footstep_planner import FootStepPlanner, Side
from bullet_sims.lip_mpc import LIPMPC
from bullet_sims.lip_mpc import LIPInterpolator
from bullet_sims.foot_trajectory import SwingFootTrajectory

################################################################################
# main
################################################################################  
    
def main(): 
    
    ############################################################################
    # setup
    ############################################################################
    
    # setup ros
    rclpy.init()
    node = rclpy.create_node('talos_walking')
    print("Starting Talos Walking Simulation")
    print("="*50)
    
    # setup the simulator
    print("Setting up PyBullet simulator...")
    simulator = PybulletWrapper()
    
    # setup the robot
    print("Setting up Talos robot...")
    robot = Talos(simulator)
    print(f"   Robot created with floating base")
    
    # initial footsteps
    print("Getting initial foot poses...")
    T_swing_w = robot.stack.getFramePose(conf.lf_frame_name)    # set initial swing foot pose to left foot
    T_support_w = robot.stack.getFramePose(conf.rf_frame_name)  # set initial support foot pose to right foot
    
    print(f"   Left foot (swing):   [{T_swing_w[0,3]:.3f}, {T_swing_w[1,3]:.3f}, {T_swing_w[2,3]:.3f}]")
    print(f"   Right foot (support): [{T_support_w[0,3]:.3f}, {T_support_w[1,3]:.3f}, {T_support_w[2,3]:.3f}]")
    
    # setup the plan with 20 steps
    no_steps = 20
    print(f"Creating footstep plan with {no_steps} steps...")
    planner = FootStepPlanner(conf)  # Create the planner
    T_0_w = pin.SE3(np.eye(3), np.array([0, 0, 0]))  # Starting position
    plan = planner.planLine(T_0_w, Side.LEFT, no_steps)  # Create the plan
    
    # Append the two last steps once more to the plan so our mpc horizon will never run out
    if len(plan) >= 2:
        plan.append(plan[-2])  # Second to last step
        plan.append(plan[-1])  # Last step
    
    print(f"   Created plan with {len(plan)} total steps")
    
    # generate reference
    print("Generating ZMP reference...")
    ZMP_ref = []  # Generate the mpc reference
    
    for i, step in enumerate(plan):
        # Simple ZMP reference: alternate between feet positions
        if i < 2:  # Initial double support
            zmp_pos = np.array([0.0, 0.0, 0.0])
        elif i >= len(plan) - 2:  # Final double support
            final_pos = plan[-1].position()
            zmp_pos = np.array([final_pos[0], 0.0, 0.0])
        else:
            # During single support, ZMP toward support foot
            step_pos = step.position()
            if step.side == Side.LEFT:
                # Right foot is support
                zmp_pos = np.array([step_pos[0], conf.step_size_y, 0.0])
            else:
                # Left foot is support  
                zmp_pos = np.array([step_pos[0], -conf.step_size_y, 0.0])
        
        ZMP_ref.append(zmp_pos)
    
    print(f"   Generated {len(ZMP_ref)} ZMP reference points")
    
    # plot the plan (make sure this works first)
    print("Plotting footstep plan in PyBullet...")
    planner.plot(simulator)
    planner.print_plan()
    
    # setup the lip models
    print("Setting up LIP MPC and interpolator...")
    mpc = LIPMPC(conf)  # setup mpc
    
    # Assume the com is over the first support foot
    com_pos = robot.stack.getCenterOfMass()
    com_vel = robot.stack.getCenterOfMassVelocity()
    x0 = np.array([com_pos[0], com_pos[1], com_vel[0], com_vel[1], com_pos[0], com_pos[1]])
    interpolator = LIPInterpolator(x0, conf)  # Create the interpolator and set the initial state
    
    print(f"   Initial state: x={x0[0]:.3f}, y={x0[1]:.3f}, vx={x0[2]:.3f}, vy={x0[3]:.3f}")
    
    # set the com task reference to the initial support foot
    c, c_dot, c_ddot = interpolator.comState()
    robot.stack.setComReference(c, c_dot, c_ddot)  # Set the COM reference to be over supporting foot
    
    # Setup foot trajectory planner
    T0 = pin.SE3(np.eye(3), np.array([0, 0, 0]))
    T1 = pin.SE3(np.eye(3), np.array([0.2, 0, 0]))
    duration = 0.8
    height = 0.08
    foot_trajectory = SwingFootTrajectory(T0, T1, duration, height)
    
    
    # Set initial support and swing feet
    robot.setSupportFoot(Side.RIGHT)
    robot.setSwingFoot(Side.LEFT)
    
    print("   MPC and interpolator configured")
    
    ############################################################################
    # logging setup
    ############################################################################

    pre_dur = 3.0   # Time to wait before walking should start
    
    # Compute number of iterations:
    N_pre = int(pre_dur / conf.dt_sim)  # number of sim steps before walking starts 
    N_sim = len(plan) * conf.no_sim_per_step  # total number of sim steps during walking
    N_mpc = N_sim // conf.no_sim_per_mpc  # total number of mpc steps during walking
    
    print(f"Simulation parameters:")
    print(f"   Pre-walking: {N_pre} steps ({pre_dur}s)")
    print(f"   Walking: {N_sim} steps ({N_sim * conf.dt_sim:.1f}s)")
    print(f"   MPC updates: {N_mpc}")
    
    # Create vectors to log all the data of the simulation
    TIME = np.nan * np.empty(N_sim)
    
    # COM data (planned reference, pinocchio and pybullet)
    COM_POS_ref = np.nan * np.empty((N_sim, 3))
    COM_VEL_ref = np.nan * np.empty((N_sim, 3))
    COM_ACC_ref = np.nan * np.empty((N_sim, 3))
    COM_POS_pin = np.nan * np.empty((N_sim, 3))
    COM_VEL_pin = np.nan * np.empty((N_sim, 3))
    
    # Angular momentum from pinocchio
    ANGULAR_MOMENTUM = np.nan * np.empty((N_sim, 3))
    
    # Foot data (planned reference, pinocchio)
    LFOOT_POS_ref = np.nan * np.empty((N_sim, 3))
    LFOOT_VEL_ref = np.nan * np.empty((N_sim, 3)) 
    LFOOT_ACC_ref = np.nan * np.empty((N_sim, 3))
    RFOOT_POS_ref = np.nan * np.empty((N_sim, 3))
    RFOOT_VEL_ref = np.nan * np.empty((N_sim, 3))
    RFOOT_ACC_ref = np.nan * np.empty((N_sim, 3))
    
    LFOOT_POS_pin = np.nan * np.empty((N_sim, 3))
    RFOOT_POS_pin = np.nan * np.empty((N_sim, 3))
    
    # ZMP data (planned reference, from estimator)
    ZMP_ref_log = np.nan * np.empty((N_sim, 3))
    ZMP_est = np.nan * np.empty((N_sim, 3))
    
    # DCM from estimator
    DCM_est = np.nan * np.empty((N_sim, 3))
    
    # Forces (from sensors and pinocchio)
    LFOOT_FORCE = np.nan * np.empty((N_sim, 6))
    RFOOT_FORCE = np.nan * np.empty((N_sim, 6))
    
    print("   Data logging arrays initialized")
    
    ############################################################################
    # main control loop
    ############################################################################
    
    k = 0                           # current MPC index                          
    plan_idx = 1                    # current index of the step within foot step plan
    t_step_elapsed = 0.0            # elapsed time within current step (use to evaluate spline)
    t_publish = 0.0                 # last publish time (last time we published something)
    
    print("\nStarting main control loop...")
    print("   Pre-walking phase: robot will stabilize")
    print("   Walking phase: following footstep plan")
    print("   Press Ctrl+C to stop simulation")
    
    try:
        for i in range(-N_pre, N_sim):
            t = simulator.simTime()  # simulator time
            dt = conf.dt_sim        # simulator dt
            
            ########################################################################
            # update the mpc every no_sim_per_mpc steps
            ########################################################################
            
            if i >= 0 and i % conf.no_sim_per_mpc == 0:  # when to update mpc
                # Get current LIP state
                xk = interpolator.x
                
                # Extract ZMP reference over the current horizon
                horizon_start = k
                horizon_end = min(horizon_start + conf.mpc_horizon, len(ZMP_ref))
                ZMP_ref_k = ZMP_ref[horizon_start:horizon_end]
                
                # Pad with last value if necessary
                while len(ZMP_ref_k) < conf.mpc_horizon:
                    ZMP_ref_k.append(ZMP_ref_k[-1] if ZMP_ref_k else np.zeros(3))
                
                # Solve MPC
                uk = mpc.solve(xk, ZMP_ref_k)
                
                # Update interpolator
                interpolator.update(uk)
                
                k += 1        

            ########################################################################
            # update the foot spline every no_sim_per_step steps
            ########################################################################

            if i >= 0 and i % conf.no_sim_per_step == 0 and plan_idx < len(plan):  # when to update spline
                # Get next step from plan
                next_step = plan[plan_idx]
                
                print(f"Step {plan_idx}: {next_step.side.name} foot to [{next_step.position()[0]:.3f}, {next_step.position()[1]:.3f}]")
                
                # Update support/swing feet
                robot.setSwingFoot(next_step.side)
                robot.setSupportFoot(Side.LEFT if next_step.side == Side.RIGHT else Side.RIGHT)
                
                # Get current swing foot pose
                current_swing_pose = robot.swingFootPose()
                current_pos = current_swing_pose[:3, 3]
                
                # Plan foot trajectory
                target_pos = next_step.position()
                foot_trajectory.planTrajectory(
                    start_pos=current_pos,
                    end_pos=target_pos,
                    duration=conf.step_duration,
                    step_height=conf.step_height
                )
                
                t_step_elapsed = 0.0
                plan_idx += 1
                
            ########################################################################
            # in every iteration when walking
            ########################################################################
            
            if i >= 0:
                # Update foot trajectory
                if t_step_elapsed < conf.step_duration and plan_idx > 1:
                    foot_pos_ref, foot_vel_ref, foot_acc_ref = foot_trajectory.getReference(t_step_elapsed)
                    
                    # Create homogeneous transformation matrix
                    T_swing_ref = np.eye(4)
                    T_swing_ref[:3, 3] = foot_pos_ref
                    
                    # Update swing foot reference
                    robot.updateSwingFootRef(T_swing_ref, foot_vel_ref, foot_acc_ref)
                
                # Update COM reference
                c, c_dot, c_ddot = interpolator.comState()
                robot.stack.setComReference(c, c_dot, c_ddot)
                
                t_step_elapsed += dt

            ########################################################################
            # update the simulation
            ########################################################################

            # update the simulator and the robot
            simulator.step()
            robot.update()

            # publish to ros
            if t - t_publish > 1./30.:
                t_publish = t
                try:
                    robot.publish()
                    rclpy.spin_once(node, timeout_sec=0.001)
                except:
                    pass  # Skip if ROS not available
                
            # store for visualizations
            if i >= 0:
                TIME[i] = t
                
                # Log COM data
                com_ref_pos, com_ref_vel, com_ref_acc = interpolator.comState()
                COM_POS_ref[i] = com_ref_pos
                COM_VEL_ref[i] = com_ref_vel
                COM_ACC_ref[i] = com_ref_acc
                
                COM_POS_pin[i] = robot.stack.getCenterOfMass()
                COM_VEL_pin[i] = robot.stack.getCenterOfMassVelocity()
                
                # Angular momentum
                ANGULAR_MOMENTUM[i] = robot.stack.getAngularMomentum()
                
                # Foot positions
                lfoot_pose = robot.stack.getFramePose(conf.lf_frame_name)
                rfoot_pose = robot.stack.getFramePose(conf.rf_frame_name)
                LFOOT_POS_pin[i] = lfoot_pose[:3, 3]
                RFOOT_POS_pin[i] = rfoot_pose[:3, 3]
                
                # Foot references
                if t_step_elapsed < conf.step_duration and plan_idx > 1:
                    foot_pos_ref, foot_vel_ref, foot_acc_ref = foot_trajectory.getReference(t_step_elapsed)
                    if robot.swing_foot == Side.LEFT:
                        LFOOT_POS_ref[i] = foot_pos_ref
                        LFOOT_VEL_ref[i] = foot_vel_ref
                        LFOOT_ACC_ref[i] = foot_acc_ref
                        RFOOT_POS_ref[i] = RFOOT_POS_pin[i]  # Support foot doesn't move
                        RFOOT_VEL_ref[i] = np.zeros(3)
                        RFOOT_ACC_ref[i] = np.zeros(3)
                    else:
                        RFOOT_POS_ref[i] = foot_pos_ref
                        RFOOT_VEL_ref[i] = foot_vel_ref
                        RFOOT_ACC_ref[i] = foot_acc_ref
                        LFOOT_POS_ref[i] = LFOOT_POS_pin[i]  # Support foot doesn't move
                        LFOOT_VEL_ref[i] = np.zeros(3)
                        LFOOT_ACC_ref[i] = np.zeros(3)
                else:
                    # Both feet stationary
                    LFOOT_POS_ref[i] = LFOOT_POS_pin[i]
                    RFOOT_POS_ref[i] = RFOOT_POS_pin[i]
                    LFOOT_VEL_ref[i] = np.zeros(3)
                    RFOOT_VEL_ref[i] = np.zeros(3)
                    LFOOT_ACC_ref[i] = np.zeros(3)
                    RFOOT_ACC_ref[i] = np.zeros(3)
                
                # ZMP and DCM
                if k < len(ZMP_ref):
                    ZMP_ref_log[i] = ZMP_ref[k]
                else:
                    ZMP_ref_log[i] = ZMP_ref[-1]
                
                ZMP_est[i] = robot.zmp
                DCM_est[i] = robot.dcm
                
                # Contact forces
                contact_forces = robot.stack.getContactForces()
                LFOOT_FORCE[i] = contact_forces.get("left_foot", np.zeros(6))
                RFOOT_FORCE[i] = contact_forces.get("right_foot", np.zeros(6))
            
            # Progress indicator
            if i >= 0 and i % 1000 == 0:
                progress = (i / N_sim) * 100
                print(f"Progress: {progress:.1f}% - Time: {t:.2f}s - Step: {plan_idx-1}/{len(plan)-2}")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    print("Walking simulation completed!")

    ########################################################################
    # enough with the simulation, lets plot
    ########################################################################
    
    print("Generating analysis plots...")
    
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8')
    
    # Remove NaN values for plotting
    valid_indices = ~np.isnan(TIME)
    TIME_clean = TIME[valid_indices]
    
    if len(TIME_clean) > 0:
        # Create comprehensive plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # COM Position
        axes[0,0].plot(TIME_clean, COM_POS_ref[valid_indices, 0], 'b-', label='COM Ref X', linewidth=2)
        axes[0,0].plot(TIME_clean, COM_POS_pin[valid_indices, 0], 'b--', label='COM Actual X', linewidth=1)
        axes[0,0].plot(TIME_clean, COM_POS_ref[valid_indices, 1], 'r-', label='COM Ref Y', linewidth=2)
        axes[0,0].plot(TIME_clean, COM_POS_pin[valid_indices, 1], 'r--', label='COM Actual Y', linewidth=1)
        axes[0,0].set_ylabel('COM Position [m]')
        axes[0,0].legend()
        axes[0,0].grid(True)
        axes[0,0].set_title('Center of Mass Position')
        
        # COM Velocity
        axes[0,1].plot(TIME_clean, COM_VEL_ref[valid_indices, 0], 'b-', label='COM Vel X Ref', linewidth=2)
        axes[0,1].plot(TIME_clean, COM_VEL_pin[valid_indices, 0], 'b--', label='COM Vel X Actual', linewidth=1)
        axes[0,1].plot(TIME_clean, COM_VEL_ref[valid_indices, 1], 'r-', label='COM Vel Y Ref', linewidth=2)
        axes[0,1].plot(TIME_clean, COM_VEL_pin[valid_indices, 1], 'r--', label='COM Vel Y Actual', linewidth=1)
        axes[0,1].set_ylabel('COM Velocity [m/s]')
        axes[0,1].legend()
        axes[0,1].grid(True)
        axes[0,1].set_title('Center of Mass Velocity')
        
        # Foot Positions
        axes[1,0].plot(TIME_clean, LFOOT_POS_ref[valid_indices, 0], 'b-', label='Left Foot X Ref', linewidth=2)
        axes[1,0].plot(TIME_clean, LFOOT_POS_pin[valid_indices, 0], 'b--', label='Left Foot X Actual', linewidth=1)
        axes[1,0].plot(TIME_clean, RFOOT_POS_ref[valid_indices, 0], 'r-', label='Right Foot X Ref', linewidth=2)
        axes[1,0].plot(TIME_clean, RFOOT_POS_pin[valid_indices, 0], 'r--', label='Right Foot X Actual', linewidth=1)
        axes[1,0].set_ylabel('Foot Position X [m]')
        axes[1,0].legend()
        axes[1,0].grid(True)
        axes[1,0].set_title('Foot Positions X')
        
        axes[1,1].plot(TIME_clean, LFOOT_POS_ref[valid_indices, 1], 'b-', label='Left Foot Y Ref', linewidth=2)
        axes[1,1].plot(TIME_clean, LFOOT_POS_pin[valid_indices, 1], 'b--', label='Left Foot Y Actual', linewidth=1)
        axes[1,1].plot(TIME_clean, RFOOT_POS_ref[valid_indices, 1], 'r-', label='Right Foot Y Ref', linewidth=2)
        axes[1,1].plot(TIME_clean, RFOOT_POS_pin[valid_indices, 1], 'r--', label='Right Foot Y Actual', linewidth=1)
        axes[1,1].set_ylabel('Foot Position Y [m]')
        axes[1,1].legend()
        axes[1,1].grid(True)
        axes[1,1].set_title('Foot Positions Y')
        
        # ZMP and DCM
        axes[2,0].plot(TIME_clean, ZMP_ref_log[valid_indices, 0], 'b-', label='ZMP Ref X', linewidth=2)
        axes[2,0].plot(TIME_clean, ZMP_est[valid_indices, 0], 'b--', label='ZMP Est X', linewidth=1)
        axes[2,0].plot(TIME_clean, ZMP_ref_log[valid_indices, 1], 'r-', label='ZMP Ref Y', linewidth=2)
        axes[2,0].plot(TIME_clean, ZMP_est[valid_indices, 1], 'r--', label='ZMP Est Y', linewidth=1)
        axes[2,0].plot(TIME_clean, DCM_est[valid_indices, 0], 'g:', label='DCM X', linewidth=2)
        axes[2,0].plot(TIME_clean, DCM_est[valid_indices, 1], 'm:', label='DCM Y', linewidth=2)
        axes[2,0].set_ylabel('Position [m]')
        axes[2,0].set_xlabel('Time [s]')
        axes[2,0].legend()
        axes[2,0].grid(True)
        axes[2,0].set_title('ZMP and DCM')
        
        # Contact Forces
        axes[2,1].plot(TIME_clean, LFOOT_FORCE[valid_indices, 2], 'b-', label='Left Foot Fz', linewidth=2)
        axes[2,1].plot(TIME_clean, RFOOT_FORCE[valid_indices, 2], 'r-', label='Right Foot Fz', linewidth=2)
        total_force = LFOOT_FORCE[valid_indices, 2] + RFOOT_FORCE[valid_indices, 2]
        axes[2,1].plot(TIME_clean, total_force, 'g-', label='Total Force', linewidth=1)
        axes[2,1].axhline(y=robot.robot.mass() * 9.81, color='k', linestyle='--', label='Robot Weight')
        axes[2,1].set_ylabel('Normal Force [N]')
        axes[2,1].set_xlabel('Time [s]')
        axes[2,1].legend()
        axes[2,1].grid(True)
        axes[2,1].set_title('Contact Forces')
        
        plt.tight_layout()
        plt.savefig('talos_walking_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Performance analysis
        print(f"\nPerformance Summary:")
        com_error = np.linalg.norm(COM_POS_pin[valid_indices] - COM_POS_ref[valid_indices], axis=1)
        max_com_error = np.max(com_error)
        avg_com_error = np.mean(com_error)
        
        print(f"   Simulation time: {TIME_clean[-1]:.1f}s")
        print(f"   Steps completed: {plan_idx-1}/{len(plan)-2}")
        print(f"   Max COM error: {max_com_error:.4f}m")
        print(f"   Avg COM error: {avg_com_error:.4f}m")
        
        if max_com_error < 0.05:
            print("   Excellent tracking performance!")
        elif max_com_error < 0.1:
            print("   Good tracking performance")
        else:
            print("   Tracking performance needs improvement")
        
        print("Analysis saved as 'talos_walking_analysis.png'")
    else:
        print("No valid data to plot")

if __name__ == '__main__': 
    try:
        main()
    finally:
        rclpy.shutdown()