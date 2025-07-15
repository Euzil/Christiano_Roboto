"""This file contains the configuration of TSID for the Ainex robot
"""

import os
import pinocchio as pin
import numpy as np
from ament_index_python.packages import get_package_share_directory
import os.path

################################################################################
# robot configuration for Ainex
################################################################################

# robot name
robot_name = "ainex"

# robot urdf
ainex_description = get_package_share_directory('ainex_description')
urdf = os.path.join(ainex_description, 'robots', 'ainex.urdf')

# MISSING: Add path for meshes (required by TSIDWrapper)
path = os.path.join(ainex_description, "meshes/../..")

# Number of actuated joints for Ainex
na = 24

# home position for actuated joints (24 joints for Ainex)
# Standing pose für Ainex (leicht gebeugte Knie für Stabilität)
q_actuated_home = np.array([
    # Right leg (6 DOF): hip_yaw, hip_roll, hip_pitch, knee, ank_pitch, ank_roll
    0.0, 0.0, -0.2, 0.4, -0.2, 0.0,
    # Left leg (6 DOF)
    0.0, 0.0, -0.2, 0.4, -0.2, 0.0,
    # Head (2 DOF): pan, tilt
    0.0, 0.0,
    # Right arm (5 DOF): sho_pitch, sho_roll, el_pitch, el_yaw, gripper
    0.0, 0.0, -0.5, 0.0, 0.0,
    # Left arm (5 DOF)
    0.0, 0.0, -0.5, 0.0, 0.0
])

# MISSING: Add q_home (required by TSIDWrapper)
q_home = np.hstack([np.array([0, 0, 0.225, 0, 0, 0, 1]), q_actuated_home])  # 7 DOF base + 24 DOF joints

################################################################################
# simulation
################################################################################
dt = 0.001
T_sim = 20.0

################################################################################
# COM (Center of Mass) reference
################################################################################
com_ref = np.array([0.0, 0.0, 0.15])  # Adjusted for Ainex height (lower than 0.25)
com_offset = np.array([0.0, 0.0, 0.0])

################################################################################
# feet reference
################################################################################
# Right foot position (relative to base)
rf_ref = np.array([0.0, -0.05, 0.0])  # Adjusted for Ainex dimensions
# Left foot position (relative to base)  
lf_ref = np.array([0.0, 0.05, 0.0])   # Adjusted for Ainex dimensions

################################################################################
# foot print
################################################################################

foot_scaling = 0.5  # Reduced for smaller Ainex robot
lfxp = foot_scaling*0.08                # foot length in positive x direction (reduced)
lfxn = foot_scaling*0.06                # foot length in negative x direction (reduced)
lfyp = foot_scaling*0.04                # foot length in positive y direction (reduced)
lfyn = foot_scaling*0.04                # foot length in negative y direction (reduced)

lz = 0.                                 # foot sole height with respect to ankle joint
f_mu = 0.3                              # friction coefficient
f_fMin = 5.0                            # minimum normal force
f_fMax = 1e6                            # maximum normal force
contactNormal = np.array([0., 0., 1.])  # direction of the normal to the contact surface

################################################################################
# frame names
################################################################################

# Ainex robot frame names
rf_frame_name = "r_ank_roll_link"       # right foot frame name
lf_frame_name = "l_ank_roll_link"       # left foot frame name
rh_frame_name = "r_gripper_link"        # right hand frame name
lh_frame_name = "l_gripper_link"        # left hand frame name
torso_frame_name = "body_link"          # torso frame name
base_frame_name = "base_link"           # base link

################################################################################
# TSID Task weights
################################################################################
w_com = 1e3             # Increased from 1e1
w_am = 1e-4             # weight of angular momentum task
w_foot = 1e-1           # weight of the foot motion task: here no motion
w_hand = 1e-1           # weight of the hand motion task
w_torso = 1             # weight torso orientation motion task
w_feet_contact = 1e6    # Increased from 1e5
w_hand_contact = 1e5    # weight for hand in contact
w_posture = 1e1         # Increased from 1e-3
w_force_reg = 1e-5      # weight of force regularization task (note this is really important!)
w_torque_bounds = 1.0   # weight of the torque bounds: here no bounds
w_joint_bounds = 0.0    # weight of the velocity bounds: here no bounds

################################################################################
# TSID Task gains
################################################################################
kp_contact = 100.0      # Increased from 10.0
kp_foot = 10.0          # proportional gain of contact constraint
kp_hand = 10.0          # proportional gain of hand constraint
kp_torso = 10.0         # proportional gain of torso constraint
kp_com = 100.0          # Increased from 10.0
kp_am = 10.0            # proportional gain of angular momentum task

# Posture gains - FIXED: Must be exactly 24 values for Ainex
kp_posture = np.array([
    # Right leg (6 joints): r_hip_yaw, r_hip_roll, r_hip_pitch, r_knee, r_ank_pitch, r_ank_roll
    100., 100., 100., 100., 100., 100.,
    # Left leg (6 joints): l_hip_yaw, l_hip_roll, l_hip_pitch, l_knee, l_ank_pitch, l_ank_roll
    100., 100., 100., 100., 100., 100.,
    # Head (2 joints): head_pan, head_tilt
    1000., 1000.,
    # Right arm (5 joints): r_sho_pitch, r_sho_roll, r_el_pitch, r_el_yaw, r_gripper
    100., 100., 100., 100., 100.,
    # Left arm (5 joints): l_sho_pitch, l_sho_roll, l_el_pitch, l_el_yaw, l_gripper
    100., 100., 100., 100., 100.
])

# Verify the size is correct
assert len(kp_posture) == na, f"kp_posture must have {na} elements, got {len(kp_posture)}"
assert len(q_actuated_home) == na, f"q_actuated_home must have {na} elements, got {len(q_actuated_home)}"

################################################################################
# additional TSID parameters
################################################################################
kd_contact = 2.0 * np.sqrt(kp_contact)
kd_com = 2.0 * np.sqrt(kp_com)
kd_posture = 2.0 * np.sqrt(kp_posture)

# masks for joints (all enabled for Ainex)
masks_posture = np.ones(na)                     # mask out joint (here none)

tau_max_scaling = 1.45          # scaling factor of torque bounds
v_max_scaling = 0.8             # scaling velocity bounds
