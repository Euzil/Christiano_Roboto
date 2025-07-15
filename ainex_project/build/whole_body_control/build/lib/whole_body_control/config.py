"""This file contains the configuration of TSID for the Ainex robot
"""

import os
import pinocchio as pin
import numpy as np
from ament_index_python.packages import get_package_share_directory

################################################################################
# robot
################################################################################

# Talos robot configuration (commented out)
#talos_description = get_package_share_directory('talos_description')
#urdf = os.path.join(talos_description, "robots/talos_reduced_no_hands.urdf")
#path = os.path.join(talos_description, "meshes/../..")

# Ainex robot configuration
ainex_description = get_package_share_directory('ainex_description')
urdf = os.path.join(ainex_description, "robots/ainex.urdf")
path = os.path.join(ainex_description, "meshes/../..")    

dt = 0.001                                      # controller time step
f_cntr = 1.0/dt                                 # controller freq
na = 24                                         # number of actuated joints for Ainex (was 30 for Talos)

# homing pose for Ainex robot
q_actuated_home = np.zeros(na)
# Ainex joint configuration: left leg (6) + right leg (6) + left arm (6) + right arm (6) = 24 joints
q_actuated_home[:6] = np.array([0.0, 0.0, -0.2, 0.4, -0.2, 0.0])    # left leg (hip_yaw, hip_roll, hip_pitch, knee, ank_pitch, ank_roll)
q_actuated_home[6:12] = np.array([0.0, 0.0, -0.2, 0.4, -0.2, 0.0])  # right leg (hip_yaw, hip_roll, hip_pitch, knee, ank_pitch, ank_roll)
q_actuated_home[12:18] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # left arm (sho_pitch, sho_roll, el_pitch, el_yaw, gripper)
q_actuated_home[18:24] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # right arm (sho_pitch, sho_roll, el_pitch, el_yaw, gripper)
q_home = np.hstack([np.array([0, 0, 0.3, 0, 0, 0, 1]), q_actuated_home])  # Lower initial height for Ainex

# Talos homing pose (commented out)
#q_actuated_home = np.zeros(30)  # Talos had 30 actuated joints
#q_actuated_home[:6] = np.array([0.0004217227847487237, -0.00457389353360238, -0.44288825380502317, 0.9014217614029372, -0.4586176441428318, 0.00413219379047014])
#q_actuated_home[6:12] = np.array([-0.0004612402198835852, -0.0031162522884748967, -0.4426315354712109, 0.9014369887125069, -0.4588832011407824, 0.003546732694320376])
#q_home = np.hstack([np.array([0, 0, 0.9, 0, 0, 0, 1]), q_actuated_home])  # Talos was taller (0.9m)

'''
Ainex joint mapping:
0, 1, 2, 3, 4, 5,       # left leg (l_hip_yaw, l_hip_roll, l_hip_pitch, l_knee, l_ank_pitch, l_ank_roll)
6, 7, 8, 9, 10, 11,     # right leg (r_hip_yaw, r_hip_roll, r_hip_pitch, r_knee, r_ank_pitch, r_ank_roll)
12, 13, 14, 15, 16, 17, # left arm (l_sho_pitch, l_sho_roll, l_el_pitch, l_el_yaw, l_gripper)
18, 19, 20, 21, 22, 23, # right arm (r_sho_pitch, r_sho_roll, r_el_pitch, r_el_yaw, r_gripper)
'''

'''
Talos joint mapping (commented out):
0, 1, 2, 3, 4, 5,       # left leg
6, 7, 8, 9, 10, 11,     # right leg
12, 13,                 # torso
14, 15, 16, 17, 18, 19, 20      # left arm
21, 22, 23, 24, 25, 26, 27      # right arm
28, 29                  # head
'''

################################################################################
# foot print
################################################################################

foot_scaling = 1.
lfxp = foot_scaling*0.12                # foot length in positive x direction
lfxn = foot_scaling*0.08                # foot length in negative x direction
lfyp = foot_scaling*0.065               # foot length in positive y direction
lfyn = foot_scaling*0.065               # foot length in negative y direction

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

# Talos robot frame names (commented out)
#rf_frame_name = "leg_right_sole_fix_joint"  # right foot frame name
#lf_frame_name = "leg_left_sole_fix_joint"   # left foot frame name
#rh_frame_name = "contact_right_link"        # right arm frame name
#lh_frame_name = "contact_left_link"         # left arm frame name
#torso_frame_name = "torso_2_link"           # keep the imu horizontal
#base_frame_name = "base_link"               # base link

################################################################################
# TSID
################################################################################

# Task weights
w_com = 1e1             # weight of center of mass task
w_am = 1e-4             # weight of angular momentum task
w_foot = 1e-1           # weight of the foot motion task: here no motion
w_hand = 1e-1           # weight of the hand motion task
w_torso = 1             # weight torso orientation motion task
w_feet_contact = 1e5    # weight of foot in contact (negative means infinite weight)
w_hand_contact = 1e5    # weight for hand in contact
w_posture = 1e-3        # weight of joint posture task
w_force_reg = 1e-5      # weight of force regularization task (note this is really important!)
w_torque_bounds = 1.0   # weight of the torque bounds: here no bounds
w_joint_bounds = 0.0    # weight of the velocity bounds: here no bounds

# weights
kp_contact = 10.0       # proportional gain of contact constraint
kp_foot = 10.0          # proportional gain of contact constraint
kp_hand = 10.0          # proportional gain of hand constraint
kp_torso = 10.0         # proportional gain of torso constraint
kp_com = 10.0           # proportional gain of com task
kp_am = 10.0            # proportional gain of angular momentum task

# proportional gain of joint posture task for Ainex (24 joints)
kp_posture = np.array([
        10., 10., 10., 10., 10., 10.,           # left leg
        10., 10., 10., 10., 10., 10.,           # right leg
        10., 10., 10., 10., 10., 10.,           # left arm
        10., 10., 10., 10., 10., 10.,           # right arm
])

# Talos proportional gain of joint posture task (commented out)
#kp_posture = np.array([
#        10., 10., 10., 10., 10., 10.,           # left leg  #low gain on axis along y and knee
#        10., 10., 10., 10., 10., 10.,           # right leg #low gain on axis along y and knee
#        5000., 5000.,                           # torso really high to make them stiff
#        10., 10., 10., 10., 10., 10., 10.,      # right arm make the x direction soft
#        10., 10., 10., 10., 10., 10., 10.,      # left arm make the x direction soft
#        1000., 1000.                            # head
#])

masks_posture = np.ones(na)                     # mask out joint (here none)

tau_max_scaling = 1.45          # scaling factor of torque bounds
v_max_scaling = 0.8             # scaling velocity bounds
