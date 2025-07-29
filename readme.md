# AiNex Humanoid Robot Walking Controller

**Team Members:** Theresa Gräbner, Youran Wang, Djamal Halim

## Contribution

Every team member contributed equally to the project.

## Overview

This project implements a comprehensive walking controller for the AiNex humanoid robot that features a complete motion sequence from standing to ball kicking with celebration. The system uses Task Space Inverse Dynamics (TSID) for whole-body control and integrates with ROS2 for real-time robot communication.

## Features

### 🚶 Multi-Phase Walking Sequence
- **Phase 1: HOME** - Initialize robot to home position
- **Phase 2: STANDING** - Establish stable base and balance
- **Phase 3: PLANNING** - Generate footstep trajectory and path visualization
- **Phase 4: WALKING** - Execute coordinated walking motion
- **Phase 5: END POSITION** - Transition to stable end stance
- **Phase 6: KICKING** - Perform ball kicking motion with recovery
- **Phase 7: CELEBRATION** - Victory sequence with posture changes

### 🎯 Technical Capabilities
- TSID-based trajectory optimization and control
- Real-time footstep planning and execution
- Coordinated center-of-mass (COM) management
- Hardware safety mechanisms and error handling
- Real-time visualization in RViz
- Comprehensive logging and feedback

## Architecture

### Dependencies
- **ROS2** (Robot Operating System) - Communication framework
- **Pinocchio** - Rigid body dynamics library
- **TSID** (Task Space Inverse Dynamics) - Control framework
- **RViz** - 3D visualization and monitoring
- **AiNex Motion Controller** - Hardware interface

### Control Framework
```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   walking.py    │───▶│ TSID Wrapper │───▶│ Hardware Robot  │
│ (State Machine) │    │ (Control)    │    │ (AiNex)         │
└─────────────────┘    └──────────────┘    └─────────────────┘

```

## Installation & Setup

### 1. Prerequisites
```bash
# Ensure you're in the dev container with Ubuntu 24.04.2 LTS
# Install ROS2 dependencies (if not already installed)
sudo apt update
sudo apt install ros-rolling-desktop python3-colcon-common-extensions
```

### 2. Build the Package
```bash
cd /Christiano_Roboto/ainex_project
colcon build --symlink-install
source install/setup.bash
```

### 3. Verify Installation
```bash
# Check if the package is properly installed
ros2 pkg list | grep whole_body_control
ros2 run whole_body_control --help
```

## Usage

### 🔬 Simulation Mode with RViz
Run the walking controller in full simulation with RViz visualization:

```bash
# Terminal 1: Start simulation with RViz
cd /Christiano_Roboto/ainex_project
source install/setup.bash
ros2 launch whole_body_control launch_simulation.py

# This automatically launches:
# - AiNex robot model
# - RViz with pre-configured visualization
# - Joint state publishers
# - Transform publishers
```

**What happens in simulation:**
- RViz opens with robot visualization and trajectory display
- Robot starts in home position
- Establishes stable standing posture
- Plans footstep trajectory (20 steps forward) 
- Executes walking motion with real-time joint state feedback

**RViz Features:**
- Real-time robot model visualization
- Joint state monitoring
- Center of mass trajectory

### 🤖 Hardware Control Mode
Run the walking controller directly without simulation:

```bash
cd /Christiano_Roboto/ainex_project
source install/setup.bash
ros2 launch whole_body_control launch_hardware.py
```

**What happens in hardware mode:**
- Robot starts in home position
- Establishes stable standing posture
- Plans footstep trajectory (20 steps forward)
- Executes walking motion with TSID control
- Performs ball kicking sequence
- Concludes with celebration routine

### 🔧 Configuration Parameters

Key parameters in `walking.py` that can be adjusted:

```python
# Phase durations (seconds)
home_duration = 2.0
standing_duration = 1.5
planning_duration = 3.0
end_duration = 4.0

# Walking parameters
first_step = 0.065      # First/last step length (m)
other_step = 0.13       # Regular step length (m)
num_steps = 20          # Total number of steps
height = 0.03           # Step height (m)
phase_duration = 1.5    # Duration per step phase (s)

# Center of Mass adjustments
shift_com_x = 0.03      # Forward COM shift (m)
shift_com_y = 0.005     # Lateral COM shift (m)
shift_com_z = -0.02     # Vertical COM adjustment (m)
```

## What Happens in walking.py

### State Machine Overview
The walking controller implements a finite state machine with 7 distinct phases:

#### Phase 1: HOME
- Initializes robot configuration
- Sets up TSID wrapper and control framework
- Establishes initial joint positions

#### Phase 2: STANDING 
- Activates hardware control interface
- Establishes stable bipedal stance
- Centers COM between feet
- Prepares for dynamic motion

#### Phase 3: PLANNING 
- Generates footstep trajectory
- Creates visual markers in RViz
- Calculates optimal step sequence
- Validates path feasibility

#### Phase 4: WALKING 
Each step consists of 4 sub-phases:
1. **COM Shift** - Transfer weight to support foot
2. **Lift & Move** - Lift swing foot and move forward
3. **Place** - Lower foot to target position
4. **Shift Back** - Transfer weight to new support foot

#### Phase 5: END POSITION 
- Stabilizes robot after walking
- Centers COM for balanced stance
- Prepares for kicking motion

#### Phase 6: KICKING
1. **Preparation** - Shift COM to support leg
2. **Backswing** - Retract kicking leg
3. **Strike** - Forward kick motion
4. **Recovery** - Return to stable stance

#### Phase 7: CELEBRATION 
- Victory pose sequence
- Posture changes (crouch → stand → arms down)
- System shutdown and cleanup

### Hardware Integration

**Hardware Control Flow:**
1. `hardware_controller.setPosture()` - High-level posture commands
2. `hardware_controller.setJointPositions()` - Direct joint control
3. TSID generates optimal joint trajectories
4. Commands sent to robot at 30Hz frequency
5. Safety monitoring prevents dangerous motions

**Safety Features:**
- Joint limit checking
- COM stability monitoring
- Emergency stop capability
- Gradual motion transitions
- Error recovery mechanisms


## Troubleshooting

### Common Issues

#### 1. Build Errors
```bash
# Clean and rebuild
rm -rf build/ install/ log/
colcon build --packages-select whole_body_control
```

#### 2. Hardware Connection Check
Before running any hardware commands, verify the robot connection:

```bash
# Step 1: Check network connectivity to robot
ping 192.168.50.203

# Step 2: Verify ROS2 nodes are running
ros2 node list

# Look for these essential nodes:
# - /Joint_Control
# - /camera_publisher
```

**If nodes are missing:**
1. **Power cycle the robot** - Turn robot off and on again (usually 1-3 times)
2. **Check battery level** - Low battery can cause connection issues, plug in charger if needed
3. **Wait 30-60 seconds** after power-on before checking nodes again
4. **Repeat ping and node list** commands until both nodes appear

**Typical startup sequence:**
```bash
# After robot power-on, wait and check:
ping 192.168.50.203
ros2 node list | grep -E "(Joint_Control|camera_publisher)"

# If missing, power cycle again and repeat
```

#### 3. ROS Domain ID Configuration
Check and configure the correct ROS Domain ID in your Docker container:

```bash
# Check current ROS Domain ID
echo $ROS_DOMAIN_ID

# If empty or incorrect, set to 43 (our project setting)
export ROS_DOMAIN_ID=43

# Verify the setting
echo $ROS_DOMAIN_ID

# Make it permanent for current session
echo "export ROS_DOMAIN_ID=43" >> ~/.bashrc
source ~/.bashrc
```

**Note:** Our project uses ROS_DOMAIN_ID=43. Ensure this matches between your Docker container and the robot system.

