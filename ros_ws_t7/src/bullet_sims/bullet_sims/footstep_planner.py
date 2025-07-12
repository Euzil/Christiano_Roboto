"""
Enhanced footstep_planner.py
Adds more functionality and better integration support
"""

import numpy as np
import pinocchio as pin
import pybullet as p
import pybullet_data
import time
from enum import Enum

class Side(Enum):
    """Side
    Describes which foot to use
    """
    LEFT=0
    RIGHT=1

def other_foot_id(id):
    if id == Side.LEFT:
        return Side.RIGHT
    else:
        return Side.LEFT
        
class FootStep:
    """FootStep
    Holds all information describing a single footstep
    """
    def __init__(self, pose, footprint, side=Side.LEFT, time=0.0):
        """init FootStep

        Args:
            pose (pin.SE3): the pose of the footstep
            footprint (np.array): 3 by n matrix of foot vertices
            side (Side, optional): Foot identifier. Defaults to Side.LEFT.
            time (float, optional): Time when this step should be executed. Defaults to 0.0.
        """
        self.pose = pose
        self.footprint = footprint
        self.side = side
        self.time = time
        
    def poseInWorld(self):
        return self.pose
    
    def position(self):
        """Get position as numpy array [x, y, z]"""
        return self.pose.translation
    
    def to_dict(self):
        """Convert to dictionary format for compatibility"""
        return {
            'position': self.pose.translation.tolist(),
            'side': self.side,
            'time': self.time
        }
        
    def plot(self, simulation):
        """Plot footstep in PyBullet with enhanced visualization"""
        T = self.pose
        footprint_world = T.rotation @ self.footprint + T.translation.reshape(3,1)
        
        # Choose colors: Green for left, Blue for right
        if self.side == Side.LEFT:
            color = [0, 0.8, 0.2, 0.8]  # Green with transparency
            line_color = [0, 0.6, 0]    # Darker green for lines
        else:
            color = [0.2, 0.2, 1, 0.8]  # Blue with transparency
            line_color = [0, 0, 0.8]    # Darker blue for lines
        
        # Draw footprint outline
        for i in range(self.footprint.shape[1]):
            p.addUserDebugLine(
                footprint_world[:, i], 
                footprint_world[:, (i+1) % self.footprint.shape[1]], 
                line_color, 
                lineWidth=3.0,
                lifeTime=0
            )
        
        # Add side label
        p.addUserDebugText(
            "L" if self.side == Side.LEFT else "R", 
            T.translation + np.array([0, 0, 0.05]), 
            textColorRGB=line_color, 
            textSize=1.2,
            lifeTime=0
        )
        
        # Add step number if available
        step_info = f"t={self.time:.1f}s"
        p.addUserDebugText(
            step_info,
            T.translation + np.array([0, 0, 0.08]),
            textColorRGB=[0.2, 0.2, 0.2],
            textSize=0.8,
            lifeTime=0
        )
        
        # Plot target position with vertical line
        p.addUserDebugLine(
            T.translation, 
            T.translation + np.array([0, 0, 0.1]), 
            [1, 0, 0], 
            lineWidth=2.0,
            lifeTime=0
        )
        
        # Add small filled rectangle for better visibility
        self._add_filled_footstep(T, color)

    def _add_filled_footstep(self, pose, color):
        """Add a filled rectangular footstep for better visualization"""
        try:
            # Create a small box visual shape
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.1, 0.05, 0.005],  # foot dimensions
                rgbaColor=color
            )
            
            # Create the visual object
            p.createMultiBody(
                baseMass=0,  # No mass (visual only)
                baseVisualShapeIndex=visual_shape,
                basePosition=pose.translation + np.array([0, 0, 0.005])  # Slightly above ground
            )
        except:
            pass  # Fallback if creation fails

class FootStepPlanner:
    """FootStepPlanner
    Creates footstep plans (list of right and left steps)
    """
    
    def __init__(self, conf):
        self.conf = conf
        self.steps = []
        self.debug_ids = []  # Store debug visualization IDs for cleanup
        
    def planLine(self, T_0_w, side, no_steps, step_duration=0.8):
        """plan a sequence of steps in a straight line

        Args:
            T_0_w (pin.SE3): The initial starting position of the plan
            side (Side): The initial foot for starting the plan
            no_steps (int): The number of steps to take
            step_duration (float): Duration of each step in seconds

        Returns:
            list: sequence of steps
        """
        
        # the displacement between steps in x and y direction
        dx = self.conf.step_size_x
        dy = 2 * self.conf.step_size_y
        
        # the footprint of the robot
        lfxp, lfxn = self.conf.lfxp, self.conf.lfxn
        lfyp, lfyn = self.conf.lfyp, self.conf.lfyn
        
        footprint = np.array([
            [lfxp, lfxp, lfxn, lfxn],
            [lfyp, lfyn, lfyn, lfyp],
            [0.0,  0.0,  0.0,  0.0]
        ])

        steps = []
        current_time = 0.0

        # Starting stance: left and right foot next to each other
        for s in [Side.LEFT, Side.RIGHT]:
            offset_y = dy/2 if s == Side.LEFT else -dy/2
            pose = T_0_w * pin.SE3(np.eye(3), np.array([0, offset_y, 0]))
            steps.append(FootStep(pose, footprint, s, current_time))

        # Intermediate steps
        for i in range(no_steps):
            current_time = (i + 1) * step_duration
            step_side = Side.LEFT if i % 2 == 0 else Side.RIGHT
            offset_y = dy/2 if step_side == Side.LEFT else -dy/2
            offset = np.array([(i+1)*dx, offset_y, 0])
            pose = T_0_w * pin.SE3(np.eye(3), offset)
            steps.append(FootStep(pose, footprint, step_side, current_time))

        # Ending stance: add other foot next to final step
        if steps:
            last_pose = steps[-1].pose
            other_side = other_foot_id(steps[-1].side)
            offset_y = dy if other_side == Side.LEFT else -dy
            final_pose = last_pose * pin.SE3(np.eye(3), np.array([0, offset_y, 0]))
            final_time = current_time + step_duration
            steps.append(FootStep(final_pose, footprint, other_side, final_time))
                                
        self.steps = steps
        return steps
    
    def planCircle(self, T_0_w, radius, no_steps, step_duration=0.8):
        """Plan a circular walking pattern

        Args:
            T_0_w (pin.SE3): The initial starting position
            radius (float): Radius of the circle
            no_steps (int): Number of steps
            step_duration (float): Duration of each step

        Returns:
            list: sequence of steps
        """
        footprint = np.array([
            [self.conf.lfxp, self.conf.lfxp, self.conf.lfxn, self.conf.lfxn],
            [self.conf.lfyp, self.conf.lfyn, self.conf.lfyn, self.conf.lfyp],
            [0.0, 0.0, 0.0, 0.0]
        ])
        
        steps = []
        angle_increment = 2 * np.pi / no_steps
        dy = 2 * self.conf.step_size_y
        
        for i in range(no_steps + 2):  # +2 for initial stance
            if i < 2:
                # Initial stance
                angle = 0
                offset_y = dy/2 if i == 0 else -dy/2  # LEFT, RIGHT
                side = Side.LEFT if i == 0 else Side.RIGHT
                current_time = 0.0
            else:
                # Circular steps
                angle = (i - 2) * angle_increment
                side = Side.LEFT if (i - 2) % 2 == 0 else Side.RIGHT
                offset_y = dy/2 if side == Side.LEFT else -dy/2
                current_time = (i - 1) * step_duration
            
            # Calculate position on circle
            x = radius * np.cos(angle)
            y = radius * np.sin(angle) + offset_y
            
            # Create rotation matrix for orientation
            rotation = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            
            pose = T_0_w * pin.SE3(rotation, np.array([x, y, 0]))
            steps.append(FootStep(pose, footprint, side, current_time))
        
        self.steps = steps
        return steps
    
    def to_dict_list(self):
        """Convert steps to dictionary list for compatibility with other code"""
        return [step.to_dict() for step in self.steps]
    
    def clear_visualization(self):
        """Clear all visualization elements"""
        for debug_id in self.debug_ids:
            try:
                p.removeUserDebugItem(debug_id)
            except:
                pass
        self.debug_ids.clear()
    
    def plot(self, simulation):
        """Plot all footsteps with path connection"""
        self.clear_visualization()
        
        # Plot individual footsteps
        for step in self.steps:
            step.plot(simulation)
        
        # Plot path connections
        if len(self.steps) > 1:
            for i in range(len(self.steps) - 1):
                start_pos = self.steps[i].position()
                end_pos = self.steps[i + 1].position()
                
                # Draw connection line
                line_id = p.addUserDebugLine(
                    start_pos + np.array([0, 0, 0.02]),
                    end_pos + np.array([0, 0, 0.02]),
                    lineColorRGB=[0.8, 0.8, 0],  # Yellow path
                    lineWidth=2.0,
                    lifeTime=0
                )
                self.debug_ids.append(line_id)
        
        print(f"Footstep Plan Summary:")
        print(f"   Total steps: {len(self.steps)}")
        print(f"   Duration: {self.steps[-1].time:.1f}s")
        print(f"   Path length: {self._calculate_path_length():.2f}m")
    
    def _calculate_path_length(self):
        """Calculate total path length"""
        if len(self.steps) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(self.steps) - 1):
            pos1 = self.steps[i].position()
            pos2 = self.steps[i + 1].position()
            total_length += np.linalg.norm(pos2 - pos1)
        
        return total_length
    
    def print_plan(self):
        """Print detailed plan information"""
        print(f"\nDetailed Footstep Plan:")
        print(f"{'Step':<4} {'Side':<5} {'X':<8} {'Y':<8} {'Time':<8}")
        print("-" * 40)
        for i, step in enumerate(self.steps):
            pos = step.position()
            side_str = "LEFT" if step.side == Side.LEFT else "RIGHT"
            print(f"{i:<4} {side_str:<5} {pos[0]:<8.3f} {pos[1]:<8.3f} {step.time:<8.1f}")
            
if __name__=='__main__':
    """ Test footstep planner with enhanced features
    """
    
    class Config:
        def __init__(self):
            self.step_size_x = 0.15  # Slightly smaller steps
            self.step_size_y = 0.095  # TALOS-appropriate width
            self.lfxp = 0.1
            self.lfxn = -0.1
            self.lfyp = 0.05
            self.lfyn = -0.05

    # Initialize PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    # Set camera for better view
    p.resetDebugVisualizerCamera(
        cameraDistance=3.0,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[1, 0, 0]
    )
    
    conf = Config()
    planner = FootStepPlanner(conf)
    
    # Test linear plan
    T0 = pin.SE3(np.eye(3), np.array([0, 0, 0]))
    linear_steps = planner.planLine(T0, Side.LEFT, 8, step_duration=0.8)
    
    # Plot and analyze
    planner.plot(p)
    planner.print_plan()
    
    # Test conversion to dictionary format
    dict_plan = planner.to_dict_list()
    print(f"\nConverted to dictionary format: {len(dict_plan)} steps")
    
    # Add some interactive features
    print(f"\nInteractive Features:")
    print(f"   Press 'c' to test circular pattern")
    print(f"   Press 'r' to reset to linear pattern")
    print(f"   Press Ctrl+C to exit")
    
    try:
        while True:
            # Simple keyboard interaction
            keys = p.getKeyboardEvents()
            
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                print("Switching to circular pattern...")
                planner.clear_visualization()
                planner.planCircle(T0, radius=1.0, no_steps=12)
                planner.plot(p)
                planner.print_plan()
            
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                print("Switching back to linear pattern...")
                planner.clear_visualization()
                planner.planLine(T0, Side.LEFT, 8)
                planner.plot(p)
                planner.print_plan()
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print(f"\nExiting footstep planner test")
        planner.clear_visualization()
        p.disconnect()