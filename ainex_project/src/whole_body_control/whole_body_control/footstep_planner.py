"""
Enhanced footstep_planner.py
Adds more functionality and better integration support
"""

import numpy as np
import pinocchio as pin
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
            