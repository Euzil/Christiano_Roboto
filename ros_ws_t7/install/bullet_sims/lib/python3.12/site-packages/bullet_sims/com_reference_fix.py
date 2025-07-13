"""
COM Reference Fix Module
This module provides multiple approaches to fix the COM reference issue
that was preventing proper weight shifting during walking
"""

import numpy as np
import pinocchio as pin

class COMReferenceFixer:
    """
    Helper class to fix COM reference setting issues in TSID
    Provides multiple fallback methods to ensure COM references work
    """
    
    def __init__(self, robot, tsid_wrapper, verbose=True):
        self.robot = robot
        self.tsid_wrapper = tsid_wrapper
        self.verbose = verbose
        
        # Test different COM reference methods to find working ones
        self.working_methods = []
        self._test_com_methods()
    
    def _test_com_methods(self):
        """Test different COM reference methods to find which ones work"""
        if self.verbose:
            print("Testing COM reference methods...")
        
        # Method 1: Direct TSID wrapper
        if hasattr(self.tsid_wrapper, 'setComReference'):
            self.working_methods.append('tsid_wrapper_direct')
            if self.verbose:
                print("  ✓ Method 1: tsid_wrapper.setComReference() available")
        
        # Method 2: Robot stack
        if hasattr(self.robot, 'stack') and hasattr(self.robot.stack, 'setComReference'):
            self.working_methods.append('robot_stack_direct')
            if self.verbose:
                print("  ✓ Method 2: robot.stack.setComReference() available")
        
        # Method 3: Task-based approach
        if hasattr(self.robot, 'stack') and hasattr(self.robot.stack, 'setTaskReference'):
            self.working_methods.append('task_based')
            if self.verbose:
                print("  ✓ Method 3: Task-based setTaskReference() available")
        
        # Method 4: Formulation-based approach
        if (hasattr(self.robot, 'stack') and 
            hasattr(self.robot.stack, 'formulation') and
            hasattr(self.robot.stack.formulation, 'comTask')):
            self.working_methods.append('formulation_based')
            if self.verbose:
                print("  ✓ Method 4: Formulation-based comTask available")
        
        # Method 5: Alternative naming conventions
        alternative_methods = ['setCenterOfMassReference', 'setCoMReference', 'updateComReference']
        for method_name in alternative_methods:
            if hasattr(self.robot.stack, method_name):
                self.working_methods.append(f'alternative_{method_name}')
                if self.verbose:
                    print(f"  ✓ Method 5: {method_name}() available")
        
        if self.verbose:
            print(f"  Found {len(self.working_methods)} working methods")
    
    def setComReference(self, pos_ref, vel_ref=None, acc_ref=None):
        """
        Set COM reference using the first available working method
        """
        # Set defaults
        if vel_ref is None:
            vel_ref = np.zeros(3)
        if acc_ref is None:
            acc_ref = np.zeros(3)
        
        # Ensure numpy arrays
        pos_ref = np.asarray(pos_ref, dtype=np.float64)
        vel_ref = np.asarray(vel_ref, dtype=np.float64)
        acc_ref = np.asarray(acc_ref, dtype=np.float64)
        
        success = False
        
        for method in self.working_methods:
            try:
                if method == 'tsid_wrapper_direct':
                    self.tsid_wrapper.setComReference(pos_ref, vel_ref, acc_ref)
                    success = True
                    break
                
                elif method == 'robot_stack_direct':
                    self.robot.stack.setComReference(pos_ref, vel_ref, acc_ref)
                    success = True
                    break
                
                elif method == 'task_based':
                    self.robot.stack.setTaskReference("com", pos_ref, vel_ref, acc_ref)
                    success = True
                    break
                
                elif method == 'formulation_based':
                    self.robot.stack.formulation.comTask.setReference(pos_ref, vel_ref, acc_ref)
                    success = True
                    break
                
                elif method.startswith('alternative_'):
                    method_name = method.replace('alternative_', '')
                    method_func = getattr(self.robot.stack, method_name)
                    method_func(pos_ref, vel_ref, acc_ref)
                    success = True
                    break
                    
            except Exception as e:
                if self.verbose:
                    print(f"  Method {method} failed: {e}")
                continue
        
        if not success:
            if self.verbose:
                print(f"  Warning: All COM reference methods failed. Storing for manual handling.")
            # Store reference for manual handling
            self.robot.com_pos_ref = pos_ref
            self.robot.com_vel_ref = vel_ref
            self.robot.com_acc_ref = acc_ref
        
        return success

def patch_robot_com_reference(robot, tsid_wrapper):
    """
    Patch a robot instance with fixed COM reference methods
    """
    # Create the fixer
    com_fixer = COMReferenceFixer(robot, tsid_wrapper, verbose=True)
    
    # Patch the robot's _setComReference method
    def _setComReference_fixed(pos_ref, vel_ref=None, acc_ref=None):
        return com_fixer.setComReference(pos_ref, vel_ref, acc_ref)
    
    def setComReference_fixed(pos_ref, vel_ref=None, acc_ref=None):
        return com_fixer.setComReference(pos_ref, vel_ref, acc_ref)
    
    # Patch both methods
    robot._setComReference = _setComReference_fixed
    robot.setComReference = setComReference_fixed
    robot.com_fixer = com_fixer
    
    print("✓ Robot patched with COM reference fix")
    return robot

def create_alternative_com_controller(robot, tsid_wrapper):
    """
    Create an alternative COM controller that works around TSID limitations
    """
    
    class AlternativeCOMController:
        def __init__(self, robot, tsid_wrapper):
            self.robot = robot
            self.tsid_wrapper = tsid_wrapper
            self.com_target = None
            self.com_kp = 50.0  # Proportional gain
            self.com_kd = 10.0  # Derivative gain
            
        def setTarget(self, pos_ref, vel_ref=None, acc_ref=None):
            """Set COM target for alternative controller"""
            self.com_target = np.asarray(pos_ref, dtype=np.float64)
            if vel_ref is not None:
                self.com_vel_target = np.asarray(vel_ref, dtype=np.float64)
            else:
                self.com_vel_target = np.zeros(3)
        
        def update(self):
            """Update the alternative COM controller"""
            if self.com_target is None:
                return np.zeros(3)
            
            try:
                # Get current COM state
                com_current = self.robot.stack.getCenterOfMass()
                com_vel_current = self.robot.stack.getCenterOfMassVelocity()
                
                # Compute desired COM acceleration using PD control
                com_error = self.com_target - com_current
                com_vel_error = self.com_vel_target - com_vel_current
                
                com_acc_desired = self.com_kp * com_error + self.com_kd * com_vel_error
                
                # Limit acceleration
                max_acc = 2.0  # m/s^2
                com_acc_norm = np.linalg.norm(com_acc_desired)
                if com_acc_norm > max_acc:
                    com_acc_desired = com_acc_desired / com_acc_norm * max_acc
                
                return com_acc_desired
                
            except Exception as e:
                print(f"Alternative COM controller error: {e}")
                return np.zeros(3)
    
    controller = AlternativeCOMController(robot, tsid_wrapper)
    
    # Patch robot with alternative controller
    def setComReference_alternative(pos_ref, vel_ref=None, acc_ref=None):
        controller.setTarget(pos_ref, vel_ref, acc_ref)
    
    robot.setComReference = setComReference_alternative
    robot.alt_com_controller = controller
    
    print("✓ Robot patched with alternative COM controller")
    return robot

def debug_com_interface(robot, tsid_wrapper):
    """
    Debug function to understand the COM interface structure
    """
    print("\n=== COM Interface Debug ===")
    
    # Check TSID wrapper
    print("TSID Wrapper methods:")
    tsid_methods = [attr for attr in dir(tsid_wrapper) if 'com' in attr.lower()]
    for method in tsid_methods:
        print(f"  - tsid_wrapper.{method}")
    
    # Check robot stack
    if hasattr(robot, 'stack'):
        print("\nRobot Stack methods:")
        stack_methods = [attr for attr in dir(robot.stack) if 'com' in attr.lower()]
        for method in stack_methods:
            print(f"  - robot.stack.{method}")
    
    # Check for tasks
    if hasattr(robot, 'stack') and hasattr(robot.stack, 'tasks'):
        print("\nAvailable tasks:")
        try:
            if hasattr(robot.stack.tasks, 'keys'):
                for task_name in robot.stack.tasks.keys():
                    print(f"  - {task_name}")
            else:
                print("  - Tasks object doesn't have keys()")
        except:
            print("  - Could not access task keys")
    
    # Check formulation
    if hasattr(robot, 'stack') and hasattr(robot.stack, 'formulation'):
        print("\nFormulation attributes:")
        formulation_attrs = [attr for attr in dir(robot.stack.formulation) if 'com' in attr.lower() or 'task' in attr.lower()]
        for attr in formulation_attrs:
            print(f"  - robot.stack.formulation.{attr}")
    
    # Try to get current COM
    try:
        current_com = robot.stack.getCenterOfMass()
        print(f"\nCurrent COM: [{current_com[0]:.3f}, {current_com[1]:.3f}, {current_com[2]:.3f}]")
    except Exception as e:
        print(f"\nCOM retrieval error: {e}")
    
    print("=== Debug Complete ===\n")