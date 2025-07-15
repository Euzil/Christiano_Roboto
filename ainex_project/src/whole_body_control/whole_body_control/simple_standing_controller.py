import numpy as np
import pinocchio as pin

class SimpleStandingController:
    """
    Einfacher Controller für das Stehen ohne Torque-Sensoren
    Verwendet Gravity Compensation + PD Control für Joint Positions
    """
    
    def __init__(self, model, target_pose=None):
        self.model = model
        self.data = model.createData()
        
        # Target pose (home position)
        if target_pose is None:
            self.target_pose = np.zeros(model.nq - 7)  # Nur Gelenke, ohne Base
        else:
            self.target_pose = target_pose
            
        # PD Gains für Gelenke
        self.kp_joints = np.array([
            # Right leg (6 joints) - höhere Gains für Beine
            200., 200., 300., 300., 200., 200.,
            # Left leg (6 joints)
            200., 200., 300., 300., 200., 200.,
            # Head (2 joints) - niedrigere Gains
            50., 50.,
            # Right arm (5 joints) - mittlere Gains
            100., 100., 100., 100., 50.,
            # Left arm (5 joints)
            100., 100., 100., 100., 50.
        ])
        
        self.kd_joints = 2.0 * np.sqrt(self.kp_joints) * 0.7  # Damping
        
    def compute_control(self, q, v):
        """
        Berechnet Torque-Befehle für das Stehen
        
        Args:
            q: Vollständige Konfiguration (base + joints)
            v: Vollständige Geschwindigkeit (base + joints)
            
        Returns:
            tau: Torque-Befehle für aktive Gelenke
        """
        
        # Update Pinocchio model
        pin.computeAllTerms(self.model, self.data, q, v)
        
        # Gravity compensation für alle Gelenke
        gravity_compensation = self.data.g
        
        # Joint positions und velocities (ohne Base)
        q_joints = q[7:]  # Skip base (7 DOF)
        v_joints = v[6:]  # Skip base (6 DOF)
        
        # PD Control für Gelenke
        joint_error = self.target_pose - q_joints
        joint_error_dot = -v_joints  # Target velocity = 0
        
        pd_torques = self.kp_joints * joint_error + self.kd_joints * joint_error_dot
        
        # Combine: Gravity + PD (nur für aktive Gelenke)
        tau = gravity_compensation[6:] + pd_torques  # Skip base DOF in gravity
        
        return tau
        
    def set_target_pose(self, target):
        """Setze neue Zielpose"""
        self.target_pose = target