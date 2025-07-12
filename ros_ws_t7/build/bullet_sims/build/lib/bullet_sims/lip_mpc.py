import numpy as np
from pydrake.all import MathematicalProgram, Solve

################################################################################
# Helper fnc
################################################################################

def continious_LIP_dynamics(g, h):
    """returns the static matrices A,B of the continious LIP dynamics
    """
    #>>>>TODO: Compute
    # LIP dynamics: 
    # x = [c_x, c_y, c_dot_x, c_dot_y, xi_x, xi_y]^T
    # u = [xi_dot_x, xi_dot_y]^T
    # where xi = c + c_dot/omega (DCM), omega = sqrt(g/h)
    
    omega = np.sqrt(g / h)
    
    # State: [c_x, c_y, c_dot_x, c_dot_y, xi_x, xi_y]
    A = np.array([
        [0, 0, 1, 0, 0, 0],           # c_dot_x
        [0, 0, 0, 1, 0, 0],           # c_dot_y  
        [omega**2, 0, 0, 0, -omega**2, 0],  # c_ddot_x = omega^2 * (c_x - xi_x)
        [0, omega**2, 0, 0, 0, -omega**2],  # c_ddot_y = omega^2 * (c_y - xi_y)
        [0, 0, 0, 0, 0, 0],           # xi_dot_x (control input)
        [0, 0, 0, 0, 0, 0]            # xi_dot_y (control input)
    ])
    
    # Control input: [xi_dot_x, xi_dot_y]
    B = np.array([
        [0, 0],
        [0, 0],
        [omega**2, 0],
        [0, omega**2],
        [1, 0],
        [0, 1]
    ])
    
    return A, B

def discrete_LIP_dynamics(g, h, dt):
    """returns the matrices static Ad,Bd of the discretized LIP dynamics
    """
    #>>>>TODO: Compute
    # Get continuous dynamics
    A, B = continious_LIP_dynamics(g, h)
    
    # Discretize using matrix exponential
    # x[k+1] = Ad * x[k] + Bd * u[k]
    
    n = A.shape[0]
    m = B.shape[1]
    
    # Create augmented matrix for discretization
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A * dt
    M[:n, n:] = B * dt
    
    # Matrix exponential
    exp_M = np.eye(n + m)
    M_power = np.eye(n + m)
    
    # Taylor series approximation for matrix exponential
    for i in range(1, 20):  # 20 terms should be sufficient
        M_power = M_power @ M / i
        exp_M += M_power
    
    A_d = exp_M[:n, :n]
    B_d = exp_M[:n, n:]
    
    return A_d, B_d

################################################################################
# LIPInterpolator
################################################################################

class LIPInterpolator:
    """Integrates the linear inverted pendulum model using the 
    continous dynamics. To interpolate the solution to hight
    """
    def __init__(self, x_inital, conf):
        self.conf = conf
        self.dt = conf.dt
        self.x = x_inital.copy()
        #>>>>TODO: Finish
        self.g = conf.g if hasattr(conf, 'g') else 9.81
        self.h = conf.h if hasattr(conf, 'h') else 0.8
        self.omega = np.sqrt(self.g / self.h)
        
        # Get discrete dynamics
        self.A_d, self.B_d = discrete_LIP_dynamics(self.g, self.h, self.dt)
        
    def integrate(self, u):
        #>>>>TODO: integrate with dt
        # x[k+1] = A_d * x[k] + B_d * u[k]
        u = np.array(u).reshape(-1)  # Ensure u is 1D array
        if len(u) == 1:
            u = np.array([u[0], 0])  # If only x-direction given, assume y=0
        
        self.x = self.A_d @ self.x + self.B_d @ u
        return self.x
        
    def comState(self):
        #>>>>TODO: return the center of mass state
        # that is position \in R3, velocity \in R3, acceleration \in R3
        
        # Extract from state vector [c_x, c_y, c_dot_x, c_dot_y, xi_x, xi_y]
        c_x, c_y = self.x[0], self.x[1]
        c_dot_x, c_dot_y = self.x[2], self.x[3]
        xi_x, xi_y = self.x[4], self.x[5]
        
        # Position (assume constant height)
        c = np.array([c_x, c_y, self.h])
        
        # Velocity 
        c_dot = np.array([c_dot_x, c_dot_y, 0])
        
        # Acceleration from LIP dynamics: c_ddot = omega^2 * (c - xi)
        c_ddot_x = self.omega**2 * (c_x - xi_x)
        c_ddot_y = self.omega**2 * (c_y - xi_y)
        c_ddot = np.array([c_ddot_x, c_ddot_y, 0])
        
        return c, c_dot, c_ddot
        
    def dcm(self):
        #>>>>TODO: return the computed dcm
        # DCM = c + c_dot/omega
        xi_x, xi_y = self.x[4], self.x[5]
        dcm = np.array([xi_x, xi_y])
        return dcm
        
    def zmp(self):
        #>>>>TODO: return the zmp
        # ZMP = c - c_ddot/omega^2 = c - (c - xi) = xi
        # But actually ZMP should be computed differently
        # ZMP = c - (c_ddot - g*e_z)/omega^2
        # For flat ground: ZMP = xi (DCM)
        xi_x, xi_y = self.x[4], self.x[5]
        zmp = np.array([xi_x, xi_y])
        return zmp
            
################################################################################
# LIPMPC
################################################################################

class LIPMPC:
    def __init__(self, conf):
        self.conf = conf
        self.dt = conf.dt
        self.no_samples = conf.no_mpc_samples_per_horizon
        
        # Get dynamics
        self.g = conf.g if hasattr(conf, 'g') else 9.81
        self.h = conf.h if hasattr(conf, 'h') else 0.8
        self.omega = np.sqrt(self.g / self.h)
        self.A_d, self.B_d = discrete_LIP_dynamics(self.g, self.h, self.dt)
        
        # solution and references over the horizon
        self.X_k = None
        self.U_k = None
        self.ZMP_ref_k = None
        
    def buildSolveOCP(self, x_k, ZMP_ref_k, terminal_idx=None):
        """build and solve ocp
        Args:
            x_k (_type_): inital mpc state
            ZMP_ref_k (_type_): zmp reference over horizon
            terminal_idx (_type_): index within horizon to apply terminal constraint
        Returns:
            _type_: control
        """
        
        #>>>>TODO: build and solve the ocp
        #>>>>Note: start without terminal constraints
        
        # Create optimization problem
        prog = MathematicalProgram()
        
        # Decision variables
        n_states = self.A_d.shape[0]  # 6 states
        n_controls = self.B_d.shape[1]  # 2 controls
        N = self.no_samples
        
        # State variables: X = [x_0, x_1, ..., x_N]
        X = prog.NewContinuousVariables(n_states, N + 1, "x")
        # Control variables: U = [u_0, u_1, ..., u_{N-1}]
        U = prog.NewContinuousVariables(n_controls, N, "u")
        
        # Initial condition constraint
        prog.AddLinearConstraint(X[:, 0] == x_k)
        
        # Dynamics constraints
        for k in range(N):
            prog.AddLinearConstraint(X[:, k+1] == self.A_d @ X[:, k] + self.B_d @ U[:, k])
        
        # Cost function
        cost = 0
        
        # Weight matrices
        Q_com = 100.0  # CoM position weight
        Q_vel = 1.0    # CoM velocity weight  
        Q_dcm = 1000.0 # DCM weight
        R = 0.1        # Control weight
        Q_zmp = 1000.0 # ZMP tracking weight
        
        for k in range(N):
            # State cost
            # CoM position cost
            cost += Q_com * (X[0, k]**2 + X[1, k]**2)
            # CoM velocity cost
            cost += Q_vel * (X[2, k]**2 + X[3, k]**2)
            # DCM cost (keep DCM reasonable)
            cost += Q_dcm * (X[4, k]**2 + X[5, k]**2)
            
            # Control cost
            cost += R * (U[0, k]**2 + U[1, k]**2)
            
            # ZMP tracking cost
            # ZMP = DCM for flat ground
            zmp_x, zmp_y = X[4, k], X[5, k]
            if k < len(ZMP_ref_k):
                if len(ZMP_ref_k[k]) >= 2:
                    zmp_ref_x, zmp_ref_y = ZMP_ref_k[k][0], ZMP_ref_k[k][1]
                else:
                    zmp_ref_x, zmp_ref_y = ZMP_ref_k[k][0], 0.0
            else:
                zmp_ref_x, zmp_ref_y = 0.0, 0.0
                
            cost += Q_zmp * ((zmp_x - zmp_ref_x)**2 + (zmp_y - zmp_ref_y)**2)
        
        # Terminal cost
        Q_terminal = 10.0
        cost += Q_terminal * (X[0, N]**2 + X[1, N]**2 + X[2, N]**2 + X[3, N]**2)
        
        # Add terminal constraint if specified
        if terminal_idx is not None and terminal_idx < N:
            # Terminal DCM constraint (optional)
            prog.AddLinearConstraint(X[4, terminal_idx] <= 0.5)
            prog.AddLinearConstraint(X[4, terminal_idx] >= -0.5)
            prog.AddLinearConstraint(X[5, terminal_idx] <= 0.2)
            prog.AddLinearConstraint(X[5, terminal_idx] >= -0.2)
        
        # Control constraints (optional)
        for k in range(N):
            prog.AddLinearConstraint(U[0, k] <= 5.0)
            prog.AddLinearConstraint(U[0, k] >= -5.0)
            prog.AddLinearConstraint(U[1, k] <= 5.0) 
            prog.AddLinearConstraint(U[1, k] >= -5.0)
        
        # Set objective
        prog.AddCost(cost)
        
        # Solve
        result = Solve(prog)
        
        if result.is_success():
            self.X_k = result.GetSolution(X)
            self.U_k = result.GetSolution(U)
            self.ZMP_ref_k = ZMP_ref_k
            
            return self.U_k[:, 0]  # Return first control
        else:
            print("MPC optimization failed!")
            return np.zeros(n_controls)

def generate_zmp_reference(foot_steps, no_samples_per_step):
    """generate the zmp reference given a sequence of footsteps
    """
    
    #>>>>TODO: use the previously footstep type to build the reference
    # trajectory for the zmp
    
    zmp_ref = []
    
    for i, footstep in enumerate(foot_steps):
        # Extract footstep position
        if hasattr(footstep, 'position'):
            pos = footstep.position
        elif isinstance(footstep, dict):
            pos = footstep['position']
        else:
            pos = footstep  # Assume it's already a position array
        
        # For each footstep, create constant ZMP reference
        for _ in range(no_samples_per_step):
            if len(pos) >= 2:
                zmp_ref.append([pos[0], pos[1]])
            else:
                zmp_ref.append([pos[0], 0.0])
    
    return np.array(zmp_ref)

# Test function
def test_lip_mpc():
    """Test the LIP MPC implementation"""
    
    # Configuration class
    class Config:
        def __init__(self):
            self.dt = 0.1
            self.no_mpc_samples_per_horizon = 16
            self.g = 9.81
            self.h = 0.8
    
    conf = Config()
    
    # Test continuous dynamics
    print("=== Testing Continuous Dynamics ===")
    A, B = continious_LIP_dynamics(conf.g, conf.h)
    print(f"A matrix shape: {A.shape}")
    print(f"B matrix shape: {B.shape}")
    print(f"Omega = {np.sqrt(conf.g/conf.h):.3f}")
    
    # Test discrete dynamics  
    print("\n=== Testing Discrete Dynamics ===")
    A_d, B_d = discrete_LIP_dynamics(conf.g, conf.h, conf.dt)
    print(f"A_d matrix shape: {A_d.shape}")
    print(f"B_d matrix shape: {B_d.shape}")
    
    # Test interpolator
    print("\n=== Testing LIP Interpolator ===")
    x_initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # [c_x, c_y, c_dot_x, c_dot_y, xi_x, xi_y]
    interpolator = LIPInterpolator(x_initial, conf)
    
    # Test integration
    u = np.array([0.1, 0.0])  # Small control input
    x_new = interpolator.integrate(u)
    print(f"New state after integration: {x_new}")
    
    # Test state extraction
    c, c_dot, c_ddot = interpolator.comState()
    print(f"CoM position: {c}")
    print(f"CoM velocity: {c_dot}")
    print(f"CoM acceleration: {c_ddot}")
    
    dcm = interpolator.dcm()
    zmp = interpolator.zmp()
    print(f"DCM: {dcm}")
    print(f"ZMP: {zmp}")
    
    # Test MPC
    print("\n=== Testing LIP MPC ===")
    mpc = LIPMPC(conf)
    
    # Create simple ZMP reference
    zmp_ref = []
    for i in range(conf.no_mpc_samples_per_horizon):
        zmp_ref.append([0.1 * i, 0.0])  # Moving forward
    
    # Solve MPC
    x_k = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u_opt = mpc.buildSolveOCP(x_k, zmp_ref)
    print(f"Optimal control: {u_opt}")
    
    # Test ZMP reference generation
    print("\n=== Testing ZMP Reference Generation ===")
    
    # Create sample footsteps
    class Footstep:
        def __init__(self, pos):
            self.position = pos
    
    footsteps = [
        Footstep([0.0, 0.1]),
        Footstep([0.2, -0.1]), 
        Footstep([0.4, 0.1]),
        Footstep([0.6, -0.1])
    ]
    
    zmp_ref_generated = generate_zmp_reference(footsteps, 10)
    print(f"Generated ZMP reference shape: {zmp_ref_generated.shape}")
    print(f"First few ZMP references: {zmp_ref_generated[:5]}")

if __name__ == "__main__":
    test_lip_mpc()