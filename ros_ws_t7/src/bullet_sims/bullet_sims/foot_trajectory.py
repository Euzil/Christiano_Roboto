import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

class SwingFootTrajectory:
    """SwingFootTrajectory
    Interpolate Foot trajectory between SE3 T0 and T1
    """
    def __init__(self, T0, T1, duration, height=0.05):
        """initialize SwingFootTrajectory 
        Args:
            T0 (pin.SE3): Inital foot pose
            T1 (pin.SE3): Final foot pose
            duration (float): step duration
            height (float, optional): setp height. Defaults to 0.05.
        """
        self._height = height
        self._t_elapsed = 0.0
        self._duration = duration
        self.reset(T0, T1)
     
    def reset(self, T0, T1):
        '''reset back to zero, update poses
        '''
        #>>>>TODO: plan the spline
        self._t_elapsed = 0.0
        self._T0 = T0.copy()
        self._T1 = T1.copy()
        
        # 提取起始和结束位置、旋转
        self._p0 = T0.translation
        self._p1 = T1.translation
        self._R0 = T0.rotation
        self._R1 = T1.rotation
        
        # 计算5次多项式系数用于位置插值
        # 边界条件: p(0)=p0, p(T)=p1, dp(0)=0, dp(T)=0, ddp(0)=0, ddp(T)=0
        # 对于Z方向，额外约束: p(T/2) = p0[2] + height
        
        T = self._duration
        
        # X和Y方向的5次多项式系数 (满足位置和速度边界条件)
        # p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        self._coeffs_x = self._compute_position_coeffs(self._p0[0], self._p1[0], T)
        self._coeffs_y = self._compute_position_coeffs(self._p0[1], self._p1[1], T)
        
        # Z方向需要额外考虑中点高度约束
        self._coeffs_z = self._compute_height_coeffs(self._p0[2], self._p1[2], self._height, T)
        
        # 为旋转插值预计算四元数
        self._q0 = pin.Quaternion(self._R0)
        self._q1 = pin.Quaternion(self._R1)
        
    def _compute_position_coeffs(self, p0, p1, T):
        """计算5次多项式系数 (X, Y方向)"""
        # 边界条件矩阵
        # [1   0   0   0    0    0  ] [a0]   [p0]
        # [1   T   T^2 T^3  T^4  T^5] [a1]   [p1]
        # [0   1   0   0    0    0  ] [a2] = [0 ]
        # [0   1   2T  3T^2 4T^3 5T^4] [a3]   [0 ]
        # [0   0   2   0    0    0  ] [a4]   [0 ]
        # [0   0   2   6T   12T^2 20T^3][a5]  [0 ]
        
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2, 0, 0, 0],
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]
        ])
        
        b = np.array([p0, p1, 0, 0, 0, 0])
        
        return np.linalg.solve(A, b)
    
    def _compute_height_coeffs(self, z0, z1, height, T):
        """计算Z方向的多项式系数 (包含中点高度约束)"""
        # 边界条件 + 中点高度约束
        # p(0) = z0, p(T) = z1, p(T/2) = z0 + height
        # dp(0) = 0, dp(T) = 0, ddp(0) = 0
        
        T_half = T / 2
        
        A = np.array([
            [1, 0, 0, 0, 0, 0],  # p(0) = z0
            [1, T, T**2, T**3, T**4, T**5],  # p(T) = z1
            [1, T_half, T_half**2, T_half**3, T_half**4, T_half**5],  # p(T/2) = z0 + height
            [0, 1, 0, 0, 0, 0],  # dp(0) = 0
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],  # dp(T) = 0
            [0, 0, 2, 0, 0, 0]   # ddp(0) = 0
        ])
        
        b = np.array([z0, z1, z0 + height, 0, 0, 0])
        
        return np.linalg.solve(A, b)
    
    def _eval_polynomial(self, coeffs, t):
        """计算多项式及其导数值"""
        # p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        # dp(t) = a1 + 2*a2*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4
        # ddp(t) = 2*a2 + 6*a3*t + 12*a4*t^2 + 20*a5*t^3
        
        t_powers = np.array([1, t, t**2, t**3, t**4, t**5])
        position = np.dot(coeffs, t_powers)
        
        if t == 0:
            velocity = coeffs[1]
            acceleration = 2 * coeffs[2]
        else:
            dt_powers = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4])
            velocity = np.dot(coeffs, dt_powers)
            
            ddt_powers = np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3])
            acceleration = np.dot(coeffs, ddt_powers)
        
        return position, velocity, acceleration
    
    def _slerp(self, q0, q1, t):
        """球面线性插值 (SLERP)"""
        # 计算四元数点积
        dot = q0.x * q1.x + q0.y * q1.y + q0.z * q1.z + q0.w * q1.w
        
        # 如果点积为负，取相反四元数确保最短路径
        if dot < 0.0:
            q1 = pin.Quaternion(-q1.x, -q1.y, -q1.z, -q1.w)
            dot = -dot
        
        # 如果四元数非常接近，使用线性插值
        if dot > 0.9995:
            result = pin.Quaternion(
                q0.x + t * (q1.x - q0.x),
                q0.y + t * (q1.y - q0.y),
                q0.z + t * (q1.z - q0.z),
                q0.w + t * (q1.w - q0.w)
            )
            result.normalize()
            return result
        
        # 标准SLERP
        theta_0 = np.arccos(np.abs(dot))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return pin.Quaternion(
            s0 * q0.x + s1 * q1.x,
            s0 * q0.y + s1 * q1.y,
            s0 * q0.z + s1 * q1.z,
            s0 * q0.w + s1 * q1.w
        )
     
    def isDone(self):
        return self._t_elapsed >= self._duration
          
    def evaluate(self, t):
        """evaluate at time t
        """
        #>>>>TODO: evaluate the spline at time t, return pose, velocity, acceleration
        # 限制时间范围
        t_clamped = np.clip(t, 0, self._duration)
        
        # 计算位置、速度、加速度
        if t <= 0:
            position = self._p0.copy()
            velocity = np.zeros(6)
            acceleration = np.zeros(6)
            rotation = self._R0.copy()
        elif t >= self._duration:
            position = self._p1.copy()
            velocity = np.zeros(6)
            acceleration = np.zeros(6)
            rotation = self._R1.copy()
        else:
            # 计算位置分量
            x, vx, ax = self._eval_polynomial(self._coeffs_x, t_clamped)
            y, vy, ay = self._eval_polynomial(self._coeffs_y, t_clamped)
            z, vz, az = self._eval_polynomial(self._coeffs_z, t_clamped)
            
            position = np.array([x, y, z])
            v_linear = np.array([vx, vy, vz])
            a_linear = np.array([ax, ay, az])
            
            # 旋转插值
            t_normalized = t_clamped / self._duration
            q_interp = self._slerp(self._q0, self._q1, t_normalized)
            rotation = q_interp.toRotationMatrix()
            
            # 角速度计算 (数值微分)
            dt = 0.001
            if t_clamped + dt <= self._duration:
                t_next_norm = (t_clamped + dt) / self._duration
                q_next = self._slerp(self._q0, self._q1, t_next_norm)
                
                # 计算角速度向量
                q_diff = pin.Quaternion(q_next.x - q_interp.x, q_next.y - q_interp.y, 
                                       q_next.z - q_interp.z, q_next.w - q_interp.w)
                q_conj = q_interp.conjugate()
                q_omega = pin.Quaternion(
                    2.0 * (q_conj.w * q_diff.x - q_conj.x * q_diff.w - q_conj.y * q_diff.z + q_conj.z * q_diff.y) / dt,
                    2.0 * (q_conj.w * q_diff.y + q_conj.x * q_diff.z - q_conj.y * q_diff.w - q_conj.z * q_diff.x) / dt,
                    2.0 * (q_conj.w * q_diff.z - q_conj.x * q_diff.y + q_conj.y * q_diff.x - q_conj.z * q_diff.w) / dt,
                    0.0
                )
                v_angular = np.array([q_omega.x, q_omega.y, q_omega.z])
            else:
                v_angular = np.zeros(3)
            
            # 角加速度 (简化为零)
            a_angular = np.zeros(3)
            
            velocity = np.concatenate([v_linear, v_angular])
            acceleration = np.concatenate([a_linear, a_angular])
        
        # 创建SE3位姿
        pose = pin.SE3(rotation, position)
        
        # 更新经过时间
        self._t_elapsed = t
        
        return pose, velocity, acceleration

if __name__=="__main__":
    T0 = pin.SE3(np.eye(3), np.array([0, 0, 0]))
    T1 = pin.SE3(np.eye(3), np.array([0.2, 0, 0]))
    
    #>>>>TODO: plot to make sure everything is correct
    # 创建轨迹
    duration = 0.8
    height = 0.08
    trajectory = SwingFootTrajectory(T0, T1, duration, height)
    
    # 时间序列
    t = np.linspace(0, duration, 100)
    
    # 评估轨迹
    poses = []
    velocities = []
    accelerations = []
    
    for ti in t:
        pose, vel, acc = trajectory.evaluate(ti)
        poses.append(pose)
        velocities.append(vel)
        accelerations.append(acc)
    
    # 提取数据用于绘图
    positions = np.array([pose.translation for pose in poses])
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    
    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 3D轨迹
    ax1 = plt.figure().add_subplot(111, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax1.scatter([T0.translation[0]], [T0.translation[1]], [T0.translation[2]], 
               c='green', s=100, label='Start')
    ax1.scatter([T1.translation[0]], [T1.translation[1]], [T1.translation[2]], 
               c='red', s=100, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Foot Trajectory')
    ax1.legend()
    
    # 位置分量
    axes[0, 0].plot(t, positions[:, 0], 'r-', label='X')
    axes[0, 0].plot(t, positions[:, 1], 'g-', label='Y')
    axes[0, 0].plot(t, positions[:, 2], 'b-', label='Z')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].set_title('Position Components')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 线性速度
    axes[0, 1].plot(t, velocities[:, 0], 'r-', label='Vx')
    axes[0, 1].plot(t, velocities[:, 1], 'g-', label='Vy')
    axes[0, 1].plot(t, velocities[:, 2], 'b-', label='Vz')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Linear Velocity (m/s)')
    axes[0, 1].set_title('Linear Velocity')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 线性加速度
    axes[0, 2].plot(t, accelerations[:, 0], 'r-', label='Ax')
    axes[0, 2].plot(t, accelerations[:, 1], 'g-', label='Ay')
    axes[0, 2].plot(t, accelerations[:, 2], 'b-', label='Az')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Linear Acceleration (m/s²)')
    axes[0, 2].set_title('Linear Acceleration')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # 角速度
    axes[1, 0].plot(t, velocities[:, 3], 'r-', label='ωx')
    axes[1, 0].plot(t, velocities[:, 4], 'g-', label='ωy')
    axes[1, 0].plot(t, velocities[:, 5], 'b-', label='ωz')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Angular Velocity (rad/s)')
    axes[1, 0].set_title('Angular Velocity')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 边界条件验证
    axes[1, 1].plot([0, duration/2, duration], 
                    [positions[0, 2], positions[len(t)//2, 2], positions[-1, 2]], 
                    'bo-', markersize=8)
    axes[1, 1].axhline(y=T0.translation[2] + height, color='r', linestyle='--', 
                       label=f'Target height: {T0.translation[2] + height:.3f}')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Z Position (m)')
    axes[1, 1].set_title('Height Verification')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 速度边界条件
    axes[1, 2].plot(t, np.linalg.norm(velocities[:, :3], axis=1), 'b-', linewidth=2)
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Speed (m/s)')
    axes[1, 2].set_title('Linear Speed')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 验证边界条件
    print("=== 边界条件验证 ===")
    _, vel_start, acc_start = trajectory.evaluate(0)
    _, vel_end, acc_end = trajectory.evaluate(duration)
    
    print(f"起始线性速度: {vel_start[:3]}")
    print(f"结束线性速度: {vel_end[:3]}")
    print(f"起始线性加速度: {acc_start[:3]}")
    print(f"结束线性加速度: {acc_end[:3]}")
    
    # 验证最高点
    mid_pose, _, _ = trajectory.evaluate(duration/2)
    print(f"中点高度: {mid_pose.translation[2]:.3f} m (目标: {T0.translation[2] + height:.3f} m)")