import numpy as np
import pinocchio as pin
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot
import whole_body_control.config as conf
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped

class Ainex(Robot):
    def __init__(self, simulator, urdf, model, node, q=None, verbose=True, useFixedBase=False):
        self.simulator = simulator
        
        # Floating base für echten Roboter
        base_position = [0, 0, 0.23]

        super().__init__(
            simulator,
            urdf,
            model,
            basePosition=base_position,
            baseQuationerion=[0, 0, 0, 1],
            q=q,
            useFixedBase=useFixedBase,
            verbose=verbose,
        )
        self.node = node

        # ROS Publishers
        self.pub_joint = self.node.create_publisher(JointState, "/joint_states", 10)
        self.joint_msg = JointState()
        self.joint_msg.name = self.actuatedJointNames()
        
        self.br = tf2_ros.TransformBroadcaster(self.node)

    def update(self):
        super().update()

    def publish(self, tau):
        # Publish joint state
        self.joint_msg.header.stamp = self.node.get_clock().now().to_msg()
        self.joint_msg.position = self.actuatedJointPosition().tolist()
        self.joint_msg.velocity = self.actuatedJointVelocity().tolist()
        self.joint_msg.effort = tau.tolist()
        self.pub_joint.publish(self.joint_msg)

        # Publish base transform (von Simulation/IMU)
        q_base = self.q()[:7]
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.node.get_clock().now().to_msg()
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = self.baseName()

        tf_msg.transform.translation.x = q_base[0]
        tf_msg.transform.translation.y = q_base[1]
        tf_msg.transform.translation.z = q_base[2]
        
        tf_msg.transform.rotation.x = q_base[3]
        tf_msg.transform.rotation.y = q_base[4]
        tf_msg.transform.rotation.z = q_base[5]
        tf_msg.transform.rotation.w = q_base[6]

        self.br.sendTransform(tf_msg)


class VeryConservativeStandingController:
    """
    Extrem konservativer Controller - startet sehr sanft
    """
    
    def __init__(self, model, target_pose=None):
        self.model = model
        self.data = model.createData()
        
        # Target pose für Joints (ohne Base)
        if target_pose is None:
            self.target_pose = np.zeros(model.nq - 7)  # 24 joints
        else:
            self.target_pose = target_pose
            
        # Etwas niedrigere PD Gains
        self.kp_joints = np.array([
            # Right leg (6 joints) - etwas niedriger
            6., 5., 12., 16., 10., 5.,
            # Left leg (6 joints)
            6., 5., 12., 16., 10., 5.,
            # Head (2 joints) - sehr niedrig
            1.5, 1.5,
            # Right arm (5 joints) - niedriger
            3., 2.5, 2.5, 2.5, 1.5,
            # Left arm (5 joints)
            3., 2.5, 2.5, 2.5, 1.5
        ])
        
        self.kd_joints = 2.0 * np.sqrt(self.kp_joints) * 0.12  # Noch niedrigere Dämpfung
        
    def compute_control(self, q, v, time_factor=1.0):
        """
        Sehr konservative Torque-Berechnung
        """
        
        # Update Pinocchio model
        pin.computeAllTerms(self.model, self.data, q, v)
        
        # Extract joint state
        joint_pos = q[7:]  # Skip base (7 DOF)
        joint_vel = v[6:]  # Skip base (6 DOF)
        
        # Scaled gains
        current_kp = self.kp_joints * time_factor
        current_kd = self.kd_joints * time_factor
        
        # PD control
        joint_error = self.target_pose - joint_pos
        joint_error_dot = -joint_vel  # Target velocity = 0
        
        pd_torques = current_kp * joint_error + current_kd * joint_error_dot
        
        # Gravity compensation - volle Stärke aber begrenzt
        gravity_comp = self.data.g[6:]  # Skip 6 base DOFs
        
        # Kombiniere Gravity + PD
        tau = gravity_comp + pd_torques
        
        # Etwas niedrigere Torque-Limits
        max_torque = 12.0  # Von 15.0 auf 12.0 reduziert
        tau = np.clip(tau, -max_torque, max_torque)
        
        return tau
        
    def set_target_pose(self, target):
        """Set new target joint positions"""
        self.target_pose = target


class Environment(Node):
    def __init__(self):
        super().__init__('conservative_standing_node')

        # Floating base model
        self.model = pin.buildModelFromUrdf(conf.urdf, pin.JointModelFreeFlyer())
        
        self.simulator = PybulletWrapper()

        # Konservative Startposition - näher an der aktuellen Robot-Position
        q_init = np.hstack([
            np.array([0, 0, 0.23, 0, 0, 0, 1]),  # Start-Höhe
            conf.q_actuated_home * 0.25  # Nur 25% der Zielposition als Start (von 30% auf 25%)
        ])

        self.robot = Ainex(
            self.simulator,
            conf.urdf,
            self.model,
            self,
            q=q_init,
            verbose=True,
            useFixedBase=False)

        self.robot.enableTorqueControl()
        
        # Konservativer Controller
        self.controller = VeryConservativeStandingController(
            self.model, 
            target_pose=conf.q_actuated_home
        )

        self.get_logger().info("Very Conservative Standing Controller initialized")
        self.get_logger().info("Starting with very gentle approach")
        
        self.t_publish = 0.0
        self.start_time = 0.0
        self.stabilization_duration = 12.0  # Von 10 auf 12 Sekunden verlängert

    def update(self):
        t = self.simulator.simTime()
        
        if self.start_time == 0.0:
            self.start_time = t

        self.simulator.step()
        self.simulator.debug()
        self.robot.update()

        # Get robot state
        q = self.robot.q()
        v = self.robot.v()

        # Noch sanfterer Zeit-Faktor
        elapsed = t - self.start_time
        
        if elapsed < self.stabilization_duration:
            # Noch sanfteres Hochfahren: von 8% auf 55% über 12 Sekunden
            time_factor = 0.08 + (elapsed / self.stabilization_duration) * 0.47  # 8% -> 55%
            phase = "GENTLE RAMP-UP"
        else:
            # Nach 12 Sekunden: langsam auf 90% hochfahren (nicht 100%)
            time_factor = min(0.55 + (elapsed - self.stabilization_duration) / 20.0 * 0.35, 0.9)  # 55% -> 90%
            phase = "ACTIVE"

        # Compute conservative control
        tau_sol = self.controller.compute_control(q, v, time_factor)

        # Debug every 3 seconds
        if int(t * 333) % 1000 == 0:  # Alle 3 Sekunden
            self.debug_robot_state(q, v, tau_sol, time_factor, phase)

        # RELAXED Emergency stop - viel großzügigere Grenzen
        if abs(q[0]) > 10 or abs(q[1]) > 10 or q[2] < 0.02 or q[2] > 2.0:
            self.get_logger().error("EMERGENCY STOP - Robot really out of bounds!")
            self.get_logger().error(f"Position: x={q[0]:.3f}, y={q[1]:.3f}, z={q[2]:.3f}")
            tau_sol = np.zeros(24)

        # Send commands
        self.robot.setActuatedJointTorques(tau_sol)

        # Publish to ROS
        if t - self.t_publish > 1./30.:
            self.t_publish = t
            self.robot.publish(tau_sol)

    def debug_robot_state(self, q, v, tau, time_factor, phase):
        """Debug with conservative monitoring"""
        self.get_logger().info(f"=== {phase} ({time_factor:.1%} gains) ===")
        self.get_logger().info(f"Base position: [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}]")
        self.get_logger().info(f"Base velocity: [{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}] m/s")
        self.get_logger().info(f"Max torque: {np.max(np.abs(tau)):.1f} Nm")
        
        # Leg joint positions
        leg_joints = q[7:19]  # Erste 12 = beide Beine
        self.get_logger().info(f"Right leg joints: {leg_joints[:6]}")
        
        # Gentle warnings
        if abs(q[0]) > 2 or abs(q[1]) > 2:
            self.get_logger().warn(f"Base drifting: x={q[0]:.3f}, y={q[1]:.3f}")
        if q[2] < 0.1:
            self.get_logger().warn(f"Base height getting low: z={q[2]:.3f}")
        if q[2] > 1.0:
            self.get_logger().warn(f"Base height getting high: z={q[2]:.3f}")
        if np.max(np.abs(v[:3])) > 2.0:
            self.get_logger().warn(f"High base velocity: {np.max(np.abs(v[:3])):.3f} m/s")
        
        # Check for reasonable torques
        if np.max(np.abs(tau)) > 10:  # Von 12 auf 10 reduziert
            self.get_logger().warn(f"Higher torques: {np.max(np.abs(tau)):.1f} Nm")


def main(args=None):
    rclpy.init(args=args)
    env = Environment()
    try:
        while rclpy.ok():
            env.update()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
