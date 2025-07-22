from ainex_motion.joint_controller import JointController
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import time  # Import time module for frequency calculation

class JointStatePublisher(Node):
    """
    Goal: Retrieve the joint positions of the robot and publish as JointState messages
    """
    def __init__(self):
        super().__init__('joint_state_publisher')
        # Instantiate the JointController with the current node
        self.joint_controller = JointController(self)
        
        # Define a publisher for the joint states 
        # with the topic name 'joint_states'
        # and message type sensor_msgs/JointState
        self.joint_states_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Initialize variables for frequency calculation
        self.last_publish_time = None
        self.publish_count = 0

    def publish_joint_states(self):
        # Retrieve current joint positions with the getJointPositions method
        # and publish them to the 'joint_states' topic
        
        # Get all joint names from the joint controller
        joint_names = list(self.joint_controller.joint_id.keys())
        
        # Get current joint positions
        positions = self.joint_controller.getJointPositions(joint_names)
        
        if positions is not None:
            # Create JointState message
            joint_state = JointState()
            
            # Set header with current timestamp
            joint_state.header = Header()
            joint_state.header.stamp = self.get_clock().now().to_msg()
            
            # Set joint names and positions
            joint_state.name = joint_names
            joint_state.position = positions
            
            # Set velocity and effort to 0.0 as specified
            joint_state.velocity = [0.0] * len(joint_names)
            joint_state.effort = [0.0] * len(joint_names)
            
            # Publish the message
            self.joint_states_pub.publish(joint_state)
            self.get_logger().debug(f"Published joint states: {len(joint_names)} joints")
            
            # Calculate and log the publishing frequency
            current_time = time.time()
            if self.last_publish_time is not None:
                time_diff = current_time - self.last_publish_time
                frequency = 1.0 / time_diff
                self.get_logger().info(f"Publishing frequency: {frequency:.2f} Hz")
            self.last_publish_time = current_time
            self.publish_count += 1
        else:
            self.get_logger().warn("Failed to get joint positions")


def main(args=None):
    rclpy.init(args=args)

    joint_state_publisher = JointStatePublisher()

    try:
        while rclpy.ok():
            joint_state_publisher.publish_joint_states()
            rclpy.spin_once(joint_state_publisher, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        joint_state_publisher.destroy_node()
        rclpy.shutdown()