from ainex_motion.joint_controller import JointController
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStatePublisher(Node):
    """
    Goal: Retreive the joint positions of the robot and publish as JointState messages
    """
    def __init__(self):
        super().__init__('joint_state_publisher')
        # instaintiate the JointController with the current node
        self.joint_controller = JointController(self)
        
        # TODO: define a publisher for the joint states 
        # with the topic name 'joint_states'
        # and message type sensor_msgs/JointState
        self.joint_states_pub = 

    def publish_joint_states(self):
        # TODO: Retrieve current joint positions with the getJointPositions method
        # and publish them to the 'joint_states' topic
        # Hint: Check the message definition here: https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/JointState.html
        # Set the joints velocity and effort to 0.0, header.stamp to the current time
        pass


def main(args=None):
    rclpy.init(args=args)

    joint_state_publisher = JointStatePublisher()

    while rclpy.ok():
        joint_state_publisher.publish_joint_states()

    joint_state_publisher.destroy_node()
    rclpy.shutdown()