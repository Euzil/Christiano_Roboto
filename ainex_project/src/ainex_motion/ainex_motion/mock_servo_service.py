import rclpy
from rclpy.node import Node

# Import the servo service interfaces
try:
    from servo_service.srv import JointPosition, JointRange, JointLock, ServoTemp, ServoDeviation, ServoVoltage
except ImportError as e:
    print(f"Could not import servo services: {e}")
    print("The servo_service package may not be properly built or sourced")

class MockServoServiceProvider(Node):
    def __init__(self):
        super().__init__('mock_servo_service_provider')
        
        # Create service servers for the services that joint_controller expects
        try:
            self.joint_position_srv = self.create_service(
                JointPosition, 'Joint_Position', self.joint_position_callback)
            self.joint_range_srv = self.create_service(
                JointRange, 'Joint_Range', self.joint_range_callback)
            self.joint_lock_srv = self.create_service(
                JointLock, 'Joint_Lock', self.joint_lock_callback)
            self.servo_temp_srv = self.create_service(
                ServoTemp, 'Servo_Temp', self.servo_temp_callback)
            self.servo_deviation_srv = self.create_service(
                ServoDeviation, 'Servo_Deviation', self.servo_deviation_callback)
            self.servo_voltage_srv = self.create_service(
                ServoVoltage, 'Servo_Voltage', self.servo_voltage_callback)
            
            self.get_logger().info('Mock servo service provider started with all services')
            
        except Exception as e:
            self.get_logger().error(f'Failed to create services: {e}')
    
    def joint_position_callback(self, request, response):
        # Return mock position data
        self.get_logger().info(f'Joint position request for joint: {request.joint_id}')
        response.position = 0.0  # Mock position
        response.success = True
        return response
    
    def joint_range_callback(self, request, response):
        # Return mock range data
        self.get_logger().info(f'Joint range request for joint: {request.joint_id}')
        response.min_position = -3.14
        response.max_position = 3.14
        response.success = True
        return response
    
    def joint_lock_callback(self, request, response):
        # Mock joint lock response
        self.get_logger().info(f'Joint lock request: {request.lock_state}')
        response.success = True
        return response
    
    def servo_temp_callback(self, request, response):
        response.temperature = 25.0  # Mock temperature
        response.success = True
        return response
    
    def servo_deviation_callback(self, request, response):
        response.deviation = 0.0  # Mock deviation
        response.success = True
        return response
    
    def servo_voltage_callback(self, request, response):
        response.voltage = 12.0  # Mock voltage
        response.success = True
        return response

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = MockServoServiceProvider()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()