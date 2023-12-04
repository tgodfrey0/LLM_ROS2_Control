import rclpy

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Vector3
from typing import List

THRESHOLD_CM = 30
    
class ScanSubscriber(Node):
    def __init__(self):
      super().__init__('scan_subscriber')
      self.subscription = self.create_subscription(
        LaserScan,
        "/scan",
        self.listener_callback,
        10
      )
      self.subscription
    
    def listener_callback(self, msg):
      rs: List[float] = msg.ranges
      less_than_min = False
      
      self.get_logger().info(f"Ranges: {rs}")
      
      for r in rs:
        less_than_min = r < THRESHOLD_CM
        
      if(less_than_min):
        zero_vel()
        
class TwistPublisher(Node):
  def __init__(self):
    super().__init__('twist_publisher')
    self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
    period = 0.1
    self.timer = self.create_timer(period, self.timer_callback)
  
  def timer_callback(self):
    msg = Twist()
    lin = Vector3()
    ang = Vector3()
    
    lin.x = 0
    lin.y = 0
    lin.z = 0
    
    ang.x = 0
    ang.y = 0
    ang.z = 0

    msg.linear = lin
    msg.angular = ang
    
    self.publisher_.publish(msg)
    self.get_logger().error("Min LIDAR reading, sending 0 velocity")
    
    

def zero_vel():
  twist_publisher = TwistPublisher()
  rclpy.spin(twist_publisher)
  twist_publisher.destroy_node()
  rclpy.shutdown()
   
def main(args=None):
  rclpy.init(args=args)
  
  scan_subscriber = ScanSubscriber()
    
  rclpy.spin(scan_subscriber)
  
  scan_subscriber.destroy_node()
  rclpy.shutdown()
   
if __name__ == '__main__':
  main()