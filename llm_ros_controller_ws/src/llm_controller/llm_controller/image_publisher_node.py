import rclpy
import cv2 
import threading

from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge

TARGET_FPS: int = 20
TIMER_PERIOD: float = round(1/30)
  
class ImagePublisher(Node):
  def __init__(self):
    super().__init__('image_publisher')
       
    self.publisher_ = self.create_publisher(Image, 'video_frames', 10)
    timer_period = TIMER_PERIOD
    self.timer = self.create_timer(timer_period, self.timer_callback)
    self.cap = cv2.VideoCapture(0)
    self.br = CvBridge() # Convert between ROS and OpenCV images
    
  def timer_callback(self):
    ret, frame = self.cap.read()
           
    if ret == True:
      self.publisher_.publish(self.br.cv2_to_imgmsg(frame, encoding='rgb8'))
      
    self.get_logger().info('Publishing video frame')
    
class CameraInfoPublisher(Node):
    def __init__(self):
        super().__init__('camera_info_publisher')
        self.publisher_ = self.create_publisher(CameraInfo, 'camera_info', 10)
        timer_period = TIMER_PERIOD
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.msg = CameraInfo()
        self.msg.height = 640
        self.msg.width = 480
        self.msg.distortion_model = "plumb_bob"
        self.msg.d = [0.038385, -0.042656, 0.000869, -0.002758, 0.000000]
        self.msg.k = [445.448888, 0.000000, 317.524152, 0.000000, 444.667306, 241.555012, 0.000000, 0.000000, 1.000000]

    def timer_callback(self):
        self.publisher_.publish(self.msg)
        self.get_logger().info("Publishing camera info")
        self.i += 1

   
def main(args=None):
  rclpy.init(args=args)
  
  camera_info_publisher = CameraInfoPublisher()
  image_publisher = ImagePublisher()
  
# rclpy.spin(camera_info_publisher)
# rclpy.spin(image_publisher)
  
  executor = rclpy.executors.MultiThreadedExecutor()
  executor.add_node(camera_info_publisher)
  executor.add_node(image_publisher)
  executor_thread = threading.Thread(target=executor.spin, daemon=True)
  executor_thread.start()
  try:
      while rclpy.ok():
          pass
  except KeyboardInterrupt:
      pass
    
  rclpy.shutdown()
  executor_thread.join()
  
  # image_publisher.destroy_node()
  # rclpy.shutdown()
   
if __name__ == '__main__':
  main()