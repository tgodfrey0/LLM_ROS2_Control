from llm_controller.SwarmNet.swarmnet import SwarmNet
from openai import OpenAI
from math import pi
from threading import Lock
from time import sleep

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

CMD_FORWARD = "@FORWARD"
CMD_ROTATE_CLOCKWISE = "@CLOCKWISE"
CMD_ROTATE_ANTICLOCKWISE = "@ANTICLOCKWISE"
CMD_SUPERVISOR = "@SUPERVISOR"

LINEAR_SPEED = 0.15 # m/s
LINEAR_DISTANCE = 0.45 # m
LINEAR_TIME = LINEAR_DISTANCE / LINEAR_SPEED

ANGULAR_SPEED = 0.3 # rad/s
ANGULAR_DISTANCE = pi/2.0 # rad
ANGULAR_TIME = ANGULAR_DISTANCE / ANGULAR_SPEED

global_conv = []
client: OpenAI = None
max_stages = 10
this_agents_turn = True
tl = Lock()
vp = None

class VelocityPublisher(Node):
  def __init__(self):
    super().__init__("velocity_publisher")
    self.publisher_ = self.create_publisher(Twist, "/cmd_vel", 10)
    
    self.linear_rate = self.create_rate(LINEAR_SPEED / LINEAR_DISTANCE, self.get_clock()) # Hz so inverse the fraction
    self.angular_rate = self.create_rate(ANGULAR_SPEED / ANGULAR_DISTANCE, self.get_clock())
    
    cmd = global_conv[len(global_conv)-1]["content"]
    for s in cmd.split("\n"):
      if(CMD_FORWARD in s):
        self.pub_forward()
      elif(CMD_ROTATE_CLOCKWISE in s):
        self.pub_clockwise()
      elif(CMD_ROTATE_ANTICLOCKWISE in s):
        self.pub_anticlockwise()
      elif(CMD_SUPERVISOR in s):
        pass
      else:
        self.get_logger().error("Unrecognised command")
    
    # ss = ["@FORWARD", "@CLOCKWISE", "@FORWARD", "@CLOCKWISE"]
    # for s in ss:
    #   if(CMD_FORWARD in s):
    #     self.pub_forward()
    #   elif(CMD_ROTATE_CLOCKWISE in s):
    #     self.pub_clockwise()
    #   elif(CMD_ROTATE_ANTICLOCKWISE in s):
    #     self.pub_anticlockwise()
    #   elif(CMD_SUPERVISOR in s):
    #     pass
    #   else:
    #     self.get_logger().error("Unrecognised command")
    
    self.get_logger().info("Full plan parsed")
        
  def _delay(self, t_target):
    t0 = self.get_clock().now()
    while(self.get_clock().now() - t0 < rclpy.duration.Duration(seconds=t_target)):
      pass
    self.get_logger().info(f"Delayed for {t_target} seconds")
    
  def linear_delay(self):
    self._delay(LINEAR_TIME)
    
  def angular_delay(self):
    self._delay(ANGULAR_TIME)
    
  def _publish_cmd(self, msg: Twist):
    self.publisher_.publish(msg)
    self.get_logger().info("Publishing to /cmd_vel")
  
  def _publish_zero(self):
    self.get_logger().info("Zero velocity requested")
    msg = Twist()
    
    msg.linear.x = 0.0
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 0.0
    
    self._publish_cmd(msg)
    
  def pub_forward(self):
    self.get_logger().info("Forward command")
    msg = Twist()
    
    msg.linear.x = LINEAR_SPEED #? X, Y or Z?
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 0.0
    
    self._publish_cmd(msg)
    
    self.linear_delay()
        
    self._publish_zero()
    
  def _pub_rotation(self, dir: float):
    msg = Twist()
    
    msg.linear.x = 0.0
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = dir * ANGULAR_SPEED #? X Y or Z
    
    self._publish_cmd(msg)
    
    self.angular_delay()
    
    self._publish_zero()
    
  def pub_anticlockwise(self):
    self.get_logger().info("Anticlockwise command")
    self._pub_rotation(-1)
    
  def pub_clockwise(self):
    self.get_logger().info("Clockwise command")
    self._pub_rotation(1)
    

def is_my_turn():
  tl.acquire()
  b = this_agents_turn
  tl.release()
  return b

def toggle_turn():
  global this_agents_turn
  tl.acquire()
  this_agents_turn = not this_agents_turn
  tl.release()
  
def send_req():
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=global_conv,
    # max_tokens=500
  )

  # print(completion.choices[0].message)
  global_conv.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
  sn_ctrl.send(f"LLM {completion.choices[0].message.role} {completion.choices[0].message.content}")
  
def get_api_key() -> str:
  with open("openai_key", "r") as f:
    return f.readline().rstrip()
  
def toggle_role(r: str):
  if r == "assistant":
    return "user"
  elif r == "user":
    return "assistant"
  else:
    return ""
  
def plan_completed():
  print("Plan completed:")
  for m in global_conv:
    print(f"{m['role']}: {m['content']}")
  
def llm_recv(msg: str) -> None: 
  m = msg.split(" ", 1) # Msg are LLM ROLE CONTENT
  r = m[0]
  c = m[1]
  global_conv.append({"role": toggle_role(r), "content": c}) #! Don't think this is adding to the list
  toggle_turn()
  # if("@SUPERVISOR" not in c):
  #   send_req(client)
  # else:
  #   plan_completed() #? This may have issues with only one agent finishing. Could just add a SN command

def negotiate():
  current_stage = 0
  
  if this_agents_turn:
    global_conv.append({"role": "user", "content": "I am at D1, you are at D7. I must end at D7 and you must end at D1"})
  
  while(current_stage < max_stages or not global_conv[len(global_conv)-1]["content"].endswith("@SUPERVISOR")):
    while(not is_my_turn()): # Wait to receive from the other agent
      sleep(0.5)
      print("waiting")
    
    send_req()
    toggle_turn()
    current_stage += 1
    print(f"Stage {current_stage}")
    print(global_conv);
      
  plan_completed()
  current_stage = 0

def main(args=None):
  global global_conv
  global client
  global logger
  
  rclpy.init()
  velocity_publisher = VelocityPublisher()
  logger = velocity_publisher.get_logger()
  
  sn_ctrl = SwarmNet({"LLM": llm_recv})
  sn_ctrl.start()
  print("Communications initialised")
  input("Press any key to start")
  client = OpenAI(api_key=get_api_key())
  global_conv = [
    {"role": "system", "content": f"You and I are wheeled robots, and can only move forwards, backwards, and rotate clockwise or anticlockwise.\
      We will negotiate with other robots to navigate a path without colliding. You should negotiate and debate the plan until all agents agree.\
        Once this has been decided you should call the 'f{CMD_SUPERVISOR}' tag at the end of your plan and print your plan in a concise numbered list using only the following command words:\
          - '{CMD_FORWARD}' to move one square forwards\
          - '{CMD_ROTATE_CLOCKWISE}' to rotate 90 degrees clockwise\
          - '{CMD_ROTATE_ANTICLOCKWISE}' to rotate 90 degrees clockwise\
          "}]
  negotiate()
  input("Press any key to finish")
  sn_ctrl.kill()
  
  # global global_conv
  
  # global_conv = [
  #   {"role": "system", "content": f"@FORWARD"}]
  rclpy.spin_once(velocity_publisher) #* spin_once will parse the given plan then return
  velocity_publisher.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
    main()
