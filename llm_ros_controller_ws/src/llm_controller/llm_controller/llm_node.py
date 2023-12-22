from swarmnet import SwarmNet
from openai import OpenAI
from math import pi
from threading import Lock
from typing import Optional, List, Tuple
import os

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

#! Will need some way of determining which command in the plan is for which agent
#! Use some ID prefixed to the command?

dl: List[Tuple[str, int]] = [("192.168.0.120", 51000)] # Other device
# dl: List[Tuple[str, int]] = [("192.168.0.121", 51000)] # Other device

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

WAITING_TIME = 0.5

class VelocityPublisher(Node):
  def __init__(self):
    super().__init__("velocity_publisher")
    self.publisher_ = self.create_publisher(Twist, "/cmd_vel", 10)
    self.global_conv = []
    self.client: OpenAI = None
    self.max_stages = 10
    self.this_agents_turn = True
    self.other_agent_ready = False
    self.turn_lock = Lock()
    self.ready_lock = Lock()
  
    self.create_plan()
    
    if(len(self.global_conv) > 1):
      cmd = self.global_conv[len(self.global_conv)-1]["content"]
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
          self.get_logger().error(f"Unrecognised command: {s}")
    
    # ss = ["@FORWARD", "@ANTICLOCKWISE", "@FORWARD", "@CLOCKWISE", "@FORWARD"]
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
    
    self.get_logger().info(f"Full plan parsed")
        
  def _delay(self, t_target):
    t0 = self.get_clock().now()
    while(self.get_clock().now() - t0 < rclpy.duration.Duration(seconds=t_target)):
      pass
    self.get_logger().info(f"Delayed for {t_target} seconds")
    
  def linear_delay(self):
    self._delay(LINEAR_TIME)
    
  def angular_delay(self):
    self._delay(ANGULAR_TIME)
    
  def wait_delay(self):
    self._delay(WAITING_TIME)
    
  def _publish_cmd(self, msg: Twist):
    self.publisher_.publish(msg)
    self.get_logger().info(f"Publishing to /cmd_vel")
  
  def _publish_zero(self):
    self.get_logger().info(f"Zero velocity requested")
    msg = Twist()
    
    msg.linear.x = 0.0
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 0.0
    
    self._publish_cmd(msg)
    
  def pub_forward(self):
    self.get_logger().info(f"Forward command")
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
    self.get_logger().info(f"Anticlockwise command")
    self._pub_rotation(1)
    
  def pub_clockwise(self):
    self.get_logger().info(f"Clockwise command")
    self._pub_rotation(-1)
    
  def create_plan(self):
    self.get_logger().info(f"Initialising SwarmNet")
    self.sn_ctrl = SwarmNet({"LLM": self.llm_recv, "READY": self.ready_recv}, device_list = dl)
    # self.sn_ctrl.set_logger_fn(self.get_logger().info)
    self.sn_ctrl.start()
    self.get_logger().info(f"SwarmNet initialised") 
    
    #* Wait until another agent connects
    # while(len(self.sn_ctrl.get_devices()) == 0): #? Does does an agent add itself to the list? I don't think so 
    #   self.wait_delay()
    #   self.get_logger().warn("Waiting for an agent to connect")
    while(not self.is_ready()):
      self.sn_ctrl.send("READY")
      self.get_logger().warn("Waiting for an agent to be ready")
      self.wait_delay()
    
    self.client = OpenAI() # Use the OPENAI_API_KEY environment variable
    self.global_conv = [
      {"role": "system", "content": f"You and I are wheeled robots, and can only move forwards, backwards, and rotate clockwise or anticlockwise.\
        We will negotiate with other robots to navigate a path without colliding. You should negotiate and debate the plan until all agents agree.\
          Once this has been decided you should call the '\f{CMD_SUPERVISOR}' tag at the end of your plan and print your plan in a concise numbered list using only the following command words:\
            - '{CMD_FORWARD}' to move one square forwards\
            - '{CMD_ROTATE_CLOCKWISE}' to rotate 90 degrees clockwise\
            - '{CMD_ROTATE_ANTICLOCKWISE}' to rotate 90 degrees clockwise\
            The final plan should be a numbered list only containing these commands."}]
    self.negotiate()
    self.sn_ctrl.kill()
    
  def is_my_turn(self):
    self.turn_lock.acquire()
    b = self.this_agents_turn
    self.turn_lock.release()
    return b

  def toggle_turn(self):
    self.turn_lock.acquire()
    self.this_agents_turn = not self.this_agents_turn
    self.turn_lock.release()
    
  def send_req(self):
    completion = self.client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=self.global_conv,
      # max_tokens=500
    )

    # print(completion.choices[0].message)
    self.global_conv.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
    self.sn_ctrl.send(f"LLM {completion.choices[0].message.role} {completion.choices[0].message.content}")
    
  def toggle_role(self, r: str):
    if r == "assistant":
      return "user"
    elif r == "user":
      return "assistant"
    else:
      return ""
    
  def plan_completed(self):
    self.get_logger().info(f"Plan completed:")
    for m in self.global_conv:
      self.get_logger().info(f"{m['role']}: {m['content']}")
    
  def llm_recv(self, msg: Optional[str]) -> None: 
    m = msg.split(" ", 1) # Msg are LLM ROLE CONTENT
    r = m[0]
    c = m[1]
    self.global_conv.append({"role": self.toggle_role(r), "content": c}) #! Don't think this is adding to the list
    self.toggle_turn()

  def ready_recv(self, msg: Optional[str]) -> None:
    self.ready_lock.acquire()
    self.other_agent_ready = True
    self.ready_lock.release()
  
  def is_ready(self):
    self.ready_lock.acquire()
    b = self.other_agent_ready
    self.ready_lock.release()
    return b

  def negotiate(self):
    current_stage = 0
    
    if self.this_agents_turn:
      self.global_conv.append({"role": "user", "content": "I am at D1, you are at D7. I must end at D7 and you must end at D1"})
    
    while(current_stage < self.max_stages or not self.global_conv[len(self.global_conv)-1]["content"].endswith("@SUPERVISOR")):
      while(not self.is_my_turn()): # Wait to receive from the other agent
        #! sleep(0.5)
        self.wait_delay()
        #? Will this allow another thread to get the mutex?
        self.get_logger().info(f"Waiting for a response from another agent")
      
      self.send_req()
      self.toggle_turn()
      current_stage += 1
      self.get_logger().info(f"Stage {current_stage}")
      self.get_logger().info(f"{self.global_conv}");
        
    self.plan_completed()
    current_stage = 0
  

def main(args=None):
  
  rclpy.init()
  velocity_publisher = VelocityPublisher()
  
  #* Move this logic into the node itself
  
  # global global_conv
  
  # global_conv = [
  #   {"role": "system", "content": f"@FORWARD"}]
  rclpy.spin_once(velocity_publisher) #* spin_once will parse the given plan then return
  velocity_publisher.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
    main()
