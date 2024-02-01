import base64
import csv
import datetime
import openai
import os
import sys
import yaml
from math import pi
from openai import OpenAI, ChatCompletion
from swarmnet import SwarmNet, Log_Level, set_log_level
from threading import Lock
from typing import Dict, Optional, List, Tuple

# ROS2 imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Local package imports
from .grid import Grid

#TODO Check moves in grid before IRL then replan if not working
#TODO Ensure no more messages are sent if the negotiation exit clause is hit

image_name = "layout.drawio.png"

class VelocityPublisher(Node):
  def __init__(self):
    super().__init__("velocity_publisher")
    
    self.declare_parameter('config_file', rclpy.Parameter.Type.STRING) 
    
    if(not self.load_config()):
      self.get_logger().error("!!! FAILED TO LOAD CONFIG")
      return
    
    self.get_logger().info("Successfully loaded configuration")
    
    self.publisher_ = self.create_publisher(Twist, "/cmd_vel", 10)
    qos = QoSProfile(
      reliability=ReliabilityPolicy.BEST_EFFORT,
      history=HistoryPolicy.KEEP_LAST,
      depth=10
    )
    
    self.subscription = self.create_subscription(
      LaserScan,
      "/scan",
      self.listener_callback,
      qos_profile=qos
    )
    
    self.global_conv = [
        {"role": "system", "content": f"You and I are wheeled robots, and can only move forwards, backwards, and rotate clockwise or anticlockwise.\
          We will negotiate with other robots to navigate a path without colliding. You should negotiate and debate the plan until all agents agree.\
            You cannot go outside of the grid and we cannot be in the same grid square at once. Only one of us can fit in a square at once.\
            Once this has been decided you should call the '\f{self.CMD_SUPERVISOR}' tag at the end of your plan and print your plan in a concise numbered list using only the following command words:\
              - '{self.CMD_FORWARD}' to move one square forwards\
              - '{self.CMD_BACKWARDS}' to move one square backwards \
              - '{self.CMD_ROTATE_CLOCKWISE}' to rotate 90 degrees clockwise (and stay in the same square) \
              - '{self.CMD_ROTATE_ANTICLOCKWISE}' to rotate 90 degrees clockwise (and stay in the same square) \
              The final plan should be a numbered list only containing these commands."}]
    
    if("vision" in self.MODEL_NAME):
      self.info("Vision model provided, appending image")
      image_path = self.WORKING_DIR + image_name
      with open(image_path, "rb") as image:
        image_data = base64.b64encode(image.read()).decode("utf-8")
      
      self.global_conv.append({
        "role": "user",
        "content": [
            {
              "type": "text",
              "text": "This is a diagram of the layout of the area."
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/{image_path.split('.')[-1]};base64,{image_data}"
              }
            }
          ]
      })
      
    self.client: OpenAI = None
    self.this_agents_turn = self.INITIALLY_THIS_AGENTS_TURN
    self.other_agent_ready = False
    self.other_agent_loc = ""
    self.turn_lock = Lock()
    self.ready_lock = Lock()
    self.grid = Grid(self.STARTING_GRID_LOC,self.STARTING_GRID_HEADING, 3, 8)
    self.scan_mutex = Lock()
    self.scan_ranges = False
    
    self.sn_ctrl = SwarmNet({"LLM": self.llm_recv, "READY": self.ready_recv, "FINISHED": self.finished_recv, "INFO": None, "RESTART": self.restart_recv}, device_list = self.SN_DEVICE_LIST) #! Publish INFO messages which can then be subscribed to by observers
    self.sn_ctrl.start()
    self.get_logger().info(f"SwarmNet initialised") 
    set_log_level(self.SN_LOG_LEVEL)
    self.sn_ctrl.send("INFO SwarmNet initialised successfully")
  
    while(True):
      self.create_plan()
      self.parse_plan()
      if(f"{self.grid}" != self.ENDING_GRID_LOC):
        self.restart(True)
    
  def parse_plan(self):
    if(len(self.global_conv) > 1):
      cmd = self.global_conv[len(self.global_conv)-1]["content"]
      for s in cmd.split("\n"):
        min_dist_reached = False
        with self.scan_mutex:
          self.info(f"RANGES: {self.scan_ranges}")
          min_dist_reached = any(map(lambda r: r <= self.LIDAR_THRESHOLD, self.scan_ranges))
          self.info(f"{len(self.scan_ranges)} ranges in topic")
          if(min_dist_reached):
            self.sn_ctrl.send("RESTART")
            break
        if(min_dist_reached):
          self.info("Min LIDAR reading")
          self.pub_backwards()
        elif(self.CMD_FORWARD in s):
          if(self.grid.check_forwards()):
            self.info("Invalid move in plan")
            break
          self.pub_forwards()
        elif(self.CMD_BACKWARDS in s):
          if(self.grid.check_backwards()):
            self.info("Invalid move in plan")
            break
          self.pub_backwards()
        elif(self.CMD_ROTATE_CLOCKWISE in s):
          self.pub_clockwise()
        elif(self.CMD_ROTATE_ANTICLOCKWISE in s):
          self.pub_anticlockwise()
        elif(self.CMD_SUPERVISOR in s):
          pass
        elif(s.strip() == ""):
          pass
        else:
          self.get_logger().error(f"Unrecognised command: {s}")
        self.wait_delay()

      self.info(f"Full plan parsed")
    
  # TODO Replan and restart from current position 
  def restart(self, this_agent_stuck: bool):
    self.info("Replanning")
    
    with self.ready_lock:
      self.other_agent_ready = False
      
    with self.turn_lock:
      self.this_agents_turn = this_agent_stuck
    
    if(this_agent_stuck):
      self.global_conv.append({"role": "user", "content": f"I am stuck, we need to replan."})
    
  def info(self, s: str) -> None:
    self.get_logger().info(s)
    self.sn_ctrl.send(f"INFO {self.AGENT_NAME}: {s}")
    
  def listener_callback(self, msg: LaserScan) -> None:      
    # Clip ranges to those in front
    angle_increment = msg.angle_increment
    angle_min = msg.angle_min
    angle_max = msg.angle_max
    ranges = msg.ranges

    # Calculate the start and end indices for the 90-degree cone
    start_index = int((45 - angle_min) / angle_increment)
    end_index = int((45 - angle_max) / angle_increment)

    # Ensure indices are within bounds
    start_index = max(0, start_index)
    end_index = min(len(ranges) - 1, end_index)
      
    with self.scan_mutex:
      self.scan_ranges = ranges[start_index:end_index + 1]
        
  def _delay(self, t_target):
    t0 = self.get_clock().now()
    while(self.get_clock().now() - t0 < rclpy.duration.Duration(seconds=t_target)):
      pass
    self.get_logger().info(f"Delayed for {t_target} seconds")
    
  def linear_delay(self):
    self._delay(self.LINEAR_TIME)
    
  def angular_delay(self):
    self._delay(self.ANGULAR_TIME)
    
  def wait_delay(self):
    self._delay(self.WAITING_TIME)
    
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
    
  def _pub_linear(self, dir: int):
    msg = Twist()
    
    msg.linear.x = dir * self.LINEAR_SPEED #? X, Y or Z?
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
    msg.angular.z = dir * self.ANGULAR_SPEED #? X Y or Z
    
    self._publish_cmd(msg)
    self.angular_delay()
    self._publish_zero()
    
  def pub_forwards(self):
    self.info(f"Forwards command")
    self.grid.forwards()
    self._pub_linear(1)
    
  def pub_backwards(self):
    self.info(f"Backwards command")
    self.grid.backwards()
    self._pub_linear(-1)
    
  def pub_anticlockwise(self):
    self.info(f"Anticlockwise command")
    self.grid.anticlockwise()
    self._pub_rotation(1)
    
  def pub_clockwise(self):
    self.info(f"Clockwise command")
    self.grid.clockwise()
    self._pub_rotation(-1)
    
  def create_plan(self):
    while(not self.is_ready()):
      self.sn_ctrl.send(f"READY {self.grid}")
      self.info("Waiting for an agent to be ready")
      self.wait_delay()
      
    self.sn_ctrl.send(f"READY {self.grid}")
      
    self.sn_ctrl.clear_rx_queue()
    self.sn_ctrl.send("INFO Agents ready for negotiation")
        
    self.client = OpenAI() # Use the OPENAI_API_KEY environment variable
      
    self.negotiate()
    self.sn_ctrl.send(f"INFO {self.AGENT_NAME}: Negotiation finished")
    
  def is_my_turn(self):
    self.turn_lock.acquire()
    b = self.this_agents_turn
    self.turn_lock.release()
    return b

  def toggle_turn(self):
    self.turn_lock.acquire()
    self.this_agents_turn = not self.this_agents_turn
    self.turn_lock.release()
    
  def set_turn(self, b):
    self.turn_lock.acquire()
    self.this_agents_turn = b
    self.turn_lock.release()
    
  def _llm_req(self) -> ChatCompletion:
    return self.client.chat.completions.create(
      model=self.MODEL_NAME,
      messages=self.global_conv,
      max_tokens=300
    )

  def send_req(self):
    completion = self._llm_req()

    # print(completion.choices[0].message)
    self.global_conv.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
    self.sn_ctrl.send(f"LLM {completion.choices[0].message.role} {completion.choices[0].message.content}")
    self.info(completion.choices[0].message.content)
    
  def toggle_role(self, r: str):
    if r == "assistant":
      return "user"
    elif r == "user":
      return "assistant"
    else:
      return ""
    
  def plan_completed(self, n_stages: int):
    self.info(f"Plan completed:")
    for m in self.global_conv:
      self.info(f"{m['role']}: {m['content']}")
      
    self.sn_ctrl.send("FINISHED")
    
    while(not (self.sn_ctrl.rx_queue.empty() and self.sn_ctrl.tx_queue.empty())):
      self.get_logger().info("Waiting for message queues to clear")
      self.wait_delay()
    
    self.generate_summary()
    
  def generate_summary(self):
    self.global_conv.append({"role": "user", "content": f"Generate a summarised numerical list of the plan for the steps that I should complete. Use only the commands:\
      - '{self.CMD_FORWARD}' to move one square forwards\
      - '{self.CMD_BACKWARDS}' to move one square backwards \
      - '{self.CMD_ROTATE_CLOCKWISE}' to rotate 90 degrees clockwise \
      - '{self.CMD_ROTATE_ANTICLOCKWISE}' to rotate 90 degrees clockwise "})
    
    completion = self._llm_req()

    self.global_conv.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
    self.info(f"Final plan for {self.sn_ctrl.addr}: {completion.choices[0].message.content}")
  
  def _log_negotiations(self, n_stages: int):
    path = self.WORKING_DIR + f"logs/{self.AGENT_NAME}_negotiation_log.csv"

    with open(path, "a", newline="") as csvfile:
      writer = csv.writer(csvfile)
      today = datetime.date.today().strftime("%Y-%m-%d")
      writer.writerow([today, self.model, str(self.MAX_NUM_NEGOTIATION_MESSAGES), str(n_stages)])

  def restart_recv(self, msg: Optional[str]) -> None:
    self.restart(False)
  
  def finished_recv(self, msg: Optional[str]) -> None:
    self.generate_summary()
  
  def llm_recv(self, msg: Optional[str]) -> None: 
    m = msg.split(" ", 1) # Msg are LLM ROLE CONTENT
    r = m[0]
    c = m[1]
    self.global_conv.append({"role": self.toggle_role(r), "content": c})
    self.toggle_turn()

  def ready_recv(self, msg: Optional[str]) -> None:
    self.ready_lock.acquire()
    self.other_agent_ready = True
    self.other_agent_loc = msg
    self.ready_lock.release()
  
  def is_ready(self):
    self.ready_lock.acquire()
    b = self.other_agent_ready
    self.ready_lock.release()
    return b

  def negotiate(self):
    current_stage = 0
    
    if self.this_agents_turn:
      self.global_conv.append({"role": "user", "content": f"I am at {self.grid}, you are at {self.other_agent_loc}. I must end at {self.ENDING_GRID_LOC} and you must end at {self.STARTING_GRID_LOC}. I am facing {self.grid._print_heading()}."})
    else:
      current_stage = 1
    
    while(current_stage < self.MAX_NUM_NEGOTIATION_MESSAGES):
      while(not self.is_my_turn()): # Wait to receive from the other agent
        if(len(self.global_conv) > 0):
          if(isinstance(self.global_conv[len(self.global_conv)-1]["content"], list)):
            continue
          
          if(self.CMD_SUPERVISOR in self.global_conv[-1:][0]["content"].strip().split("\n")[-1:][0]):
            self.info("\n\n\n\n\n\nSUPERVISOR CALLED =====================================================================================================\n\n\n\n\n\n")
            break
        
        self.wait_delay()
        self.get_logger().info(f"Waiting for a response from another agent")
        
      # if(len(self.global_conv) > 0 and self.global_conv[len(self.global_conv)-1]["content"].rstrip().endswith(f"{CMD_SUPERVISOR}")):
      #   self.get_logger().info(f"Content ends with {CMD_SUPERVISOR}")
      #   break;
      
      self.send_req()
      self.toggle_turn()
      current_stage += 2 # Shares the current_stage
      self.get_logger().info(f"Stage {current_stage}")
      self.sn_ctrl.send(f"INFO Negotiation stage {current_stage}")
      self.get_logger().info(f"{self.global_conv}");
        
    self.plan_completed(current_stage)
    current_stage = 0
    
  def _parse_log_level(self, s: str) -> Log_Level:
    ll: Log_Level = Log_Level.INFO
    
    match s.upper():
      case "INFO":
        pass
      case "SUCCESS":
        ll = Log_Level.SUCCESS
      case "WARN":
        ll = Log_Level.WARN
      case "ERROR":
        ll = Log_Level.ERROR
      case "CRITICAL":
        ll = Log_Level.CRITICAL
      case _:
        self.get_logger().error(f"Unrecognised log level: {s}")
        
    return ll
    
  def _parse_heading(self, s: str) -> Grid.Heading:
    hd: Grid.Heading = Grid.Heading.UP
    
    match s.upper():
      case "UP":
        pass
      case "DOWN":
        hd = Grid.Heading.DOWN
      case "LEFT":
        hd = Grid.Heading.LEFT
      case "RIGHT":
        hd = Grid.Heading.RIGHT
      case _:
        self.get_logger().error(f"Unrecognised grid heading: {s}")
        
    return hd
    
  def load_config(self) -> bool:
    
    #TODO Create param and get
    path = self.get_parameter("config_file").get_parameter_value().string_value
    
    status: bool = False
    
    self.get_logger().info(f"Config file: {path}")
    
    try:
      with open(path, "r") as file:
        data = yaml.safe_load(file)
      
      self.SN_DEVICE_LIST: List[Tuple[str, int]] = list(map(lambda d: (str(d["ip"]), int(d["port"])), data["swarmnet"]["devices"]))
      self.SN_LOG_LEVEL = self._parse_log_level(data["swarmnet"]["log_level"])
      self.AGENT_NAME: str = data["agent"]["agent_name"]
      self.INITIALLY_THIS_AGENTS_TURN: bool = bool(data["agent"]["initially_this_agents_turn"])
      self.STARTING_GRID_LOC: str = data["agent"]["starting_grid_loc"]
      self.STARTING_GRID_HEADING: Grid.Heading = self._parse_heading(data["agent"]["starting_grid_heading"])
      self.ENDING_GRID_LOC: str = data["agent"]["ending_grid_loc"]
      self.MAX_NUM_NEGOTIATION_MESSAGES: int = data["agent"]["max_num_negotiation_messages"]
      self.WORKING_DIR: str = data["agent"]["working_dir"]
      self.MODEL_NAME = data["agent"]["model"]
      self.CMD_FORWARD: str = data["commands"]["forwards"]
      self.CMD_BACKWARDS: str = data["commands"]["backwards"]
      self.CMD_ROTATE_CLOCKWISE: str = data["commands"]["rotate_clockwise"]
      self.CMD_ROTATE_ANTICLOCKWISE: str = data["commands"]["rotate_anticlockwise"]
      self.CMD_SUPERVISOR: str = data["commands"]["supervisor"]
      self.LINEAR_SPEED: float = data["movement"]["linear"]["speed"]
      self.LINEAR_DISTANCE: float = data["movement"]["linear"]["distance"]
      self.LINEAR_TIME: float = self.LINEAR_DISTANCE / self.LINEAR_SPEED
      self.ANGULAR_SPEED: float = data["movement"]["angular"]["speed"]
      self.ANGULAR_DISTANCE: float = data["movement"]["angular"]["distance"]
      self.ANGULAR_TIME: float = self.ANGULAR_DISTANCE / self.ANGULAR_SPEED
      self.WAITING_TIME: float = data["movement"]["waiting_time"]
      self.LIDAR_THRESHOLD: float = data["movement"]["lidar_threshold"]
      status = True
    except OSError as exc:
      self.get_logger().error(f"Failed to load config file: {exc}")
    except (yaml.YAMLError, TypeError) as exc:
      self.get_logger().error(f"Error parsing YAML file: {exc}")
    except NameError as exc:
      self.get_logger().error(f"Undefined config parameter: {exc}")
      
    return status

def main(args=None):
  
  rclpy.init()
  velocity_publisher = VelocityPublisher()
  
  rclpy.spin_once(velocity_publisher) #* spin_once will parse the given plan then return
  velocity_publisher.sn_ctrl.kill()
  velocity_publisher.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
    main()
