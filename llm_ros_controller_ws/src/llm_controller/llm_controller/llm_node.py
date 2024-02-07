import base64
import csv
import datetime
import openai
import os
import sys
import yaml
from math import pi, radians
from openai import OpenAI, ChatCompletion
from swarmnet import SwarmNet, Log_Level, set_log_level
from tenacity import retry, stop_after_attempt, wait_random_exponential
from threading import Lock
from typing import Dict, Optional, List, Tuple

# ROS2 imports
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Local package imports
from .grid import Grid

valid_grid_positions: List[str] = ["A3", "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "C4"]

image_name = "layout.drawio.png"

scan_mutex = Lock()
scan_ranges = []

class ScanSubscriber(Node):
  def __init__(self):
    super().__init__("scan_subscriber")
    
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
    self.subscription
  
  def listener_callback(self, msg: LaserScan) -> None:
    global scan_ranges
    with scan_mutex:
      scan_ranges = msg.ranges

class VelocityPublisher(Node):
  def __init__(self):
    super().__init__("velocity_publisher")
    
    self.declare_parameter('config_file', rclpy.Parameter.Type.STRING) 
    
    if(not self.load_config()):
      self.get_logger().error("!!! FAILED TO LOAD CONFIG")
      self.destroy_node()
      return
    
    self.get_logger().info("Successfully loaded configuration")
    
    self.publisher_ = self.create_publisher(Twist, "/cmd_vel", 10)
    
    self.client: OpenAI = None
    self.this_agents_turn = self.INITIALLY_THIS_AGENTS_TURN
    self.other_agent_ready = False
    self.other_agent_loc = ""
    self.turn_lock = Lock()
    self.ready_lock = Lock()
    self.finish_lock = Lock()
    self.other_agent_finished = False
    self.restart_lock = Lock()
    self.should_restart = False
    self.grid = Grid(self.STARTING_GRID_LOC, self.STARTING_GRID_HEADING, 3, 8)
    self.ready_movement_lock = Lock()
    self.other_agent_ready_movement = False
    
    self.sn_ctrl = SwarmNet({"LLM": self.llm_recv, "READY": self.ready_recv, "FINISHED": self.finished_recv, "INFO": None, "RESTART": self.restart_recv, "MOVE": self.move_recv}, device_list = self.SN_DEVICE_LIST) #! Publish INFO messages which can then be subscribed to by observers
    self.sn_ctrl.start()
    self.get_logger().info(f"SwarmNet initialised") 
    set_log_level(self.SN_LOG_LEVEL)
    self.info("SwarmNet initialised successfully")
    
    # self.global_conv = [
    #     {"role": "system", "content": f"You and I are wheeled robots, and can only move forwards, backwards, and rotate clockwise or anticlockwise.\
    #       We will negotiate with other robots to navigate a path without colliding. You should negotiate and debate the plan until all agents agree.\
    #         You cannot go outside of the grid and we cannot be in the same grid square at once. Only one of us can fit in a square at once.\
    #         Once this has been decided you should call the '\f{self.CMD_SUPERVISOR}' tag at the end of the message but you should agree on the plan first. And print your plan in a concise numbered list using only the following command words:\
    #           - '{self.CMD_FORWARD}' to move one square forwards\
    #           - '{self.CMD_BACKWARDS}' to move one square backwards \
    #           - '{self.CMD_ROTATE_CLOCKWISE}' to rotate 90 degrees clockwise (and stay in the same square) \
    #           - '{self.CMD_ROTATE_ANTICLOCKWISE}' to rotate 90 degrees clockwise (and stay in the same square) \
    #           - '{self.CMD_WAIT}' to wait for the time taken to move one square \
    #         Do not create new commands and we should both move at the same time.\
    #         We should keep our messages as short as possible.\
    #         We cannot fit past each other in the narrow corridor and we cannot leave the corridor.\
    #         The final plan should be a numbered list only containing these commands and we should try to complete the task as quickly as possible."}]
    
    # self.global_conv = [
    #     {"role": "system", "content": f"You and I are on a grid and can only move forwards, backwards, and rotate clockwise or anticlockwise.\
    #       We will negotiate with each other to navigate a path without colliding. You should negotiate and debate the plan until all agents agree.\
    #         We cannot go outside of the grid and we cannot be in the same grid square at once. Only one of us can fit in a square at once.\
    #         Once this has been decided you should call the '\f{self.CMD_SUPERVISOR}' tag at the end of the message but you should agree on the plan first. And print your plan in a concise numbered list using only the following command words:\
    #           - '{self.CMD_FORWARD}' to move one square forwards\
    #           - '{self.CMD_BACKWARDS}' to move one square backwards \
    #           - '{self.CMD_ROTATE_CLOCKWISE}' to rotate 90 degrees clockwise (and stay in the same square) \
    #           - '{self.CMD_ROTATE_ANTICLOCKWISE}' to rotate 90 degrees clockwise (and stay in the same square) \
    #           - '{self.CMD_WAIT}' to wait for the time taken to move one square \
    #         Do not create new commands and we should both move at the same time.\
    #         We should keep our messages as short as possible.\
    #         We cannot fit past each other in the narrow corridor and we cannot leave the corridor. We cannot move past\
    #         The final plan should be a numbered list only containing these commands and we should try to complete the task as quickly as possible."}]
    
    # self.global_conv = [
    #     {"role": "system", "content": f"You and I are on a grid and can only move forwards, backwards, and rotate clockwise or anticlockwise.\
    #       We will negotiate with each other to navigate a path without colliding. You should negotiate and debate the plan until all agents agree.\
    #         We cannot go outside of the grid and we cannot be in the same grid square at once. Only one of us can fit in a square at once.\
    #         Once this has been decided you should call the '\f{self.CMD_SUPERVISOR}' tag at the end of the message but you should agree on the plan first. And print your plan in a concise numbered list using only the following command words:\
    #           - '{self.CMD_FORWARD}' to move one square forwards\
    #           - '{self.CMD_BACKWARDS}' to move one square backwards \
    #           - '{self.CMD_ROTATE_CLOCKWISE}' to rotate 90 degrees clockwise (and stay in the same square) \
    #           - '{self.CMD_ROTATE_ANTICLOCKWISE}' to rotate 90 degrees clockwise (and stay in the same square) \
    #           - '{self.CMD_WAIT}' to wait for the time taken to move one square \
    #         Do not create new commands and we should both move at the same time.\
    #         We should keep our messages as short as possible.\
    #         We cannot fit past each other in the narrow corridor and we cannot leave the corridor. Areas A3 and C4 allow one of us to move out of the way of the other.\
    #         The final plan should be a numbered list only containing these commands and we should try to complete the task as quickly as possible"}]
    
    self.global_conv = [
        {"role": "system", "content": f"You are {self.AGENT_NAME} and you are an agent on a grid and can only move forwards, backwards, and rotate clockwise or anticlockwise.\
          You will negotiate with another agent on the grid to navigate a path without colliding. You should negotiate and debate the plan until all agents agree.\
            You cannot go outside of the grid and you cannot be in the same grid square as another other agent at the same time. Only one agent can fit in a square at once.\
            Once this has been decided you should call the '\f{self.CMD_SUPERVISOR}' tag at the end of the message but you should agree on the plan first. And print your plan in a concise numbered list using only the following command words:\
              - '{self.CMD_FORWARD}' to move one square forwards\
              - '{self.CMD_BACKWARDS}' to move one square backwards \
              - '{self.CMD_ROTATE_CLOCKWISE}' to rotate 90 degrees clockwise (and stay in the same square) \
              - '{self.CMD_ROTATE_ANTICLOCKWISE}' to rotate 90 degrees clockwise (and stay in the same square) \
              - '{self.CMD_WAIT}' to wait for the time taken to move one square \
            Do not create new commands and all agents start movement at the same time.\
            The space is discrete and so you can think of it in terms of the grid.\
            You should try to keep your messages as short as possible.\
            You cannot fit past other agents in the narrow corridor and you cannot leave the corridor. Areas A3 and C4 allow one agent to move out of the way of another, but only have space for one agent.\
            The final plan should be a numbered list only containing these commands and we should try to complete the task as quickly as possible. Make the plan specific for this discrete environment."}]
    
    
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
  
    while(True):
      with self.restart_lock:
        b = self.should_restart
        
      self.create_plan(b)
      self.parse_plan()
      
      with self.restart_lock:
        b = self.should_restart
        self.should_restart = False
        
      if(b):
        continue
      elif(f"{self.grid}" != self.ENDING_GRID_LOC):
        self.sn_ctrl.send("RESTART")
        self.restart(True)
      else:
        self.info("Task completed :)")
        self.destroy_node()
    
  def parse_plan(self):
    if(len(self.global_conv) > 1):
      
      self.info("Waiting for an agent to be ready for movement")
      while(not self.is_ready_movement()):
        self.sn_ctrl.send(f"MOVE")
        self.wait_delay()
        
      self.sn_ctrl.send(f"MOVE")
        
      self.sn_ctrl.clear_rx_queue()
      self.info("Agents ready for movement")
      
      cmd = self.global_conv[len(self.global_conv)-1]["content"]
      for s in cmd.split("\n"):
        if(self.should_restart):
          break
        
        min_dist_reached = False
        
        with scan_mutex:
          rs = scan_ranges
        
        self.info(f"RANGES: {rs}")
        min_dist_reached = any(map(lambda r: r <= self.LIDAR_THRESHOLD, rs))
        self.info(f"{len(rs)} ranges in topic")
        if(min_dist_reached):
          self.info("Min LIDAR reading")
          self.sn_ctrl.send("RESTART")
          break
        elif(self.CMD_FORWARD in s):
          new_pos = self.grid.sim_forwards()
          if(new_pos not in valid_grid_positions):
            self.info(f"Invalid move in plan, {new_pos} is not a valid position")
            break
          self.pub_forwards()
        elif(self.CMD_BACKWARDS in s):
          new_pos = self.grid.sim_backwards()
          if(new_pos not in valid_grid_positions):
            self.info(f"Invalid move in plan, {new_pos} is not a valid position")
            break
          self.pub_backwards()
        elif(self.CMD_ROTATE_CLOCKWISE in s):
          self.pub_clockwise()
        elif(self.CMD_ROTATE_ANTICLOCKWISE in s):
          self.pub_anticlockwise()
        elif(self.CMD_WAIT in s):
          self.info("Wait command")
          self.wait_delay()
        elif(self.CMD_SUPERVISOR in s):
          continue
        elif(s.strip() == ""):
          pass
          continue
        else:
          self.get_logger().error(f"Unrecognised command: {s}")
          continue
        self.wait_delay()
      
    
  # TODO Replan and restart from current position 
  def restart(self, this_agent_stuck: bool):
    self.info("Replanning")
    
    with self.ready_lock:
      self.other_agent_ready = False
      
    with self.turn_lock:
      self.this_agents_turn = this_agent_stuck
    
    if(this_agent_stuck):
      self.global_conv.append({"role": "user", "content": f"I am stuck, we need to replan."})
    
    with self.restart_lock:
      self.should_restart = True
    
  def info(self, s: str) -> None:
    self.get_logger().info(s)
    self.sn_ctrl.send(f"INFO {self.AGENT_NAME}: {s}")
        
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
    
    msg.linear.x = float(dir * self.LINEAR_SPEED) #? X, Y or Z?
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
    msg.angular.z = float(dir * self.ANGULAR_SPEED) #? X Y or Z
    
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
    
  def create_plan(self, is_restart: bool):
    self.info("Waiting for an agent to be ready")
    while(not self.is_ready()):
      self.sn_ctrl.send(f"READY {self.grid} {self.grid._print_heading()}")
      self.wait_delay()
      
    self.sn_ctrl.send(f"READY {self.grid} {self.grid._print_heading()}")
      
    self.sn_ctrl.clear_rx_queue()
    self.info("Agents ready for negotiation")
        
    self.client = OpenAI() # Use the OPENAI_API_KEY environment variable
      
    self.negotiate(is_restart)
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
  
  @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6)) # Exponential backoff
  def _llm_req(self) -> ChatCompletion:
    return self.client.chat.completions.create(
      model=self.MODEL_NAME,
      messages=self.global_conv,
      max_tokens=self.MAX_TOKENS
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
    self.info(f"Negotiations completed")
    # for m in self.global_conv:
    #   self.info(f"{m['role']}: {m['content']}")
    self._log_negotiations(n_stages)
    self.sn_ctrl.send("FINISHED")
    
    self.get_logger().info("Waiting for message queues to clear")
    while(not (self.sn_ctrl.rx_queue.empty() and self.sn_ctrl.tx_queue.empty())):
      self.wait_delay()
    
    self.generate_summary()
    
  def generate_summary(self):
    self.global_conv.append({"role": "user", "content": f"Generate a summarised numerical list of the plan for the steps that I should complete. Use only the commands:\
      - '{self.CMD_FORWARD}' to move one square forwards \
      - '{self.CMD_BACKWARDS}' to move one square backwards \
      - '{self.CMD_ROTATE_CLOCKWISE}' to rotate 90 degrees clockwise (and stay in the same square) \
      - '{self.CMD_ROTATE_ANTICLOCKWISE}' to rotate 90 degrees clockwise (and stay in the same square) \
      - '{self.CMD_WAIT}' to wait for the time taken to move one square\
      You must list every individual step and only the steps agreed for me to execute."})
    
    completion = self._llm_req()

    self.global_conv.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
    self.info(f"Final plan for {self.sn_ctrl.addr}\n{completion.choices[0].message.content}")
  
  def _log_negotiations(self, n_stages: int):
    path = self.WORKING_DIR + f"logs/{self.AGENT_NAME}_negotiation_log.csv"

    with open(path, "a", newline="") as csvfile:
      writer = csv.writer(csvfile)
      today = datetime.date.today().strftime("%Y-%m-%d")
      writer.writerow([today, self.MODEL_NAME, self.MAX_TOKENS, str(self.MAX_NUM_NEGOTIATION_MESSAGES), str(n_stages)])

  def restart_recv(self, msg: Optional[str]) -> None:
    self.restart(False)
  
  def finished_recv(self, msg: Optional[str]) -> None:
    with self.finish_lock:
      self.other_agent_finished = True
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
    self.other_agent_loc = msg.split(" ")[0]
    self.other_agent_heading = msg.split(" ")[1]
    self.ready_lock.release()
    
  def move_recv(self, msg: Optional[str]) -> None:
    with self.ready_movement_lock:
      self.other_agent_ready_movement = True
  
  def is_ready(self):
    self.ready_lock.acquire()
    b = self.other_agent_ready
    self.ready_lock.release()
    return b
  
  def is_ready_movement(self):
    self.ready_movement_lock.acquire()
    b = self.other_agent_ready_movement
    self.ready_movement_lock.release()
    return b
  
  def _supervisor_called(self, s: str, is_restart: bool) -> bool:
    if(is_restart):
      return False
    
    lines = s.split("\n")
    status = False
    
    if(len(lines) == 1 and lines[0] != self.CMD_SUPERVISOR):
      status = False
    elif(self.CMD_SUPERVISOR in lines[-1:][0]):
      status = True
    else:
      status = any(map(lambda l: l.strip() == self.CMD_SUPERVISOR, lines[:-1]))

    if(status):
      self.info("Supervisor called")
    
    return status

  def negotiate(self, is_restart: bool):
    current_stage = 0
    
    s = f"I am at {self.grid}, you are at {self.other_agent_loc}. I must end at {self.ENDING_GRID_LOC} and you must end at {self.STARTING_GRID_LOC}. I am facing {self.grid._print_heading()} and you are facing {self.other_agent_heading}. One of us should move into either grid square A3 or C4 to allow the other to pass."
    
    if self.this_agents_turn:
      self.global_conv.append({"role": "user", "content": s})
    else:
      self.global_conv.append({"role": "assistant", "content": s}) #TODO This may not work
      current_stage = 1
    
    finished = False
    while(current_stage < self.MAX_NUM_NEGOTIATION_MESSAGES and not finished and not self.other_agent_finished):
      while(not self.is_my_turn()): # Wait to receive from the other agent
        if(len(self.global_conv) > 0):
          if(isinstance(self.global_conv[len(self.global_conv)-1]["content"], list)):
            continue
          
          finished = self._supervisor_called(self.global_conv[-1:][0]["content"].strip(), (current_stage == 0) and is_restart)
          
          if(current_stage >= self.MAX_NUM_NEGOTIATION_MESSAGES or finished or self.other_agent_finished):
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
      self.info(f"Negotiation stage {current_stage}")
      self.get_logger().info(f"{self.global_conv}")
      self.wait_delay()
        
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
    hd: Grid.Heading = Grid.Heading.NORTH
    
    match s.upper():
      case "NORTH":
        pass
      case "SOUTH":
        hd = Grid.Heading.SOUTH
      case "WEST":
        hd = Grid.Heading.WEST
      case "EAST":
        hd = Grid.Heading.EAST
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
      self.MAX_TOKENS = data["agent"]["max_tokens"]
      self.CMD_FORWARD: str = data["commands"]["forwards"]
      self.CMD_BACKWARDS: str = data["commands"]["backwards"]
      self.CMD_ROTATE_CLOCKWISE: str = data["commands"]["rotate_clockwise"]
      self.CMD_ROTATE_ANTICLOCKWISE: str = data["commands"]["rotate_anticlockwise"]
      self.CMD_WAIT: str = data["commands"]["wait"]
      self.CMD_SUPERVISOR: str = data["commands"]["supervisor"]
      self.LINEAR_SPEED: float = data["movement"]["linear"]["speed"]
      self.LINEAR_DISTANCE: float = data["movement"]["linear"]["distance"]
      self.LINEAR_TIME: float = self.LINEAR_DISTANCE / self.LINEAR_SPEED
      self.ANGULAR_SPEED: float = radians(float(data["movement"]["angular"]["speed"]))
      self.ANGULAR_DISTANCE: float = radians(float(data["movement"]["angular"]["distance"]))
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
  scan_subscriber = ScanSubscriber()
  executor = MultiThreadedExecutor()
  
  executor.add_node(velocity_publisher)
  executor.add_node(scan_subscriber)
  executor.spin()
  
  # rclpy.spin_once(velocity_publisher) #* spin_once will parse the given plan then return
  # rclpy.spin(velocity_publisher)
  velocity_publisher.sn_ctrl.kill()
  velocity_publisher.destroy_node()
  scan_subscriber.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
    main()
