from swarmnet import SwarmNet
from openai import OpenAI
from math import pi
from threading import Lock
from typing import Optional, List, Tuple
from enum import Enum
from time import sleep
import threading

#! Will need some way of determining which command in the plan is for which agent
#! Use some ID prefixed to the command?

dl: List[Tuple[str, int]] = [("192.168.0.120", 51000)] # Other device
# dl: List[Tuple[str, int]] = [("192.168.0.121", 51000)] # Other device
# dl: List[Tuple[str, int]] = [("192.168.0.64", 51000)] # Other device

CMD_FORWARD = "@FORWARD"
CMD_BACKWARDS = "@BACKWARDS"
CMD_ROTATE_CLOCKWISE = "@CLOCKWISE"
CMD_ROTATE_ANTICLOCKWISE = "@ANTICLOCKWISE"
CMD_SUPERVISOR = "@SUPERVISOR"

LINEAR_SPEED = 0.15 # m/s
LINEAR_DISTANCE = 0.45 # m
LINEAR_TIME = LINEAR_DISTANCE / LINEAR_SPEED

ANGULAR_SPEED = 0.3 # rad/s
ANGULAR_DISTANCE = pi/2.0 # rad
ANGULAR_TIME = ANGULAR_DISTANCE / ANGULAR_SPEED

WAITING_TIME = 1

INITIALLY_THIS_AGENTS_TURN = True
STARTING_GRID_LOC = "D1"
ENDING_GRID_LOC = "D7"

class Grid():
  class Heading(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
  
  def __init__(self, loc: str, heading: Heading, width: int, height: int):
    self.col = loc[0].upper()
    self.row = int(loc[1])
    self.max_height = height
    self.max_width = width
    self.heading = heading
    
  def __repr__(self) -> str:
    return f"{self.col}{self.row}"
  
  def _check_bound_min_row(self) -> bool:
    b = self.row < 0
    
    if(b):
      print("Row clipped at lower bound")
    
    return b
  
  def _check_bound_max_row(self) -> bool:
    b = self.row >= self.max_height
    
    if(b):
      print("Row clipped at upper bound")
    
    return b
  
  def _check_bound_min_col(self) -> bool:
    b = (ord(self.col)-ord('A')) < 0
    
    if(b):
      print("Column clipped at lower bound")
    
    return b
  
  def _check_bound_max_col(self) -> bool:
    b = (ord(self.col)-ord('A')) >= self.max_width
    
    if(b):
      print("Column clipped at upper bound")
    
    return b
  
  def _bound_loc(self):
    self.row = 0 if self._check_bound_min_row() else self.row
    self.row = (self.max_height-1) if self._check_bound_max_row() else self.row
    self.col = 'A' if self._check_bound_min_col() else self.col
    self.col = chr((self.max_width-1) + ord('A')) if self._check_bound_max_col() else self.col
    
  def _finish_move(self):
    self._bound_loc()
    print(f"Current grid location: {self}")
    print(f"Current heading: {self.heading.name}")
  
  def forwards(self):
    match self.heading:
      case Grid.Heading.UP:
        self.row += 1
      case Grid.Heading.DOWN:
        self.row -= 1
      case Grid.Heading.LEFT:
        self.col = chr(ord(self.col)-1)
      case Grid.Heading.RIGHT:
        self.col = chr(ord(self.col)+1)
    
    self._finish_move()
  
  def backwards(self):
    match self.heading:
      case Grid.Heading.UP:
        self.row -= 1
      case Grid.Heading.DOWN:
        self.row += 1
      case Grid.Heading.LEFT:
        self.col = chr(ord(self.col)+1)
      case Grid.Heading.RIGHT:
        self.col = chr(ord(self.col)-1)

    self._finish_move()
  
  def clockwise(self):
    self.heading = Grid.Heading((self.heading.value + 1) % 4)
      
    self._finish_move()
  
  def anticlockwise(self):
    self.heading = Grid.Heading((self.heading.value - 1) % 4)
      
    self._finish_move()
class LLM():
  def __init__(self):
    self.global_conv = []
    self.client: OpenAI = None
    self.max_stages = 5
    self.this_agents_turn = INITIALLY_THIS_AGENTS_TURN
    self.other_agent_ready = False
    self.turn_lock = Lock()
    self.ready_lock = Lock()
    self.grid = Grid(STARTING_GRID_LOC, Grid.Heading.UP, 8, 8) #! When moving into ROS update grid position
  
    self.create_plan()
    
    if(len(self.global_conv) > 1):
      cmd = self.global_conv[len(self.global_conv)-1]["content"]
      for s in cmd.split("\n"):
        if(CMD_FORWARD in s):
          self.pub_forwards()
        elif(CMD_BACKWARDS in s):
          self.pub_backwards()
        elif(CMD_ROTATE_CLOCKWISE in s):
          self.pub_clockwise()
        elif(CMD_ROTATE_ANTICLOCKWISE in s):
          self.pub_anticlockwise()
        elif(CMD_SUPERVISOR in s):
          pass
        elif(s.strip() == ""):
          pass
        else:
          print(f"Unrecognised command: {s}")
        
  def _delay(self, t_target):
    sleep(t_target)
    print(f"Delayed for {t_target} seconds")
    
  def linear_delay(self):
    self._delay(LINEAR_TIME)
    
  def angular_delay(self):
    self._delay(ANGULAR_TIME)
    
  def wait_delay(self):
    self._delay(WAITING_TIME)
    
  
  def _publish_zero(self):
    print("ZERO")
    
  def _pub_linear(self, dir: int):    
    self.linear_delay()
    self._publish_zero()
    
  def pub_forwards(self):
    print(f"Forwards command")
    self.grid.forwards()
    self._pub_linear(1)
    
  def pub_backwards(self):
    print(f"Backwards command")
    self.grid.backwards()
    self._pub_linear(-1)
    
  def _pub_rotation(self, dir: int):
    self.angular_delay()
    self._publish_zero()
    
  def pub_anticlockwise(self):
    print(f"Anticlockwise command")
    self.grid.anticlockwise()
    self._pub_rotation(1)
    
  def pub_clockwise(self):
    print(f"Clockwise command")
    self.grid.clockwise()
    self._pub_rotation(-1)
    
  def t(self, sn_ctrl: SwarmNet):
    while(not self.is_ready()):
      print(f"Ready: {self.is_ready()}")
      print(f"Turn: {self.is_my_turn()}")
      
      with sn_ctrl.rx_queue.mutex:
        print(f"Queue: {sn_ctrl.rx_queue.queue}")
        
      self._delay(0.1)
    
  def create_plan(self):
    print(f"Initialising SwarmNet")
    self.sn_ctrl = SwarmNet({"LLM": self.llm_recv, "READY": self.ready_recv, "FINISHED": self.finished_recv}, device_list = dl) #! Publish INFO messages which can then be subscribed to by observers
    self.sn_ctrl.start()
    print(f"SwarmNet initialised") 
    
    t1 = threading.Thread(target=self.t, args=[self.sn_ctrl])
  
    while(not self.is_ready()):
      self.sn_ctrl.send("READY")
      print("Waiting for an agent to be ready")
      self.wait_delay()
      
    self.sn_ctrl.send("READY")
      
    self.sn_ctrl.clear_rx_queue()
    
    #! Agent started second cannot proceed past READY synchronisation
    
    self.client = OpenAI() # Use the OPENAI_API_KEY environment variable
    self.global_conv = [
      {"role": "system", "content": f"You and I are wheeled robots, and can only move forwards, backwards, and rotate clockwise or anticlockwise.\
        We will negotiate with other robots to navigate a path without colliding. You should negotiate and debate the plan until all agents agree.\
          Once this has been decided you should call the '\f{CMD_SUPERVISOR}' tag at the end of your plan and print your plan in a concise numbered list using only the following command words:\
            - '{CMD_FORWARD}' to move one square forwards\
            - '{CMD_BACKWARDS}' to move one square backwards \
            - '{CMD_ROTATE_CLOCKWISE}' to rotate 90 degrees clockwise \
            - '{CMD_ROTATE_ANTICLOCKWISE}' to rotate 90 degrees clockwise \
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
      max_tokens=750
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
    print(f"Plan completed:")
    for m in self.global_conv:
      print(f"{m['role']}: {m['content']}")
      
    self.sn_ctrl.send("FINISHED")
    self.generate_summary()
    
  def generate_summary(self):
    self.global_conv.append({"role": "user", "content": "Generate a summarised numerical list of the plan for the steps that I should complete"})
    
    completion = self.client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=self.global_conv,
      max_tokens=750
    )

    # print(completion.choices[0].message)
    self.global_conv.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
  
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
    self.ready_lock.release()
  
  def is_ready(self):
    self.ready_lock.acquire()
    b = self.other_agent_ready
    self.ready_lock.release()
    return b

  def negotiate(self):
    current_stage = 0
    
    if self.this_agents_turn:
      self.global_conv.append({"role": "user", "content": f"I am at {self.grid}, you are at {ENDING_GRID_LOC}. I must end at {ENDING_GRID_LOC} and you must end at {STARTING_GRID_LOC}"})
    
    while(current_stage < self.max_stages):
      if(len(self.global_conv) > 0 and self.global_conv[len(self.global_conv)-1]["content"].endswith("@SUPERVISOR")):
        break;
      
      while(not self.is_my_turn()): # Wait to receive from the other agent
        self.wait_delay()
        print(f"Waiting for a response from another agent")
      
      self.send_req()
      self.toggle_turn()
      current_stage += 1
      print(f"Stage {current_stage}")
      print(f"{self.global_conv}");
        
    self.plan_completed()
    current_stage = 0
    
    while(not (self.sn_ctrl.rx_queue.empty() and self.sn_ctrl.tx_queue.empty())):
      print("Waiting for message queues to clear")
      self.wait_delay()

if __name__ == '__main__':
  x = LLM()
  print("Finished")
