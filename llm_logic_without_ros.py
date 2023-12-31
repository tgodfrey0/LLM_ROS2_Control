from swarmnet import SwarmNet
from openai import OpenAI
from math import pi
from threading import Lock
from typing import Optional, List, Tuple
import os
from time import sleep
import threading

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

WAITING_TIME = 1

class LLM():
  def __init__(self):
    self.global_conv = []
    self.client: OpenAI = None
    self.max_stages = 5
    self.this_agents_turn = True
    self.other_agent_ready = False
    self.turn_lock = Lock()
    self.ready_lock = Lock()
  
    self.create_plan()
    
    print(f"Full plan parsed")
        
  def _delay(self, t_target):
    sleep(t_target)
    print(f"Delayed for {t_target} seconds")
    
  def wait_delay(self):
    self._delay(WAITING_TIME)
    
  def t(self, sn_ctrl: SwarmNet):
    while(True):
      print(f"Ready: {self.is_ready()}")
      print(f"Turn: {self.is_my_turn()}")
      self._delay(0.5)
    
  def create_plan(self):
    print(f"Initialising SwarmNet")
    self.sn_ctrl = SwarmNet({"LLM": self.llm_recv, "READY": self.ready_recv, "FINISHED": self.generate_summary}, device_list = dl)
    self.sn_ctrl.start()
    print(f"SwarmNet initialised") 
    
    t1 = threading.Thread(target=self.t, args=[self.sn_ctrl])
    t1.start()
  
    self.sn_ctrl.send("READY")
  
    while(not self.is_ready()):
      self.sn_ctrl.send(f"READY")
      print("Waiting for an agent to be ready")
      self.wait_delay()

    self.sn_ctrl.clear_rx_queue()
    self.sn_ctrl.clear_tx_queue()
    
    #! Agent started second cannot proceed past READY synchronisation
    
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
    self.generate_summary(None)
    
  def generate_summary(self, msg: Optional[str]):
    self.global_conv.append({"role": "user", "content": "Generate a summarised numerical list of the plan"})
    
    completion = self.client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=self.global_conv,
      max_tokens=750
    )

    # print(completion.choices[0].message)
    self.global_conv.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
    
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
      self.global_conv.append({"role": "user", "content": "I am at D1, you are at D7. I must end at D7 and you must end at D1"})
    
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

if __name__ == '__main__':
  x = LLM()
  print("Finished")
