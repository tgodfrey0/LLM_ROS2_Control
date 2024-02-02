from swarmnet import SwarmNet, Log_Level, set_log_level
from typing import Optional, List, Tuple
import datetime

dl: List[Tuple[str, int]] = [("10.0.1.112", 51000), ("10.0.1.192", 51000)] # Other device
filename = "/home/tg/projects/p3p/LLM_ROS2_Control/logs/observer/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.txt")

def llm_recv(msg: Optional[str]) -> None:
  print("LLM message sent")

def ready_recv(msg: Optional[str]) -> None:
  print(f"READY: {msg.strip()}")

def finished_recv(msg: Optional[str]) -> None:
  print("FINISHED message sent")
  with open(filename, "a") as file:
    file.write(f"FINISHED message sent\n")

def info_recv(msg: Optional[str]) -> None:
  print(f"INFO from {msg.strip()}")
  with open(filename, "a") as file:
    file.write(f"INFO from {msg.strip()}\n")
    
def restart_recv(msg: Optional[str]) -> None:
  print("Restart message sent")
  with open(filename, "a") as file:
    file.write(f"Restart message sent\n")

if __name__=="__main__":
  global now
  sn = SwarmNet({"LLM": llm_recv, "READY": ready_recv, "FINISHED": finished_recv, "INFO": info_recv, "RESTART": restart_recv}, device_list = dl)
  set_log_level(Log_Level.WARN)
  sn.start()
  
  input()
  
  sn.kill()