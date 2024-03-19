from swarmnet import SwarmNet, Log_Level, set_log_level
from typing import Optional, List, Tuple
from colorama import Fore, Style
import datetime

dl = [("192.168.193.64", 51000), ("192.168.193.58", 51000), ("192.168.193.203", 51000)]

filename = "/home/tg/projects/p3p/LLM_ROS2_Control/logs/observer/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.txt")

def llm_recv(msg: Optional[str]) -> None:
  print("LLM message sent")

def ready_recv(msg: Optional[str]) -> None:
  print(f"READY: {msg.strip()}")

def move_recv(msg: Optional[str]) -> None:
  print(f"MOVE")

def finished_recv(msg: Optional[str]) -> None:
  print("FINISHED message sent")
  with open(filename, "a") as file:
    file.write(f"FINISHED message sent\n")

def info_recv(msg: Optional[str]) -> None:
  name = msg.split(":", 1)
  if name[0] == "Alice":
    print(f"{Fore.LIGHTRED_EX}INFO from {msg.strip()}{Style.RESET_ALL}")
  else:
    print(f"{Fore.BLUE}INFO from {msg.strip()}{Style.RESET_ALL}")

  with open(filename, "a") as file:
    file.write(f"INFO from {msg.strip()}\n")
    
def restart_recv(msg: Optional[str]) -> None:
  print("Restart message sent")
  with open(filename, "a") as file:
    file.write(f"Restart message sent\n")

if __name__=="__main__":
  global now
  sn = SwarmNet({"LLM": llm_recv, "READY": ready_recv, "FINISHED": finished_recv, "INFO": info_recv, "RESTART": restart_recv, "MOVE": move_recv}, device_list = dl)
  set_log_level(Log_Level.WARN)
  sn.start()
  
  input()
  
  sn.kill()