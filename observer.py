from swarmnet import SwarmNet, Log_Level, set_log_level
from typing import Optional, List, Tuple

dl: List[Tuple[str, int]] = [("10.0.1.112", 51000), ("10.0.1.191", 51000)] # Other device

def llm_recv(msg: Optional[str]) -> None:
  print("SwarmNet LLM message sent")

def ready_recv(msg: Optional[str]) -> None:
  print(f"SwarmNet READY: {msg.strip()}")

def finished_recv(msg: Optional[str]) -> None:
  print("SwarmNet FINISHED message sent")

def info_recv(msg: Optional[str]) -> None:
  print(f"SwarmNet INFO from {msg.strip()}")

if __name__=="__main__":
  sn = SwarmNet({"LLM": llm_recv, "READY": ready_recv, "FINISHED": finished_recv, "INFO": info_recv}, device_list = dl)
  set_log_level(Log_Level.WARN)
  sn.start()
  
  input()
  
  sn.kill()