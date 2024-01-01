from swarmnet import SwarmNet
from typing import Optional

dl: List[Tuple[str, int]] = [("192.168.0.120", 51000), ("192.168.0.121", 51000)] # Other device

def llm_recv(msg: Optional[str]) -> None:
  pass

def ready_recv(msg: Optional[str]) -> None:
  pass

def finished_recv(msg: Optional[str]) -> None:
  pass

def info_recv(msg: Optional[str]) -> None:
  print("SwarmNet INFO: msg")

if __name__=="__main__":
  sn = SwarmNet({"LLM": llm_recv, "READY": ready_recv, "FINISHED": finished_recv, "INFO": info_recv}, device_list = dl)
  sn.start()
  
  input()
  
  sn.kill()