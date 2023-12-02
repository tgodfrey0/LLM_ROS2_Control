import cv2
import apriltag
import openai

from SwarmNet.swarmnet import SwarmNet
from openai import OpenAI

at_options = apriltag.DetectorOptions(families="tag36h11")
tag_width = 10
global_conv = []
  
def send_req(client: OpenAI) -> str:
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=global_conv,
    # max_tokens=500
  )

  # print(completion.choices[0].message)
  
  return completion.choices[0].message
  
def get_api_key() -> str:
  with open("openai_key", "r") as f:
    return f.readline()
  
def toggle_role(r: str):
  if r == "assistant":
    return "user"
  elif r == "user":
    return "assistant"
  else:
    return ""
  
def llm_recv(msg: str) -> None: 
  m = msg.split(" ", 1) # Msg are LLM ROLE CONTENT
  r = m[0]
  c = m[1]
  global_conv.append({"role": toggle_role(r), "content": c})

if __name__=="__main__":
  sn_ctrl = SwarmNet({"LLM": llm_recv})
  sn_ctrl.start()
  print("Communications started")
  client = OpenAI(api_key=get_api_key())
  global_conv = [
    {"role": "system", "content": "You are a wheeled robot, and can only move forwards, backwards, and rotate clockwise or anticlockwise.\
      You will negotiate with other robots to navigate a path without colliding. You should negotiate and debate the plan until all agents agree.\
        Once this has been decided you should call the '@SUPERVISOR' tag at the end of your plan and print your plan in a concise numbered list using only the following command words:\
          - 'FORWARDS' to move one square forwards\
          - 'BACKWARDS' to move one square backwards\
          - 'CLOCKWISE' to rotate 90 degrees clockwise\
          - 'ANTICLOCKWISE' to rotate 90 degrees clockwise\
          "}, 
    {"role": "user", "content": "Create a plan to move on a chess board from B7 to F7 without colliding with the agent at D7"}
  ]
  res = send_req(client)
  print(res.content)
  sn_ctrl.send(f"LLM {res.role} {res.content}")
  input("Press any key to finish")
  sn_ctrl.kill()