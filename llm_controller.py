from SwarmNet.swarmnet import SwarmNet
from openai import OpenAI

global_conv = []
client: OpenAI = None
  
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
  
def plan_completed():
  print("Plan completed:")
  # map(lambda m : print(f"{m.role}: {m.content}"), global_conv)
  for m in global_conv:
    print(f"{m['role']}: {m['content']}")
  
def llm_recv(msg: str) -> None: 
  m = msg.split(" ", 1) # Msg are LLM ROLE CONTENT
  r = m[0]
  c = m[1]
  global_conv.append({"role": toggle_role(r), "content": c}) #! Don't think this is adding to the list
  if("@SUPERVISOR" not in c):
    send_req(client)
  else:
    plan_completed() #? This may have issues with only one agent finishing. Could just add a SN command

if __name__=="__main__":
  sn_ctrl = SwarmNet({"LLM": llm_recv})
  sn_ctrl.start()
  print("Communications started")
  input("Press any key to start")
  client = OpenAI(api_key=get_api_key())
  global_conv = [
    {"role": "system", "content": "You and I are wheeled robots, and can only move forwards, backwards, and rotate clockwise or anticlockwise.\
      We will negotiate with other robots to navigate a path without colliding. You should negotiate and debate the plan until all agents agree.\
        Once this has been decided you should call the '@SUPERVISOR' tag at the end of your plan and print your plan in a concise numbered list using only the following command words:\
          - 'FORWARDS' to move one square forwards\
          - 'BACKWARDS' to move one square backwards\
          - 'CLOCKWISE' to rotate 90 degrees clockwise\
          - 'ANTICLOCKWISE' to rotate 90 degrees clockwise\
          "}, 
    {"role": "user", "content": "I am at D1, you are at D7. I must end at D7 and you must end at D1"}
  ]
  res = send_req(client)
  print(res.content)
  sn_ctrl.send(f"LLM {res.role} {res.content}")
  input("Press any key to finish")
  plan_completed()
  sn_ctrl.kill()