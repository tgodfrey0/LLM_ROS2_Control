#!/bin/bash

# Create the session named "llm_control" in detached mode
tmux new-session -d -s llm_control
tmux split-window -h
tmux split-window -v

# Create the second window named "ros2-bg"
tmux new-window -n ros2-bg -t llm_control:2
tmux split-window -h
tmux split-window -v

# Optional: Attach to the session if you want to start working in it immediately
tmux attach-session -t llm_control