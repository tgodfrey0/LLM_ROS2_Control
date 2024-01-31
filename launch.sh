#!/bin/bash

# Start a new Tmux session named "llm_control"
tmux new-session -d -s llm_control -n main

# Create the "main" window with three panes
tmux split-window -h
tmux split-window -v -t 1

# Create the "ros2-bg" window with four panes
tmux new-window -n ros2-bg
tmux split-window -v
tmux split-window -h -t 0
tmux split-window -h -t 2

# Attach to the "llm_control" session
tmux attach-session -t llm_control
