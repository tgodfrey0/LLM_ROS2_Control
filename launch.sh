#!/bin/bash

tmux new-session -d -s llm_control
tmux split-window -h
tmux split-window -v

tmux new-window -n ros2-bg -t llm_control:2
tmux split-window -h
tmux split-window -v

tmux attach-session -t llm_control