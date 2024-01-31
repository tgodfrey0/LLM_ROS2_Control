from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='llm_controller',
            executable='llm_node',
            name='llm_node',
            parameters=[{"config_file": "/home/ubuntu/LLM_ROS2_Control/configs/config_alice.yaml"}],
            output="screen"
        )
    ])