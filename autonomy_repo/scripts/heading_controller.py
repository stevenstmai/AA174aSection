#!/usr/bin/env python3
import numpy
import rclpy

from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):
    def __init__(self):
        super().__init__()
        self.declare_parameter("kp", 1.1)
    
    def compute_control_with_goal(self, current_state: TurtleBotControl, goal_state: TurtleBotControl):
        heading_error = goal_state.theta - current_state.theta
        heading_error = wrap_angle(heading_error)

        control = self.get_parameter("kp").value * heading_error

        msg = TurtleBotControl()
        msg.omega = control
        return msg
    

if __name__ == "__main__":
    rclpy.init()
    controller = HeadingController()
    rclpy.spin(controller)
    rclpy.shutdown()

