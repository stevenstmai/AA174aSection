#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node

from asl_tb3_lib.navigation import BaseController
from asl_tb3_msgs.msg import TurtleBotControl

class PerceptionController(BaseController):
    def __init__(self):
        super().__init__("perception_controller")
        self.declare_parameter("active", True)
        self.startTime = None

    
    @property
    def active(self):
        return self.get_parameter("active").get_parameter_value().bool_value
    
    def compute_control(self):
        control_msg = TurtleBotControl()
        
        if self.active:
            control_msg.omega = 0.5
        else:
            self.get_logger().info("NOT ACTIVE")
            if self.startTime is None:
                self.startTime = self.get_clock().now().nanoseconds / 1e9
                
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            if (current_time - self.startTime > 5):
                self.set_parameters([rclpy.Parameter("active", value=True)])
                self.startTime = None
            control_msg.omega = 0.0
            
        return control_msg
    

if __name__ == "__main__":
    rclpy.init()
    controller = PerceptionController()
    rclpy.spin(controller)
    rclpy.shutdown()

    
    
        