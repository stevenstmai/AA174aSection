#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# import the message type to use
from std_msgs.msg import Int64, Bool

from geometry_msgs.msg import Twist


class Heartbeat(Node):
    def __init__(self) -> None:
                                # initialize base class (must happen before everything else)
        super().__init__("heartbeat")  # initialize base class
                               
                                # a heartbeat counter
        self.hb_counter = 0
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.alive = True
                                # create publisher with
                                #   self.create_publisher(<msg type>, <topic>, <qos>)
        self.hb_pub = self.create_publisher(Int64, "/heartbeat", 10)
        self.twist_pub = self.create_publisher(Twist, "/cmd_vel", 10)
       
        # create a timer
                                #   self.create_timer(<second>, <callback>)
        self.hb_timer = self.create_timer(0.2, self.hb_callback)
        self.twist_timer = self.create_timer(0.2, self.twist_callback)

                                # create subscription with
                                #   self.create_subscription(<msg type>, <topic>, <callback>, <qos>)
        self.motor_sub = self.create_subscription(Bool, "/health/motor",
                                                  self.health_callback, 10)
        
        self.kill_sub = self.create_subscription(Bool, "/kill",
                                                  self.kill_callback, 10)

    def hb_callback(self) -> None:
        """ heartbeat callback triggered by the timer """
        # construct heartbeat message
        msg = Int64()
        msg.data = self.hb_counter

        # publish heartbeat counter
        self.hb_pub.publish(msg)

        # counter increment
        self.hb_counter += 1
    
    def twist_callback(self) -> None:
        if (self.alive) :
            msg = Twist()
            msg.linear.x = self.linear_x
            msg.angular.z = self.angular_z
            self.twist_pub.publish(msg)
            self.linear_x += 1.0
            self.angular_z += 1.0   

    def health_callback(self, msg: Bool) -> None:
        """ sensor health callback triggered by subscription """
        if not msg.data:
            self.get_logger().fatal("Heartbeat stopped")
            self.hb_timer.cancel()

    def kill_callback(self, msg: Bool) -> None:
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.alive = False
        self.twist_pub.publish(msg)

if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    node = Heartbeat()  # instantiate the heartbeat node
    rclpy.spin(node)    # Use ROS2 built-in schedular for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 context