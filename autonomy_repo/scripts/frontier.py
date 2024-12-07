#!/usr/bin/env python3

import numpy as np
import typing as T
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
import rclpy
from rclpy.node import Node

from asl_tb3_lib.navigation import BaseNavigator, BaseController, TrajectoryPlan

from std_msgs.msg import Bool #, OccupancyGrid
from nav_msgs.msg import Path, OccupancyGrid
    

class exploration_controller(Node):
    def __init__(self) -> None:
        super().__init__("exploration_controller")

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.state_sub = self.create_subscription(TurtleBotState, '/state', self.state_callback, 10)
        self.nav_success_sub = self.create_subscription(Bool, '/nav_success', self.nav_success_callback, 10)
        self.cmd_nav= self.create_publisher(TurtleBotState, "/cmd_nav", 10)

        # Internal state
        self.occupancy: T.Optional[StochOccupancyGrid2D] = None
        self.current_position = None
        self.current_state: T.Optional[TurtleBotState] = None
        self.navigation_finished  = True
        self.goal_state = None
        self.init = True

        # self.detect_pub = self.create_subscription(Bool, "/detector_bool", self.detect_callback, 10)
        # self.declare_parameter("active", True)
        # self.startTime = None
        # self.detect_timer = self.create_timer(0.2, self.detector_timer)

    # @property
    # def active(self):
    #     return self.get_parameter("active").get_parameter_value().bool_value

    # def detect_callback(self, msg: Bool) -> None:
    #     if (msg.data == True and self.active == True):
    #         self.set_parameters([rclpy.Parameter("active", value=False)])     

    # def detector_timer(self):
    #     if (self.active):
    #         # DO nothing
    #         test = 0
    #     if (self.active == False):
    #         self.get_logger().info("NOT ACTIVE")
    #         if self.startTime is None:
    #             self.startTime = self.get_clock().now().nanoseconds / 1e9
    #             self.cmd_nav.publish(self.current_state)
    #             self.navigation_finished = True
                
    #         current_time = self.get_clock().now().nanoseconds / 1e9
            
    #         if (current_time - self.startTime > 5):
    #             self.set_parameters([rclpy.Parameter("active", value=True)])
    #             self.startTime = None
    #             self.cmd_nav.publish(self.goal_state) # Or find new goal
    #             self.navigation_finished = False 

    def map_callback(self, msg: OccupancyGrid) -> None:
        """ Callback triggered when the map is updated

        Args:   
            msg (OccupancyGrid): updated map message
        """ 
            
        self.occupancy = StochOccupancyGrid2D(
            resolution = msg.info.resolution,
            size_xy = np.array([msg.info.width, msg.info.height]),
            origin_xy = np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size =9,
            probs = msg.data,
        )
        
        # # replan if the new map updates causes collision in the original plan
        # if self.is_planned and not all([self.occupancy.is_free(s) for s in self.plan.path[1:]]):
        #     self.is_planned = False
        #     self.replan(self.goal_state)

    # Update the robot's current state
    def state_callback(self, msg: TurtleBotState):
        self.current_position = np.array([msg.x, msg.y])
        self.current_state = msg

        if (self.init == True):
            if (self.occupancy != None and self.current_state != None):
                    self.get_logger().info("Setting initial exploration goal...")
                    self.find_next_goal() 
                    self.init = False

    # Update navigation status based on success/failure
    def nav_success_callback(self, msg: Bool):
        # self.navigation_finished = msg.data
        # if self.navigation_finished:
        self.find_next_goal()

    def find_next_goal(self):
        if (self.occupancy != None and self.current_state != None):
            frontier_states = self.explore(self.occupancy, self.current_position)

            if (frontier_states.size == 0):
                self.get_logger().info("BBBBBBBBBBBB")
                self.cmd_nav.publish(self.current_state)

            # Find the closest frontier state to the current position
            distances = np.linalg.norm(frontier_states - self.current_position, axis=1)
            closest_frontier = frontier_states[np.argmax(distances)]

            self.get_logger().info(f"{closest_frontier}")
            self.get_logger().info(f"{self.current_position}")

            closest_distance = closest_frontier - self.current_position
            angle = np.arctan2(closest_distance[0], closest_distance[1])

            #Add heurisitcs with number on large group of unexplored

            goal_msg = TurtleBotState (
                    x = closest_frontier[0],
                    y = closest_frontier[1],
                    theta = angle
                )
            
            self.get_logger().info(f"{goal_msg}")
            
            self.goal_state = goal_msg
            self.cmd_nav.publish(goal_msg)
            self.navigation_finished = False     
              

    # TurtleBotState vs np.ndarray
    def explore(self, occupancy: StochOccupancyGrid2D, current_state: np.ndarray) -> np.ndarray:
        """ returns potential states to#!/usr/bin/env python3 explore
        Args:
            occupancy (StochasticOccupancyGrid2D): Represents the known, unknown, occupied, and unoccupied states. See class in first section of notebook.

        Returns:
            frontier_states (np.ndarray): state-vectors in (x, y) coordinates of potential states to explore. Shape is (N, 2), where N is the number of possible states to explore.

        HINTS:
        - Function `convolve2d` may be helpful in producing the number of unknown, and number of occupied states in a window of a specified cell
        - Note the distinction between physical states and grid cells. Most operations can be done on grid cells, and converted to physical states at the end of the function with `occupancy.grid2state()`
        """

        window_size = 13   # defines the window side-length for neighborhood of cells to consider for heuristics
        ########################### Code starts here ###########################
        occupancy.window_size = window_size
        kernel = np.ones((occupancy.window_size, occupancy.window_size))

        occupied_mask = (occupancy.probs >= occupancy.thresh)
        unoccupied_mask = (occupancy.probs < occupancy.thresh) & (occupancy.probs >= 0)
        unknown_mask = occupancy.probs == -1

        occupied_count = convolve2d(occupied_mask, kernel, mode="same", boundary="fill", fillvalue=0)
        unoccupied_count = convolve2d(unoccupied_mask, kernel, mode="same", boundary="fill", fillvalue=0)
        unknown_count = convolve2d(unknown_mask, kernel, mode="same", boundary="fill", fillvalue=0)

        total_cells = occupancy.window_size**2
        heuristic1 = (unknown_count / total_cells) >= 0.2
        heuristic2 = occupied_count == 0
        heuristic3 = (unoccupied_count / total_cells) >= 0.3

        valid_frontier = heuristic1 & heuristic2 & heuristic3
        frontier_indices = np.argwhere(valid_frontier)
        frontier_states = occupancy.grid2state(frontier_indices)
        # frontier_states = frontier_states[:, [1, 0]] # Adopting the row column indexing to X (horizontal), Y (Vertical) coordinate system
        ########################### Code ends here ###########################
        return frontier_states

# msgs coming in, occupany grid msg to stoch class


if __name__ == "__main__":    
    rclpy.init()
    frontier_explore_node = exploration_controller()
    rclpy.spin(frontier_explore_node)
    rclpy.shutdown()
