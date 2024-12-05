import numpy as np
import typing as T
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
import rclpy

from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan

from std_msgs.msg import Bool #, OccupancyGrid
from nav_msgs.msg import Path, OccupancyGrid
    

class exploration_controller():
    def __init__(self, speed: float, exploration: float) -> None:
        super().__init__()

        self.speed = 0
        self.exploration = 0
        self.occupancy = None
        self.stochOccupancy = None
        self.state = None

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.state_sub = self.create_subscription(TurtleBotState, '/map', self.state_callback, 10)
        self.nav_success_sub = self.create_subscription(Bool, '/nav_success', self.nav_success_callback, 10)

        self.goal_pub= self.create_publisher(TurtleBotState, "/explore_goal", 10)

        # Internal state
        self.occupancy = None
        self.current_state = None
        self.navigation_finished  = False
        # Bool for complete navigation

    def map_callback(self, msg: OccupancyGrid) -> None:
        """ Callback triggered when the map is updated

        Args:
            msg (OccupancyGrid): updated map message
        """
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )

        # replan if the new map updates causes collision in the original plan
        if self.is_planned and not all([self.occupancy.is_free(s) for s in self.plan.path[1:]]):
            self.is_planned = False
            self.replan(self.goal)

    # Update the robot's current state
    def state_callback(self, msg: TurtleBotState):
        self.current_state = np.array([msg.x, msg.y, msg.theta])

    # Update navigation status based on success/failure
    def nav_success_callback(self, msg: Bool):
        self.navigation_finished = msg.data
        if self.navigation_finished:
            self.find_next_goal()

    def find_next_goal(self):
        if self.occupancy is not None and self.current_state is not None:
            frontier_states = self.explore(self.occupancy, self.current_state)

            # Find the closest frontier state to the current position
            distances = np.linalg.norm(frontier_states - self.current_state, axis=1)
            closest_frontier = frontier_states[np.argmin(distances)]

            #Add heurisitcs with number on large group of unexplored

            # Publish the closest frontier as the next goal
            goal_msg = TurtleBotState
            goal_msg.x = closest_frontier[0]
            goal_msg.y = closest_frontier[1]
            goal_msg.theta = closest_frontier[2]
            self.goal_pub.publish(goal_msg)

            self.navigation_in_progress = True

    # TurtleBotState vs np.ndarray
    def explore(occupancy: StochOccupancyGrid2D, current_state: np.ndarray) -> np.ndarray:
        """ returns potential states to explore
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

        # Compute the distance to the closest frontier state
        # current_state = np.array([6., 5.])
        # distances = np.linalg.norm(frontier_states - current_state, axis=1)
        # closest_distance = np.min(distances)
        # print(closest_distance)

        ########################### Code ends here ###########################
        return frontier_states

# msgs coming in, occupany grid msg to stoch class


if __name__ == "__main__":    
    rclpy.init()
    frontier_explore_node = exploration_controller()
    rclpy.spin(frontier_explore_node)
    rclpy.shutdown()
