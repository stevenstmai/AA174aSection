import numpy as np
import typing as T
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
import rclpy

from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw

from std_msgs.msg import Bool, OccupancyGrid
    

class exploration_controller():
    def __init__(self, speed: float, exploration: float) -> None:
        super().__init__()

        self.speed = 0
        self.exploration = 0
        self.occupancy = None
        self.stochOccupancy = None
        self.state = None

        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)  
        self.state_sub = self.create_subscription(OccupancyGrid, "/state", self.state_callback, 10)  

    # def init_localization(self):
        # Subscribe and get the occupancy grid from robot   

        # Convert to StochOccupancyGrid

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

    def state_callback(self, msg: TurtleBotState) -> None:
        """ Callback triggered when the map is updated

        Args:
            msg (OccupancyGrid): updated map message
        """
        self.state = msg

    # Wrapper function for finding our candidate states to explore based on exploration heuristics
    def find_frontierStates(occupancy: StochOccupancyGrid2D):
        # Find all the frontier states
        frontier_states = exploreMap.explore()

    def compute_explorationFrontier():
        StochOccupancyGrid2D map = init_localization()

        frontier_states = find_frontierStates()

        # Heuristic 1: Short and Feasible Path

        # Heuristic 2: A lot of current unexplored states

        # Heurisitc 3 (Extra): Try to find optimal exploration route (Harder)


class exploreMap(object):
    """Represents a motion planning problem to be solved by Frontier Exploration Heurisitcs"""
    def __init__(self, occupancy):
        self.occupancy = occupancy 

    def explore(occupancy, current_state):
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
        # Define convolution kernel for sliding window
        kernel = np.ones((occupancy.window_size, occupancy.window_size))

        # Binary masks for the occupancy map
        occupied_mask = (occupancy.probs >= occupancy.thresh)
        unoccupied_mask = (occupancy.probs < occupancy.thresh) & (occupancy.probs >= 0)
        unknown_mask = occupancy.probs == -1

        # Convolve binary masks to count cells in each category
        occupied_count = convolve2d(occupied_mask, kernel, mode="same", boundary="fill", fillvalue=0)
        unoccupied_count = convolve2d(unoccupied_mask, kernel, mode="same", boundary="fill", fillvalue=0)
        unknown_count = convolve2d(unknown_mask, kernel, mode="same", boundary="fill", fillvalue=0)

        # Heuristic 1: Unknown cells >= 20% of surrounding cells
        total_cells = occupancy.window_size**2
        heuristic1 = (unknown_count / total_cells) >= 0.2

        # Heuristic 2: No occupied cells in the window
        heuristic2 = occupied_count == 0

        # Heuristic 3: Unoccupied cells >= 30% of surrounding cells
        heuristic3 = (unoccupied_count / total_cells) >= 0.3

        # Combine all heuristics
        valid_frontier = heuristic1 & heuristic2 & heuristic3

        # Extract valid frontier states
        frontier_indices = np.argwhere(valid_frontier)
        frontier_states = occupancy.grid2state(frontier_indices)
        frontier_states = frontier_states[:, [1, 0]] # Adopting the row column indexing to X (horizontal), Y (Vertical) coordinate system

        # Compute the distance to the closest frontier state
        # current_state = np.array([6., 5.])
        distances = np.linalg.norm(frontier_states - current_state, axis=1)
        closest_distance = np.min(distances)
        print(closest_distance)

        ########################### Code ends here ###########################
        return frontier_states

# msgs coming in, occupany grid msg to stoch class


if __name__ == "__main__":    
    rclpy.init()
    
    explore = frontierExplore()
    rclpy.spin(explore)
        
    rclpy.shutdown()
