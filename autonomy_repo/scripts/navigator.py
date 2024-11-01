#!/usr/bin/env python3

import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
import rclpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import plot_line_segments

from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw

V_PREV_THRES = 0.0001

class Navigator(BaseNavigator):
    def __init__(self, kpx: float, kpy: float, kdx: float, kdy: float,
                 V_max: float = 0.5, om_max: float = 1) -> None:
        super().__init__()

        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy

        self.V_max = V_max
        self.om_max = om_max

    def compute_heading_control(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        heading_error = goal.theta - state.theta
        heading_error = wrap_angle(heading_error)

        control = self.get_parameter("kp").value * heading_error

        msg = TurtleBotControl()
        msg.omega = control
        return msg
    
    def compute_trajectory_tracking_control(self, state: TurtleBotState, plan: TrajectoryPlan, t: float) -> TurtleBotControl:
        """
        Inputs:
            state: Current state
            plan: Trajectory
            t: Current time
        Outputs:
            V, om: Control actions
        """

        dt = t - self.t_prev
        x_d = float(scipy.interpolate.splev(dt, plan.path_x_spline, der = 0))
        y_d = float(scipy.interpolate.splev(dt, plan.path_y_spline, der = 0))
        xd_d = float(scipy.interpolate.splev(dt, plan.path_x_spline, der = 1))
        yd_d = float(scipy.interpolate.splev(dt, plan.path_y_spline, der = 1))
        xdd_d = float(scipy.interpolate.splev(dt, plan.path_x_spline, der = 2))
        ydd_d = float(scipy.interpolate.splev(dt, plan.path_y_spline, der = 2))

        # REFERENCE 2
        # # Compute control errors
        # error_x = x_d - state.x
        # error_y = y_d - state.y

        # # Compute desired heading angle to reach the goal
        # desired_theta = math.atan2(error_y, error_x)

        # ompute heading error
        # heading_error = desired_theta - state.theta
        # heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-pi, pi]

        # # Compute feedforward linear and angular velocities
        # v_ff = math.sqrt(xd_d**2 + yd_d**2)
        # omega_ff = (ydd_d * xd_d - xdd_d * yd_d) / (xd_d**2 + yd_d**2) if (xd_d**2 + yd_d**2) != 0 else 0.0

        # # Proportional control for velocity
        # k_v = 1.0  # Gain for linear velocity, tune as needed
        # k_omega = 1.0  # Gain for angular velocity, tune as needed

        # # Compute final control velocities
        # v = v_ff + k_v * math.sqrt(error_x**2 + error_y**2)
        # omega = omega_ff + k_omega * heading_error

        # # Create and return a control message
        # control_command = TurtleBotControl(v=v, omega=omega)

        # REFERENCE 1
        u1 = xdd_d + self.kpx*(x_d - x) + self.kdx*(xd_d - self.V_prev*np.cos(th))
        u2 = ydd_d + self.kpy*(y_d - y) + self.kdy*(yd_d - self.V_prev*np.sin(th))
        
        vdot = u1*np.cos(th) + u2*np.sin(th)
        V = self.V_prev + vdot * dt # np.sqrt(u1**2 + u2**2)

        if V < V_PREV_THRES:
            V = self.V_prev 


        if V > V_PREV_THRES:
            om = (u2*np.cos(th) - u1*np.sin(th)) / V
        else:
            om = 0

        # Nolan Code
        # V = xd_d * np.cos(state.theta) + yd_d * np.sin(state.theta)
        # om = ydd_d - self.kdx * xd_d * np.sin(state.theta) + self.kdy * yd_d * np.cos(state.theta)

        # save the commands that were applied and the time

        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        return V, om
    
    def compute_trajectory_plan(self, state: TurtleBotState, goal: TurtleBotState, occupancy: StochOccupancyGrid2D, resolution: float, horizon: float) -> TrajectoryPlan | None:
        # constants arbitrarily defined
        width = 10
        height = 10
        spline_alpha = 0.3
        # v_desired = 5
        lower_bounds = (0,0)
        higher_bounds = (width, height)

        # Reference 1
        # Step 1: Initialize and solve A* problem
        astar = AStar(lower_bounds, higher_bounds, state.x, goal.x, occupancy, resolution)
        if not astar.solve() or len(astar.path) < 4:
            return None

        # Step 2: Reset class variables for previous velocity and integration
        self.previous_velocity = 0.0  # Example reset
        self.previous_time = 0.0

        # Step 3: Compute timestamps for the path waypoints
        path = np.array(astar.path)
        distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cumulative_distances = np.hstack(([0], np.cumsum(distances)))
        time_stamps = cumulative_distances / horizon  # Assume uniform velocity

        # Step 4: Generate cubic splines for x and y coordinates
        x_coords = path[:, 0]
        y_coords = path[:, 1]
        x_spline = splrep(time_stamps, x_coords, s=spline_alpha )  # Adjust smoothing factor as needed
        y_spline = splrep(time_stamps, y_coords, s=spline_alpha )


        # Nolan Code
        # astar = AStar((0, 0), (width, height), state.x, goal.x, occupancy, resolution)
        # if astar.solve():
        #     self.path = astar.reconstruct_path()
        # else:
        #     return None
        # # else:
        # #     plt.rcParams['figure.figsize'] = [10, 10]
        # #     astar.plot_path()
        # #     astar.plot_tree(point_size=2)

        
        # dt = np.zeros(self.path.shape[0])
        # dt[1:] = np.linalg.norm(self.path[1:] - self.path[:-1], axis=1) / v_desired
        # ts = np.cumsum(dt)
        
        # path_x_spline = scipy.interpolate.splrep(ts, self.path[:,0], k=3, s=spline_alpha)
        # path_y_spline = scipy.interpolate.splrep(ts, self.path[:,1], k=3, s=spline_alpha)

        return TrajectoryPlan(
            path=self.path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )


class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        # Check if x is inside the state space bounds
        if not (self.statespace_lo[0] <= x[0] <= self.statespace_hi[0] and 
                self.statespace_lo[1] <= x[1] <= self.statespace_hi[1]):
            return False
        # Check if x is inside an obstacle
        return self.occupancy.is_free(x)
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        return np.linalg.norm(np.array(x1) - np.array(x2))
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        # Define the possible movements (8-connected grid)
        directions = [(0, self.resolution), (0, -self.resolution), (self.resolution, 0), (-self.resolution, 0),
                      (self.resolution, self.resolution), (-self.resolution, self.resolution), 
                      (self.resolution, -self.resolution), (-self.resolution, -self.resolution)]
        for direction in directions:
            neighbor = (x[0] + direction[0], x[1] + direction[1])
            # Snap the neighbor to grid and check if it's free
            neighbor = self.snap_to_grid(neighbor)
            if self.is_free(neighbor):
                neighbors.append(neighbor)
        return neighbors

        # check_neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        # for check in check_neighbors:
        #     neighbor = (x[0] + check[0]*self.resolution, x[1] + check[1]*self.resolution)
        #     neighbor = self.snap_to_grid(neighbor)
        #     if self.is_free(neighbor):
        #         neighbors.append(neighbor)

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def plot_path(self, fig_num=0, show_init_label=True):
        """Plots the path found in self.path and the obstacles"""
        if not self.path:
            return

        self.occupancy.plot(fig_num)

        solution_path = np.asarray(self.path)
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        if show_init_label:
            plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    def plot_tree(self, point_size=15):
        plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        
        while self.open_set:
            current = self.find_best_est_cost_through()

            if current == self.x_goal:
                self.path = self.reconstruct_path()
                return True

            self.open_set.remove(current)
            self.closed_set.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_set:
                    continue

                tentative_cost = self.cost_to_arrive[current] + self.distance(current, neighbor)

                if neighbor not in self.open_set:
                    self.open_set.add(neighbor)
                elif tentative_cost >= self.cost_to_arrive.get(neighbor, float('inf')):
                    continue

                self.came_from[neighbor] = current
                self.cost_to_arrive[neighbor] = tentative_cost
                self.est_cost_through[neighbor] = tentative_cost + self.distance(neighbor, self.x_goal)

        return False 

        # while len(self.open_set) > 0:
        #     current_state = self.find_best_est_cost_through()
            
        #     self.open_set.remove(current_state)
        #     self.closed_set.add(current_state)
            
        #     if current_state == self.x_goal:
        #         self.path = self.reconstruct_path()
        #         return True

        #     for neighbor in self.get_neighbors(current_state):
        #         if neighbor in self.closed_set:
        #             continue
                
        #         current_cost = self.cost_to_arrive[current_state] + self.distance(current_state, neighbor)
                
        #         if neighbor not in self.open_set:
        #             self.open_set.add(neighbor)
        #         elif current_cost >= self.cost_to_arrive[neighbor]:
        #             continue
                
        #         self.came_from[neighbor] = current_state
        #         self.cost_to_arrive[neighbor] = current_cost
        #         self.est_cost_through[neighbor] = current_cost + self.distance(neighbor, self.x_goal)
        
        # return False   

class DetOccupancyGrid2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic
    """
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        """Verifies that point is not inside any obstacles by some margin"""
        for obs in self.obstacles:
            if x[0] >= obs[0][0] - self.width * .01 and \
               x[0] <= obs[1][0] + self.width * .01 and \
               x[1] >= obs[0][1] - self.height * .01 and \
               x[1] <= obs[1][1] + self.height * .01:
                return False
        return True

    def plot(self, fig_num=0):
        """Plots the space and its obstacles"""
        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111, aspect='equal')
        for obs in self.obstacles:
            ax.add_patch(
            patches.Rectangle(
            obs[0],
            obs[1][0]-obs[0][0],
            obs[1][1]-obs[0][1],))
        ax.set(xlim=(0,self.width), ylim=(0,self.height))


if __name__ == "__main__":    
    rclpy.init()
    nav = Navigator()
    rclpy.spin(nav)

    

    rclpy.shutdown()
