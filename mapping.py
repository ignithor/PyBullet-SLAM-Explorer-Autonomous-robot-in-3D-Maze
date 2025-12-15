import numpy as np

# --- Constants ---
# Must match the physical values of the robot
LIDAR_RANGE = 5.0    # Max Lidar range in meters
LIDAR_RAYS = 36      # Number of rays (must match robot.py)

class Slam:
    """
    Implements an Occupancy Grid SLAM algorithm.
    
    The principle is to divide the world into a grid. Each cell contains
    a 'log-odds' value representing the probability of being an obstacle.
    """
    
    # --- Log-Odds Constants (Probabilities) ---
    # We use logs to simply add probabilities.
    # > 0 : Likely an obstacle
    # < 0 : Likely free
    LOG_ODDS_HIT = 0.9    # Value added if the laser hits an obstacle (Bonus)
    LOG_ODDS_FREE = 0.4   # Value subtracted if the laser passes through the cell (Malus)
    LOG_ODDS_CLAMP = 20.0 # Min/max value to avoid infinity (clamping)

    def __init__(self, map_size_m=40.0, map_resolution=0.1):
        """
        Initializes the grid.
        
        Args:
            map_size_m (float): World size in meters (e.g., 40x40m).
            map_resolution (float): Size of a cell in meters (e.g., 0.1 = 10cm).
        """
        self.map_size_m = map_size_m
        self.resolution = map_resolution
        
        # Calculate number of cells (e.g., 40m / 0.1m = 400 cells wide)
        self.size_cells = int(self.map_size_m / self.resolution)
        
        # --- MAP ORIGIN ADJUSTMENT ---
        # The simulation maze starts at (0,0) and extends to positive X and Y.
        # Instead of centering the map (origin = -size/2), we set the origin 
        # to a small negative value. This puts (0,0) near the bottom-left corner,
        # ensuring the whole maze fits in the map.
        self.origin_m = -5.0
        
        # Create map matrix (filled with 0 = unknown)
        self.map = np.zeros((self.size_cells, self.size_cells), dtype=np.float32)
        
        # Pre-calculate Lidar angles (0 to 2*Pi) to save time
        # 0° = Robot's right, 90° = Front
        self.lidar_angles = np.linspace(0, 2 * np.pi, LIDAR_RAYS, endpoint=False)

        print(f"INFO: SLAM initialized. Map: {self.size_cells}x{self.size_cells} cells ({map_size_m}x{map_size_m}m).")
        print(f"INFO: Map Origin: {self.origin_m}m (World (0,0) is near bottom-left)")

    def world_to_grid(self, x_world, y_world):
        """
        Step 2: Convert World Coordinates (meters) -> Grid Coordinates (indices).
        """
        # Formula: (Position - Origin) / Resolution
        x_grid = int((x_world - self.origin_m) / self.resolution)
        y_grid = int((y_world - self.origin_m) / self.resolution)

        # Check if outside the map
        if 0 <= x_grid < self.size_cells and 0 <= y_grid < self.size_cells:
            return (x_grid, y_grid)
        return None

    def update(self, robot_pos, robot_yaw, lidar_data):
        """
        Step 3 & 4: Updates the map with a new Lidar reading.
        
        Args:
            robot_pos (list): [x, y] current robot position.
            robot_yaw (float): Current robot angle (orientation).
            lidar_data (array): List of distances measured by the Lidar.
        """
        robot_x, robot_y = robot_pos
        
        # Use 'set' to store unique cells to update
        # (avoids updating the same cell 10 times for a single scan)
        free_cells = set()
        occupied_cells = set()

        # For each Lidar ray...
        for i, dist in enumerate(lidar_data):
            # 1. Calculate the actual ray angle in the world
            world_angle = robot_yaw + self.lidar_angles[i]
            
            # 2. Calculate the exact hit position (or ray end)
            hit_x = robot_x + dist * np.cos(world_angle)
            hit_y = robot_y + dist * np.sin(world_angle)
            
            # Check if it's a real detection or just max range (sky)
            is_max_range = (dist >= LIDAR_RANGE - 0.1)

            # --- Ray Tracing (Geometric Algorithm) ---
            # Traverse the line between robot and hit
            vec_x = hit_x - robot_x
            vec_y = hit_y - robot_y
            num_steps = int(dist / self.resolution) # Number of cells to traverse

            for step in range(num_steps):
                # Step incrementally along the line
                t = step / num_steps if num_steps > 0 else 0
                p_x = robot_x + (t * vec_x)
                p_y = robot_y + (t * vec_y)
                
                cell = self.world_to_grid(p_x, p_y)
                if cell:
                    free_cells.add(cell) # This cell is FREE

            # --- Marking the obstacle ---
            if not is_max_range:
                hit_cell = self.world_to_grid(hit_x, hit_y)
                if hit_cell:
                    occupied_cells.add(hit_cell) # This cell is OCCUPIED

        # --- Map Matrix Update (Log-Odds) ---
        
        # Mark free cells (Subtraction)
        for (gx, gy) in free_cells:
            # Do not mark as free a cell just detected as occupied
            if (gx, gy) not in occupied_cells:
                 self.map[gy, gx] -= self.LOG_ODDS_FREE

        # Mark occupied cells (Addition)
        for (gx, gy) in occupied_cells:
            self.map[gy, gx] += self.LOG_ODDS_HIT

        # Saturation (Clamp) to avoid infinite values
        np.clip(self.map, -self.LOG_ODDS_CLAMP, self.LOG_ODDS_CLAMP, out=self.map)

    def get_map_probabilities(self):
        """
        Step 5: Converts the log-odds map to probabilities (0.0 to 1.0) for display.
        0.0 = White (Free), 1.0 = Black (Wall), 0.5 = Gray (Unknown)
        """
        # Sigmoid Formula: p = exp(val) / (1 + exp(val))
        exp_map = np.exp(self.map)
        prob_map = exp_map / (1.0 + exp_map)
        return prob_map