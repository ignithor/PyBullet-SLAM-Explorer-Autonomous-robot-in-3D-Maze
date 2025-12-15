import numpy as np
from path_planner import PathPlanner

class FrontierExplorer:
    """
    Advanced Frontier Exploration Module.
    Features:
    - Obstacle Inflation: Ensures robot stays away from walls.
    - Frontier Clustering: Targets largest 'zones' of unknown space rather than random pixels.
    - A* Path Planning: Computes feasible paths around obstacles.
    - PID Control: Smooth path following.
    """
    def __init__(self):
        self.target_grid = None 
        self.reached_threshold_m = 0.5
        
        # Path Planner
        self.planner = PathPlanner()
        self.current_path = [] 
        self.path_index = 0
        
        # PID Control parameters
        self.max_linear_speed = 10.0 
        self.lookahead_dist = 1.0     
        
        # PID Gains
        self.kp_angular = 6.0   
        self.ki_angular = 0.01  
        self.kd_angular = 1.0   
        
        # PID State
        self.prev_angle_err = 0.0
        self.integral_angle_err = 0.0
        
        # Safety: Dilation radius for walls
        # Reduced to 2 cells (0.2m) to allow passing through maze corridors
        self.safety_margin_cells = 2

    def reset_pid(self):
        """Resets PID integral and derivative terms."""
        self.prev_angle_err = 0.0
        self.integral_angle_err = 0.0

    def get_control_command(self, robot_pose, map_probs, map_origin, map_resolution):
        rx, ry, ryaw = robot_pose
        
        # 1. Convert Robot Position to Grid
        gx = int((rx - map_origin) / map_resolution)
        gy = int((ry - map_origin) / map_resolution)
        
        height, width = map_probs.shape
        
        # --- 2. Build Safe Navigation Map (Inflation) ---
        # Identify walls (> 0.6 probability)
        raw_obstacles = map_probs > 0.6
        
        # Dilate obstacles to create a safety buffer (manual convolution)
        inflated_obstacles = raw_obstacles.copy()
        for dx in range(-self.safety_margin_cells, self.safety_margin_cells + 1):
            for dy in range(-self.safety_margin_cells, self.safety_margin_cells + 1):
                if dx == 0 and dy == 0: continue
                # Roll shift array
                shifted = np.roll(raw_obstacles, shift=(dy, dx), axis=(0, 1))
                inflated_obstacles |= shifted
        
        # Create Binary Grid for A* (1 = Unsafe/Wall, 0 = Safe)
        binary_grid = np.zeros_like(map_probs, dtype=np.int8)
        binary_grid[inflated_obstacles] = 1 
        
        # --- 3. Identify Frontiers ---
        # Free space (< 0.45) adjacent to Unknown space (0.45-0.55)
        free_mask = map_probs < 0.45
        unknown_mask = (map_probs >= 0.45) & (map_probs <= 0.55)
        
        # Dilate free space to find edges touching unknown
        free_dilated = free_mask.copy()
        free_dilated[:-1, :] |= free_mask[1:, :] 
        free_dilated[1:, :] |= free_mask[:-1, :] 
        free_dilated[:, :-1] |= free_mask[:, 1:] 
        free_dilated[:, 1:] |= free_mask[:, :-1] 
        
        frontier_mask = unknown_mask & free_dilated
        
        # Remove frontiers that are inside the Inflated (Unsafe) zone.
        frontier_mask = frontier_mask & (~inflated_obstacles)
        
        # --- 4. Logic Update ---
        replan = False
        
        # Case A: No target yet
        if self.target_grid is None:
            replan = True
        else:
            # Case B: Target reached
            tx_world = self.target_grid[0] * map_resolution + map_origin
            ty_world = self.target_grid[1] * map_resolution + map_origin
            if np.hypot(tx_world - rx, ty_world - ry) < self.reached_threshold_m:
                replan = True
            
            # Case D: Path execution finished
            elif not self.current_path or self.path_index >= len(self.current_path):
                replan = True

        if replan:
            # Find the best "Zone" to explore
            self.target_grid = self._find_best_frontier_zone(frontier_mask, map_resolution)
            
            if self.target_grid:
                # Plan Path using A*
                self.current_path = self.planner.a_star(
                    start=(gx, gy), 
                    goal=self.target_grid, 
                    grid=binary_grid, 
                    resolution=map_resolution, 
                    origin=map_origin
                )
                self.path_index = 0
                self.reset_pid() # Reset PID for new path
            else:
                self.current_path = []
        
        # --- 5. Pure Pursuit Control with PID ---
        if not self.current_path:
            print("DEBUG: No path found, searching/spinning...")
            return 0.0, 3.0 # Spin slowly if stuck

        # Update path_index to be the closest point to the robot
        if self.path_index < len(self.current_path):
            search_len = 20 
            end_search = min(self.path_index + search_len, len(self.current_path))
            
            dists = [np.hypot(p[0]-rx, p[1]-ry) for p in self.current_path[self.path_index:end_search]]
            if dists:
                min_idx = np.argmin(dists)
                self.path_index += min_idx
            
        # Find lookahead point on the path
        target_point = self.current_path[-1] # Default to end
        
        # Search forward from current index
        for i in range(self.path_index, len(self.current_path)):
            px, py = self.current_path[i]
            dist = np.hypot(px - rx, py - ry)
            if dist > self.lookahead_dist:
                target_point = (px, py)
                break
                
        tx, ty = target_point
        dx = tx - rx
        dy = ty - ry
        
        # Calculate Heading Error
        target_angle = np.arctan2(dy, dx)
        angle_err = (target_angle - ryaw + np.pi) % (2 * np.pi) - np.pi
        
        # PID Calculation
        self.integral_angle_err += angle_err
        self.integral_angle_err = np.clip(self.integral_angle_err, -1.0, 1.0)
        
        derivative = angle_err - self.prev_angle_err
        self.prev_angle_err = angle_err
        
        angular_v = (self.kp_angular * angle_err) + \
                      (self.ki_angular * self.integral_angle_err) + \
                      (self.kd_angular * derivative)
        
        # Linear Velocity Logic
        if abs(angle_err) > np.pi / 2:
            # Pivot if error is large
            linear_v = 0.0
            if abs(angular_v) < 2.0:
                 angular_v = np.sign(angle_err) * 2.0
        else:
            # Smoothly interpolate speed
            linear_v = self.max_linear_speed * np.cos(angle_err)
            if linear_v < 0: linear_v = 0
        
        angular_v = np.clip(angular_v, -5.0, 5.0)

        # --- FIX: INVERTED STEERING SIGNS ---
        # Previous: left = lin - ang, right = lin + ang (Turning Right)
        # New: left = lin + ang, right = lin - ang (Turning Left)
        left = np.clip(linear_v + angular_v, -15, 15)
        right = np.clip(linear_v - angular_v, -15, 15)
        
        # --- DEBUG PRINT ---
        print(f"DEBUG: Pos({rx:.1f},{ry:.1f}) Tgt({tx:.1f},{ty:.1f}) ErrDeg:{np.degrees(angle_err):.1f} L/R:{left:.1f}/{right:.1f} Idx:{self.path_index}/{len(self.current_path)}")

        return left, right

    def _find_best_frontier_zone(self, frontier_mask, resolution):
        """
        Groups frontier pixels into coarse blocks and selects the best zone.
        """
        indices = np.argwhere(frontier_mask)
        if len(indices) == 0: return None
        
        # 1. Coarse Grid Clustering (1.0m blocks)
        block_size = int(1.0 / resolution) 
        if block_size < 1: block_size = 1
        
        block_coords = indices // block_size
        
        # 2. Find Block with most frontier pixels
        unique_blocks, counts = np.unique(block_coords, axis=0, return_counts=True)
        best_block_idx = np.argmax(counts)
        best_block = unique_blocks[best_block_idx]
        
        # 3. Select Median Target from that Block
        mask = (block_coords[:, 0] == best_block[0]) & (block_coords[:, 1] == best_block[1])
        zone_indices = indices[mask]
        
        center_idx = len(zone_indices) // 2
        target_yx = zone_indices[center_idx]
        
        return (target_yx[1], target_yx[0])