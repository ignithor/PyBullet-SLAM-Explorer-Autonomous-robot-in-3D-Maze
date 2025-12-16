import numpy as np
from path_planner import PathPlanner

class FrontierExplorer:
    """
    Advanced Navigation Module.
    Modes:
    1. Exploration: Finds frontiers and explores.
    2. Return: Navigates to a specific fixed target (Home).
    """
    def __init__(self):
        # --- Initialize Critical Variables First ---
        self.lookahead_dist = 1.0     
        self.target_grid = None 
        self.reached_threshold_m = 0.4
        
        # Path Planner
        self.planner = PathPlanner()
        self.current_path = [] 
        self.path_index = 0
        
        # PID Control parameters
        self.max_linear_speed = 30.0   # Reduced max speed for better control
        
        # PID Gains (Tighter tuning)
        self.kp_angular = 15.0   
        self.ki_angular = 0.0  
        self.kd_angular = 4.0   # High D term to dampen oscillation quickly
        
        # PID State
        self.prev_angle_err = 0.0
        self.integral_angle_err = 0.0
        
        # Safety
        self.safety_margin_cells = 2
        self.robot_radius_m = 0.45

        # MODE: 'EXPLORE' or 'RETURN'
        self.mode = 'EXPLORE'
        self.return_target_world = None

    def set_return_target(self, x, y):
        """Switches to RETURN mode and sets the fixed goal."""
        self.mode = 'RETURN'
        self.return_target_world = (x, y)
        self.current_path = [] # Clear old path
        self.path_index = 0
        self.target_grid = None
        print(f"DEBUG: Switched to RETURN mode. Target: {self.return_target_world}")

    def reset_pid(self):
        self.prev_angle_err = 0.0
        self.integral_angle_err = 0.0

    def get_control_command(self, robot_pose, map_probs, map_origin, map_resolution):
        rx, ry, ryaw = robot_pose
        gx = int((rx - map_origin) / map_resolution)
        gy = int((ry - map_origin) / map_resolution)
        
        # Ensure lookahead_dist exists (Safety check for reloading issues)
        if not hasattr(self, 'lookahead_dist'):
            self.lookahead_dist = 1.0

        # --- 1. Build Safe Map ---
        raw_obstacles = map_probs > 0.6
        margin_cells = int(np.ceil(self.robot_radius_m / map_resolution))
        
        inflated_obstacles = raw_obstacles.copy()
        for dx in range(-margin_cells, margin_cells + 1):
            for dy in range(-margin_cells, margin_cells + 1):
                if dx**2 + dy**2 <= margin_cells**2:
                    if dx == 0 and dy == 0: continue
                    shifted = np.roll(raw_obstacles, shift=(dy, dx), axis=(0, 1))
                    inflated_obstacles |= shifted
        
        binary_grid = np.zeros_like(map_probs, dtype=np.int8)
        binary_grid[inflated_obstacles] = 1 

        # --- 2. Determine Strategy based on Mode ---
        replan = False
        
        # === RETURN MODE LOGIC ===
        if self.mode == 'RETURN':
            if self.return_target_world:
                # Convert world target to grid
                tgx = int((self.return_target_world[0] - map_origin) / map_resolution)
                tgy = int((self.return_target_world[1] - map_origin) / map_resolution)
                
                # Check if reached
                dist_to_home = np.hypot(self.return_target_world[0] - rx, self.return_target_world[1] - ry)
                if dist_to_home < self.reached_threshold_m:
                    print("DEBUG: Returned to Start!")
                    return 0.0, 0.0 # Stop
                
                # Plan if no path
                if not self.current_path:
                    print("DEBUG: Planning path Home...")
                    self.target_grid = (tgx, tgy)
                    replan = True
                # Replan if stuck (path finished but not home)
                elif self.path_index >= len(self.current_path):
                    replan = True
        
        # === EXPLORE MODE LOGIC ===
        else:
            # Identify Frontiers
            free_mask = map_probs < 0.45
            unknown_mask = (map_probs >= 0.45) & (map_probs <= 0.55)
            free_dilated = free_mask.copy()
            free_dilated[:-1, :] |= free_mask[1:, :] 
            free_dilated[1:, :] |= free_mask[:-1, :] 
            free_dilated[:, :-1] |= free_mask[:, 1:] 
            free_dilated[:, 1:] |= free_mask[:, :-1] 
            frontier_mask = unknown_mask & free_dilated
            frontier_mask = frontier_mask & (~inflated_obstacles)

            if self.target_grid is None:
                replan = True
            else:
                tx_world = self.target_grid[0] * map_resolution + map_origin
                ty_world = self.target_grid[1] * map_resolution + map_origin
                if np.hypot(tx_world - rx, ty_world - ry) < self.reached_threshold_m:
                    replan = True
                else:
                    tx, ty = self.target_grid
                    # Check bounds before accessing array
                    h, w = frontier_mask.shape
                    if 0 <= ty < h and 0 <= tx < w:
                        if not frontier_mask[ty, tx]:
                            replan = True
                    else:
                        replan = True

                    if not replan and (not self.current_path or self.path_index >= len(self.current_path)):
                        replan = True
            
            if replan:
                self.target_grid = self._find_best_frontier_zone(frontier_mask, map_resolution)

        # --- 3. Execute Planning ---
        if replan and self.target_grid:
            self.current_path = self.planner.a_star(
                start=(gx, gy), 
                goal=self.target_grid, 
                grid=binary_grid, 
                resolution=map_resolution, 
                origin=map_origin
            )
            self.path_index = 0
            self.reset_pid()
            
            if not self.current_path:
                print("DEBUG: A* failed to find path.")
        
        # --- 4. Pure Pursuit Control ---
        if not self.current_path:
            return 0.0, 3.0 # Spin to find path

        if self.path_index < len(self.current_path):
            search_len = 20 
            end_search = min(self.path_index + search_len, len(self.current_path))
            dists = [np.hypot(p[0]-rx, p[1]-ry) for p in self.current_path[self.path_index:end_search]]
            if dists:
                self.path_index += np.argmin(dists)
            
        target_point = self.current_path[-1]
        for i in range(self.path_index, len(self.current_path)):
            px, py = self.current_path[i]
            if np.hypot(px - rx, py - ry) > self.lookahead_dist:
                target_point = (px, py)
                break
                
        tx, ty = target_point
        dx = tx - rx
        dy = ty - ry
        
        target_angle = np.arctan2(dy, dx)
        angle_err = (target_angle - ryaw + np.pi) % (2 * np.pi) - np.pi
        
        self.integral_angle_err += angle_err
        self.integral_angle_err = np.clip(self.integral_angle_err, -1.0, 1.0)
        derivative = angle_err - self.prev_angle_err
        self.prev_angle_err = angle_err
        
        angular_v = (self.kp_angular * angle_err) + (self.ki_angular * self.integral_angle_err) + (self.kd_angular * derivative)
        
        if abs(angle_err) > np.pi / 2:
            linear_v = 0.0
            if abs(angular_v) < 2.0: angular_v = np.sign(angle_err) * 2.0
        else:
            linear_v = self.max_linear_speed * np.cos(angle_err)
            if linear_v < 0: linear_v = 0
        
        angular_v = np.clip(angular_v, -5.0, 5.0)

        left = np.clip(linear_v + angular_v, -30, 30)
        right = np.clip(linear_v - angular_v, -30, 30)
        
        return left, right

    def _find_best_frontier_zone(self, frontier_mask, resolution):
        indices = np.argwhere(frontier_mask)
        if len(indices) == 0: return None
        block_size = max(1, int(1.0 / resolution))
        block_coords = indices // block_size
        unique_blocks, counts = np.unique(block_coords, axis=0, return_counts=True)
        best_block = unique_blocks[np.argmax(counts)]
        mask = (block_coords[:, 0] == best_block[0]) & (block_coords[:, 1] == best_block[1])
        zone_indices = indices[mask]
        return tuple(zone_indices[len(zone_indices) // 2][::-1])