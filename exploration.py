import numpy as np
from path_planner import PathPlanner

class FrontierExplorer:
    """
    Advanced Frontier Exploration Module.
    Features:
    - Obstacle Inflation (C-Space): Inflates walls by robot radius to prevent collisions.
    - Frontier Clustering: Targets largest 'zones' of unknown space.
    - A* Path Planning: Computes feasible paths in the safe configuration space.
    - PID Control: Smooth path following with adaptive lookahead.
    """
    def __init__(self):
        self.target_grid = None 
        self.reached_threshold_m = 0.4 # Reduced threshold for precision
        
        # Path Planner
        self.planner = PathPlanner()
        self.current_path = [] 
        self.path_index = 0
        
        # PID Control parameters
        self.max_linear_speed = 50.0   # Reduced max speed for better control
        
        # PID Gains (Tighter tuning)
        self.kp_angular = 7.0   
        self.ki_angular = 0.0  
        self.kd_angular = 4.0   # High D term to dampen oscillation quickly
        
        # PID State
        self.prev_angle_err = 0.0
        self.integral_angle_err = 0.0
        
        # --- ROBOT SIZE CONFIGURATION ---
        # Increased slightly to keep robot safer from walls.
        # Robot center will stay at least 0.45m from walls.
        self.robot_radius_m = 0.45  

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
        
        # --- 2. Build Safe Navigation Map (C-Space Inflation) ---
        raw_obstacles = map_probs > 0.6
        
        # Calculate dynamic margin
        margin_cells = int(np.ceil(self.robot_radius_m / map_resolution))
        
        # Dilate obstacles
        inflated_obstacles = raw_obstacles.copy()
        for dx in range(-margin_cells, margin_cells + 1):
            for dy in range(-margin_cells, margin_cells + 1):
                if dx**2 + dy**2 <= margin_cells**2:
                    if dx == 0 and dy == 0: continue
                    shifted = np.roll(raw_obstacles, shift=(dy, dx), axis=(0, 1))
                    inflated_obstacles |= shifted
        
        # Create Binary Grid for A*
        binary_grid = np.zeros_like(map_probs, dtype=np.int8)
        binary_grid[inflated_obstacles] = 1 
        
        # --- 3. Identify Frontiers ---
        free_mask = map_probs < 0.45
        unknown_mask = (map_probs >= 0.45) & (map_probs <= 0.55)
        
        free_dilated = free_mask.copy()
        free_dilated[:-1, :] |= free_mask[1:, :] 
        free_dilated[1:, :] |= free_mask[:-1, :] 
        free_dilated[:, :-1] |= free_mask[:, 1:] 
        free_dilated[:, 1:] |= free_mask[:, :-1] 
        
        frontier_mask = unknown_mask & free_dilated
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
            
            # Case C: Auto-update path
            else:
                tx, ty = self.target_grid
                if not frontier_mask[ty, tx]:
                    replan = True
                elif not self.current_path or self.path_index >= len(self.current_path):
                    replan = True

        if replan:
            self.target_grid = self._find_best_frontier_zone(frontier_mask, map_resolution)
            if self.target_grid:
                self.current_path = self.planner.a_star(
                    start=(gx, gy), 
                    goal=self.target_grid, 
                    grid=binary_grid, 
                    resolution=map_resolution, 
                    origin=map_origin
                )
                self.path_index = 0
                self.reset_pid()
            else:
                self.current_path = []
        
        # --- 5. Adaptive Pure Pursuit Control ---
        if not self.current_path:
            print("DEBUG: No path found, searching...")
            return 0.0, 3.0 # Spin slowly

        # Find closest point on path (Robust tracking)
        if self.path_index < len(self.current_path):
            search_len = 20 
            end_search = min(self.path_index + search_len, len(self.current_path))
            dists = [np.hypot(p[0]-rx, p[1]-ry) for p in self.current_path[self.path_index:end_search]]
            if dists:
                min_idx = np.argmin(dists)
                self.path_index += min_idx
            
        # Determine Lookahead Distance dynamically
        # If we are close to the path, look further ahead for speed.
        # If we are off-path or tracking a curve, look closer for precision.
        current_error = 0
        if len(dists) > 0: current_error = dists[0]
        
        adaptive_lookahead = 0.6 # Base lookahead (closer than before)
        if current_error > 0.3:
            adaptive_lookahead = 0.4 # Tighten lookahead to get back on track
            
        # Find lookahead point
        target_point = self.current_path[-1]
        for i in range(self.path_index, len(self.current_path)):
            px, py = self.current_path[i]
            dist = np.hypot(px - rx, py - ry)
            if dist > adaptive_lookahead:
                target_point = (px, py)
                break
                
        tx, ty = target_point
        dx = tx - rx
        dy = ty - ry
        
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
        
        # Precise Velocity Logic
        # Slow down aggressively if angle error is high
        if abs(angle_err) > np.deg2rad(45):
            linear_v = 0.0 # Pivot in place
            # Boost angular if needed to overcome friction
            if abs(angular_v) < 3.0: 
                angular_v = np.sign(angle_err) * 3.0
        else:
            # Cosine profile for smooth slowing
            linear_v = self.max_linear_speed * np.cos(angle_err * 2.0) # *2 makes it drop to 0 at 45deg
            if linear_v < 0: linear_v = 0
        
        angular_v = np.clip(angular_v, -6.0, 6.0)

        # Steering Mixing (Preserving the "Working" sign convention)
        left = np.clip(linear_v + angular_v, -15, 15)
        right = np.clip(linear_v - angular_v, -15, 15)
        
        print(f"DEBUG: Pos({rx:.1f},{ry:.1f}) Tgt({tx:.1f},{ty:.1f}) Err:{np.degrees(angle_err):.1f} L/R:{left:.1f}/{right:.1f} Pth:{len(self.current_path)}")

        return left, right

    def _find_best_frontier_zone(self, frontier_mask, resolution):
        indices = np.argwhere(frontier_mask)
        if len(indices) == 0: return None
        
        block_size = int(1.0 / resolution) 
        if block_size < 1: block_size = 1
        
        block_coords = indices // block_size
        unique_blocks, counts = np.unique(block_coords, axis=0, return_counts=True)
        best_block_idx = np.argmax(counts)
        best_block = unique_blocks[best_block_idx]
        
        mask = (block_coords[:, 0] == best_block[0]) & (block_coords[:, 1] == best_block[1])
        zone_indices = indices[mask]
        
        center_idx = len(zone_indices) // 2
        target_yx = zone_indices[center_idx]
        
        return (target_yx[1], target_yx[0])