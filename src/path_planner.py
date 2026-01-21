import numpy as np
import heapq
import config as cfg

class PathPlanner:
    """
    Implements a Directional A* (A-Star) pathfinding algorithm.
    It includes orientation in the state space (x, y, orientation_index)
    to enforce smooth transitions and prevent instant U-turns.
    """
    def __init__(self):
        # 8 discrete directions (0=East, 1=NE, 2=North, ..., 7=SE)
        # (dx, dy, step_cost)
        self.actions = [
            (1, 0, 1.0),   # 0: East
            (1, 1, 1.414), # 1: NE
            (0, 1, 1.0),   # 2: North
            (-1, 1, 1.414),# 3: NW
            (-1, 0, 1.0),  # 4: West
            (-1, -1, 1.414),# 5: SW
            (0, -1, 1.0),  # 6: South
            (1, -1, 1.414) # 7: SE
        ]

    def _yaw_to_idx(self, yaw):
        """Converts continuous yaw to nearest 0-7 index."""
        # Normalize to 0..2pi
        yaw = yaw % (2 * np.pi)
        # 8 sectors of 45 deg (pi/4). 
        # Add pi/8 (22.5 deg) shift so 0 covers -22.5 to +22.5
        idx = int(((yaw + (np.pi / 8)) % (2 * np.pi)) / (np.pi / 4))
        return idx % 8

    def a_star(self, start, goal, grid, resolution, origin, start_yaw=0.0):
        """
        Computes a smooth path considering robot orientation.
        
        Args:
            start: (grid_x, grid_y)
            goal: (grid_x, grid_y)
            grid: 2D numpy array (1=Obstacle, 0=Safe)
            resolution: meters/cell
            origin: meters
            start_yaw: Robot's current heading in radians
            
        Returns:
            List of (world_x, world_y)
        """
        height, width = grid.shape
        
        # 1. Bounds & Collision Check
        if not (0 <= start[0] < width and 0 <= start[1] < height): return None
        if not (0 <= goal[0] < width and 0 <= goal[1] < height): return None
        
        if grid[start[1], start[0]] == 1: 
            start = self._find_nearest_free(start, grid)
            if start is None: return None
        if grid[goal[1], goal[0]] == 1:
            goal = self._find_nearest_free(goal, grid)
            if goal is None: return None

        # 2. Setup Search
        # State: (x, y, dir_idx)
        start_dir = self._yaw_to_idx(start_yaw)
        start_state = (start[0], start[1], start_dir)
        
        open_set = []
        heapq.heappush(open_set, (0, start_state))
        
        came_from = {} # Maps current_state -> parent_state
        g_score = {start_state: 0}
        
        max_iterations = cfg.PATH_PLANNER_MAX_ITERATIONS 
        iterations = 0
        
        final_state = None
        
        while open_set:
            iterations += 1
            if iterations > max_iterations: return None
            
            current_cost, current_state = heapq.heappop(open_set)
            cx, cy, c_dir = current_state
            
            # Goal Check (ignore final direction)
            if (cx, cy) == goal:
                final_state = current_state
                break
            
            # Expand Neighbors
            # We restrict turns to +/- 45 degrees (delta index -1, 0, 1)
            # This forces smoothness. To turn 180, it must take 4 steps.
            for i in [-1, 0, 1]:
                next_dir = (c_dir + i) % 8
                dx, dy, step_cost = self.actions[next_dir]
                
                nx, ny = cx + dx, cy + dy
                
                # Check Bounds
                if not (0 <= nx < width and 0 <= ny < height): continue
                # Check Collision
                if grid[ny, nx] == 1: continue
                
                # Calculate Cost
                # Base movement cost + Turn Penalty
                # Turning costs extra to encourage straight lines and reduce wobbling
                turn_penalty = 0.0 if i == 0 else 2.5 
                new_g = g_score[current_state] + step_cost + turn_penalty
                
                next_state = (nx, ny, next_dir)
                
                if next_state not in g_score or new_g < g_score[next_state]:
                    g_score[next_state] = new_g
                    priority = new_g + self._heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (priority, next_state))
                    came_from[next_state] = current_state
                    
        if final_state:
            return self._reconstruct_path(came_from, final_state, resolution, origin)
        return None

    def _heuristic(self, a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def _reconstruct_path(self, came_from, current, resolution, origin):
        # Backtrack
        path_indices = [current]
        while current in came_from:
            current = came_from[current]
            path_indices.append(current)
        path_indices.reverse()
        
        # Convert to World Coords
        # We drop the direction index here as the controller just needs points
        world_path = []
        for (gx, gy, _) in path_indices:
            wx = gx * resolution + origin
            wy = gy * resolution + origin
            world_path.append((wx, wy))
        return world_path

    def _find_nearest_free(self, center, grid):
        """BFS to find the closest 0-value cell."""
        queue = [center]
        visited = {center}
        height, width = grid.shape
        max_search = 500
        
        while queue and len(visited) < max_search:
            cx, cy = queue.pop(0)
            if grid[cy, cx] == 0:
                return (cx, cy)
            
            # Search neighbors
            for dx, dy, _ in self.actions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return None