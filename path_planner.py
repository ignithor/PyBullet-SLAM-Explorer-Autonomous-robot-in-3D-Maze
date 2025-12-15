import numpy as np
import heapq

class PathPlanner:
    """
    Implements the A* (A-Star) pathfinding algorithm on a grid.
    """
    def __init__(self):
        # 8-connected movement (diagonal allowed)
        # Format: (dx, dy, cost)
        self.motions = [
            (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414)
        ]

    def a_star(self, start, goal, grid, resolution, origin):
        """
        Computes the shortest path avoiding obstacles.
        
        Args:
            start: (grid_x, grid_y)
            goal: (grid_x, grid_y)
            grid: 2D numpy array where 1 = Obstacle/Unsafe, 0 = Free/Safe.
            resolution: Map resolution (m/cell)
            origin: Map origin (m)
            
        Returns:
            List of (world_x, world_y) path points.
        """
        height, width = grid.shape
        
        # 1. Bounds Check
        if not (0 <= start[0] < width and 0 <= start[1] < height): return None
        if not (0 <= goal[0] < width and 0 <= goal[1] < height): return None
        
        # 2. Safety Check
        # If start is inside an inflated obstacle (e.g., drift), search for nearest safe cell
        if grid[start[1], start[0]] == 1:
            start = self._find_nearest_free(start, grid)
            if start is None: return None

        # If goal is inside an obstacle (e.g., map update closed the frontier), find nearest safe cell
        if grid[goal[1], goal[0]] == 1:
            goal = self._find_nearest_free(goal, grid)
            if goal is None: return None

        # 3. A* Algorithm
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        max_iterations = 20000 # Safety break for large maps
        iterations = 0
        
        while open_set:
            iterations += 1
            if iterations > max_iterations: return None
            
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                return self._reconstruct_path(came_from, current, resolution, origin)
            
            cx, cy = current
            for dx, dy, cost in self.motions:
                neighbor = (cx + dx, cy + dy)
                
                # Check Bounds
                if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                    continue
                
                # Check Collision (Grid 1 = Obstacle)
                if grid[neighbor[1], neighbor[0]] == 1:
                    continue
                
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))
                    
        return None # No path found

    def _heuristic(self, a, b):
        # Euclidean distance heuristic
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def _reconstruct_path(self, came_from, current, resolution, origin):
        path_indices = [current]
        while current in came_from:
            current = came_from[current]
            path_indices.append(current)
        
        path_indices.reverse()
        
        # Convert to world coordinates
        world_path = []
        for (gx, gy) in path_indices:
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
            
            for dx, dy, _ in self.motions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return None