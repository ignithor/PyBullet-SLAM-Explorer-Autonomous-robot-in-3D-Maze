import pybullet as p
import numpy as np
import random

# --- Configuration Constants ---
MAZE_SIZE = 15  # Grid size
WALL_HEIGHT = 1.0  # Height of the 3D walls
WALL_THICKNESS = 0.1  # Thickness of the walls
CELL_SIZE = 1.0  # Width of a single corridor/cell

class MazeGenerator:
    """
    Generates the logical 2D maze structure using Recursive Backtracking.
    Each cell stores which walls are present (N, S, E, W).
    """
    def __init__(self, size):
        self.size = size
        # Maze grid: 1 means the wall is present, 0 means it's a path
        # [0]=N, [1]=S, [2]=E, [3]=W
        self.maze = np.ones((size, size, 4), dtype=int)
        self.visited = np.zeros((size, size), dtype=bool)

    def _get_unvisited_neighbors(self, r, c):
        """Returns a list of unvisited neighbors and the wall index to remove."""
        neighbors = []
        directions = [
            (-1, 0, 0, 1),  # North: remove wall 0 (current) and 1 (neighbor)
            (1, 0, 1, 0),   # South: remove wall 1 (current) and 0 (neighbor)
            (0, 1, 2, 3),   # East: remove wall 2 (current) and 3 (neighbor)
            (0, -1, 3, 2)   # West: remove wall 3 (current) and 2 (neighbor)
        ]

        for (dr, dc, cur_wall, nei_wall) in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and not self.visited[nr, nc]:
                neighbors.append(((nr, nc), cur_wall, nei_wall))
        return neighbors

    def generate(self, r=0, c=0):
        """Recursive Backtracking algorithm to generate a perfect maze."""
        self.visited[r, c] = True
        
        neighbors = self._get_unvisited_neighbors(r, c)
        random.shuffle(neighbors)

        for (nr, nc), cur_wall, nei_wall in neighbors:
            if not self.visited[nr, nc]:
                # 1. Knock down the wall between the current cell and the neighbor
                self.maze[r, c, cur_wall] = 0
                self.maze[nr, nc, nei_wall] = 0
                
                # 2. Recurse into the neighbor cell
                self.generate(nr, nc)

    def get_walls_to_build(self):
        """
        Translates the 2D grid into a list of 3D wall properties (center position, dimensions, orientation),
        ensuring a closed perimeter with designated start/end openings.
        """
        walls = []
        half_t = WALL_THICKNESS / 2.0
        
        # --- Internal Walls ---
        for r in range(self.size):
            for c in range(self.size):
                center_x = c * CELL_SIZE + CELL_SIZE / 2
                center_y = r * CELL_SIZE + CELL_SIZE / 2
                
                # East Wall (Horizontal separation between cell c and c+1)
                # Only check for East wall, as West wall of the next cell (c+1) is identical
                if c < self.size - 1 and self.maze[r, c, 2] == 1:
                    pos_x = c * CELL_SIZE + CELL_SIZE + half_t
                    pos_y = center_y
                    half_extents = [half_t, CELL_SIZE / 2.0, WALL_HEIGHT / 2.0]
                    walls.append(( [pos_x, pos_y, WALL_HEIGHT / 2.0], half_extents, [0, 0, 0, 1] ))
                    
                # North Wall (Vertical separation between cell r and r-1)
                # Only check for North wall, as South wall of the cell above (r-1) is identical
                if r > 0 and self.maze[r, c, 0] == 1: 
                    pos_x = center_x
                    pos_y = r * CELL_SIZE - half_t
                    half_extents = [CELL_SIZE / 2.0, half_t, WALL_HEIGHT / 2.0]
                    walls.append(( [pos_x, pos_y, WALL_HEIGHT / 2.0], half_extents, [0, 0, 0, 1] ))

        # --- Perimeter Walls (Ensuring a closed boundary) ---
        
        # 1. Outer North Boundary (Top of the grid, r = size - 1)
        for c in range(self.size):
            pos_x = c * CELL_SIZE + CELL_SIZE / 2
            pos_y = self.size * CELL_SIZE + half_t
            half_extents = [CELL_SIZE / 2.0 + half_t, half_t, WALL_HEIGHT / 2.0]
            walls.append(( [pos_x, pos_y, WALL_HEIGHT / 2.0], half_extents, [0, 0, 0, 1] ))
            
        # 2. Outer South Boundary (Bottom of the grid, r = 0)
        for c in range(self.size):
            pos_x = c * CELL_SIZE + CELL_SIZE / 2
            pos_y = -half_t
            half_extents = [CELL_SIZE / 2.0 + half_t, half_t, WALL_HEIGHT / 2.0]
            walls.append(( [pos_x, pos_y, WALL_HEIGHT / 2.0], half_extents, [0, 0, 0, 1] ))
            
        # 3. Outer East Boundary (Right side of the grid, c = size - 1)
        for r in range(self.size):
            pos_x = self.size * CELL_SIZE + half_t
            pos_y = r * CELL_SIZE + CELL_SIZE / 2
            half_extents = [half_t, CELL_SIZE / 2.0 + half_t, WALL_HEIGHT / 2.0]
            walls.append(( [pos_x, pos_y, WALL_HEIGHT / 2.0], half_extents, [0, 0, 0, 1] ))
            
        # 4. Outer West Boundary (Left side of the grid, c = 0)
        for r in range(self.size):
            pos_x = -half_t
            pos_y = r * CELL_SIZE + CELL_SIZE / 2
            half_extents = [half_t, CELL_SIZE / 2.0 + half_t, WALL_HEIGHT / 2.0]
            walls.append(( [pos_x, pos_y, WALL_HEIGHT / 2.0], half_extents, [0, 0, 0, 1] ))
            
        # # --- Handle Start and End Openings ---
        
        # # Start Opening (Robot starts at (0, 0) - Remove segment of the South/West wall)
        # # Let's open the wall segment at (0, 0) on the South side.
        # # Find the wall segment corresponding to the South boundary of cell (0, 0)
        # start_wall_x = CELL_SIZE / 2
        # start_wall_y = -half_t 
        
        # # Remove the wall at the start position
        # walls = [w for w in walls if not (np.allclose(w[0][:2], [start_wall_x, start_wall_y]) and w[1][1] == half_t)]
        
        # # End Opening (Target is typically at (size-1, size-1) - Remove segment of the North/East wall)
        # # Let's open the wall segment at (size-1, size-1) on the North side.
        # end_wall_x = (self.size - 1) * CELL_SIZE + CELL_SIZE / 2
        # end_wall_y = self.size * CELL_SIZE + half_t
        
        # # Remove the wall at the end position
        # walls = [w for w in walls if not (np.allclose(w[0][:2], [end_wall_x, end_wall_y]) and w[1][1] == half_t)]
        
        return walls