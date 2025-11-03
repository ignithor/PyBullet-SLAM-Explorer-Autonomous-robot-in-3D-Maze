from mazegenerator import MazeGenerator, MAZE_SIZE, CELL_SIZE
import pybullet as p
import pybullet_data
import numpy as np
import time

class SimulationManager:
    """
    Initializes the PyBullet environment and loads the 3D maze walls.
    """
    def __init__(self):
        # 1. Initialize PyBullet (p.GUI for visualization, p.DIRECT for headless)
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(numSolverIterations=10)

        self.robot_id = None
        
        # 2. Instantiate and generate the logical maze
        self.maze_logic = MazeGenerator(MAZE_SIZE)
        self.maze_logic.generate()
        
        self._load_environment()
        self._load_robot()
        self._setup_camera()
        
    def _load_environment(self):
        """Loads the floor and 3D walls into the PyBullet environment."""
        
        # Load the ground plane (floor)
        p.loadURDF("plane.urdf")
        
        # Create a single floor box to better represent the maze area 
        # (Optional: plane.urdf is usually enough)
        
        # Load the walls as static rigid bodies (mass=0)
        walls_to_build = self.maze_logic.get_walls_to_build()
        
        wall_color = [0.2, 0.2, 0.8, 1] # Blue walls
        
        for pos, half_extents, orientation in walls_to_build:
            # Create a Collision Shape (defines physical boundaries)
            collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, 
                                                        halfExtents=half_extents)
            # Create a Visual Shape (defines appearance)
            visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, 
                                                  halfExtents=half_extents,
                                                  rgbaColor=wall_color)
            
            # Create the MultiBody (the actual object in the simulation)
            p.createMultiBody(baseMass=0, 
                              baseCollisionShapeIndex=collision_shape_id,
                              baseVisualShapeIndex=visual_shape_id,
                              basePosition=pos,
                              baseOrientation=orientation)
        
        print(f"INFO: Loaded {len(walls_to_build)} wall segments into PyBullet.")
        
    def _load_robot(self):
        """Loads a simple robot (e.g., a small cube) for initial testing."""
        
        robot_start_pos = [CELL_SIZE/2, CELL_SIZE/2, 0.1] # Start at the center of the first cell (0,0)
        robot_half_extents = [0.1, 0.1, 0.1]
        
        # Robot Collision and Visual shape
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=robot_half_extents)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=robot_half_extents, rgbaColor=[1, 0, 0, 1])
        
        # Create the robot body (Mass > 0)
        self.robot_id = p.createMultiBody(baseMass=1, 
                                          baseCollisionShapeIndex=col_shape, 
                                          baseVisualShapeIndex=vis_shape, 
                                          basePosition=robot_start_pos)
        print(f"INFO: Loaded Robot with ID: {self.robot_id}")

    def _setup_camera(self):
        """Sets the camera to view the maze from above."""
        center_x = MAZE_SIZE * CELL_SIZE / 2
        center_y = MAZE_SIZE * CELL_SIZE / 2
        
        p.resetDebugVisualizerCamera(cameraDistance=MAZE_SIZE * CELL_SIZE * 0.8,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[center_x, center_y, 0])

    def run_simulation(self):
        """Main simulation loop (only for visualization in this phase)."""
        try:
            while p.isConnected():
                # In this phase, we just let the simulation run
                p.stepSimulation()
                time.sleep(1./240.) # 240 fps

        except p.error:
            # PyBullet disconnects when closing the window
            print("INFO: PyBullet simulation ended.")
            
    def disconnect(self):
        """Cleanly disconnects the PyBullet client."""
        p.disconnect()

# --- Main Execution ---
if __name__ == "__main__":
    sim = SimulationManager()
    
    # Simple test movement for the robot to verify physics
    print("INFO: Applying a small force to the robot for 2 seconds.")
    p.applyExternalForce(sim.robot_id, -1, [10, 0, 0], [0, 0, 0], p.WORLD_FRAME)
    
    # Run simulation for 2 seconds before entering the main loop
    for _ in range(2 * 240):
        p.stepSimulation()
        time.sleep(1./240.)

    sim.run_simulation()
    sim.disconnect()