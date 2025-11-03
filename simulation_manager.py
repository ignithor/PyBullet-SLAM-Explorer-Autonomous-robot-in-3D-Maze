from maze_generator import MazeGenerator, MAZE_SIZE, CELL_SIZE
from robot import Robot, WHEEL_RADIUS, WHEEL_AXLE_LENGTH # Import the Robot class and constants
import pybullet as p
import pybullet_data
import numpy as np
import time

class SimulationManager:
    """
    Initializes the PyBullet environment, loads the 3D maze walls, 
    and manages the robot's lifecycle and the simulation loop.
    """
    def __init__(self):
        # 1. Initialize PyBullet (p.GUI for visualization, p.DIRECT for headless)
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(numSolverIterations=10)

        # Instance of the Robot class, not just the ID
        self.robot = None 
        
        # 2. Instantiate and generate the logical maze
        self.maze_logic = MazeGenerator(MAZE_SIZE)
        self.maze_logic.generate()
        
        self._load_environment()
        self._load_robot()
        self._setup_camera()
        
    def _load_environment(self):
        """Loads the floor and 3D walls into the PyBullet environment."""
        
        # Load the ground plane (floor)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        
        # Load the walls as static rigid bodies (mass=0)
        walls_to_build = self.maze_logic.get_walls_to_build()
        
        wall_color = [0.2, 0.2, 0.8, 1] # Blue walls
        
        for pos, half_extents, orientation in walls_to_build:
            # Create a Collision Shape (defines physical boundaries)
            collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, 
                                                        halfExtents=half_extents,
                                                        physicsClientId=self.client)
            # Create a Visual Shape (defines appearance)
            visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, 
                                                  halfExtents=half_extents,
                                                  rgbaColor=wall_color,
                                                  physicsClientId=self.client)
            
            # Create the MultiBody (the actual object in the simulation)
            p.createMultiBody(baseMass=0, 
                              baseCollisionShapeIndex=collision_shape_id,
                              baseVisualShapeIndex=visual_shape_id,
                              basePosition=pos,
                              baseOrientation=orientation,
                              physicsClientId=self.client)
        
        print(f"INFO: Loaded {len(walls_to_build)} wall segments into PyBullet.")
        
    def _load_robot(self):
        """Initializes the Robot class instance and loads its model."""
        
        # Determine start position based on maze cell size
        robot_start_pos = [CELL_SIZE/2, CELL_SIZE/2, 0.1] 
        
        # Instantiate the Robot class. It handles the actual PyBullet loading.
        self.robot = Robot(self.client, start_pos=robot_start_pos)
        
        # The robot's PyBullet ID is now stored within the self.robot object (self.robot.robot_id)
        print(f"INFO: Robot loaded at position {robot_start_pos}.")

    def _setup_camera(self):
        """Sets the camera to view the maze from above."""
        center_x = MAZE_SIZE * CELL_SIZE / 2
        center_y = MAZE_SIZE * CELL_SIZE / 2
        
        p.resetDebugVisualizerCamera(cameraDistance=MAZE_SIZE * CELL_SIZE * 0.8,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[center_x, center_y, 0],
                                     physicsClientId=self.client)

    def run_simulation(self):
        """
        Main simulation loop. 
        Demonstrates basic robot control and sensor data acquisition.
        """
        try:
            # Simple constant movement for testing: move forward and turn slightly
            left_speed = 5.0
            right_speed = 6.0
            
            # Run simulation for a fixed duration to observe movement
            duration_steps = 10 * 240 # 10 seconds at 240 FPS
            step_count = 0
            
            while p.isConnected() and step_count < duration_steps:
                
                # --- Control ---
                self.robot.set_velocity(left_speed, right_speed)

                # --- Perception (Placeholder for next phases) ---
                lidar_ranges = self.robot.get_lidar_data()
                camera_rgb = self.robot.get_camera_image()
                
                # print(f"Lidar Min Range: {np.min(lidar_ranges):.2f}") # Uncomment to debug Lidar
                
                # --- Step Physics ---
                p.stepSimulation(physicsClientId=self.client)
                time.sleep(1./240.) 
                step_count += 1

        except p.error:
            # PyBullet disconnects when closing the window
            print("INFO: PyBullet simulation ended.")
            
    def disconnect(self):
        """Cleanly disconnects the PyBullet client."""
        p.disconnect(self.client)