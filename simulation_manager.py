from maze_generator import MazeGenerator, MAZE_SIZE, CELL_SIZE
from robot import Robot  # Import the Robot class and constants
import pybullet as p
import pybullet_data
import numpy as np
import time

# --- Physics Constants ---
# Set restitution (bounciness) to zero and lateral friction high for stable contact
WALL_RESTITUTION = 0.0
WALL_FRICTION = 0.8
SOLVER_ITERATIONS = 50  # Increased from 10 for better stability
TIME_STEP = 1. / 480.0  # Increased frequency (480 Hz) for smoother collisions


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

        # --- FIX 1: Enhanced Solver Parameters ---
        p.setTimeStep(TIME_STEP)
        p.setPhysicsEngineParameter(numSolverIterations=SOLVER_ITERATIONS)
        # ------------------------------------------

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

        # Load the ground plane (floor) and apply dynamics fix
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(plane_id, -1, restitution=WALL_RESTITUTION, lateralFriction=WALL_FRICTION,
                         physicsClientId=self.client)

        # Load the walls as static rigid bodies (mass=0)
        walls_to_build = self.maze_logic.get_walls_to_build()

        wall_color = [0.2, 0.2, 0.8, 1]  # Blue walls

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
            wall_body_id = p.createMultiBody(baseMass=0,
                                             baseCollisionShapeIndex=collision_shape_id,
                                             baseVisualShapeIndex=visual_shape_id,
                                             basePosition=pos,
                                             baseOrientation=orientation,
                                             physicsClientId=self.client)

            # --- FIX 2: Set Wall Collision Dynamics ---
            p.changeDynamics(wall_body_id,
                             linkIndex=-1,  # Target the base link
                             restitution=WALL_RESTITUTION,
                             lateralFriction=WALL_FRICTION,
                             physicsClientId=self.client)
            # ------------------------------------------

        print(f"INFO: Loaded {len(walls_to_build)} wall segments into PyBullet.")

    def _load_robot(self):
        """Initializes the Robot class instance and loads its model."""

        # Determine start position based on maze cell size
        robot_start_pos = [CELL_SIZE / 2, CELL_SIZE / 2, 0.1]

        # Instantiate the Robot class. It handles the actual PyBullet loading
        # and should handle the robot's dynamics settings (base and wheels) internally.
        self.robot = Robot(self.client, start_pos=robot_start_pos)

        print(f"INFO: Robot loaded at position {robot_start_pos}.")

    def _setup_camera(self):
        """Sets the camera to view the maze from above."""
        # ... (unchanged camera code, but ensure it's a stable view) ...
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
        """
        try:
            left_speed = 5.0
            right_speed = 6.0
            duration_steps = 10 * int(1.0 / TIME_STEP)  # Use the new TIME_STEP for calculation
            step_count = 0

            while p.isConnected() and step_count < duration_steps:
                # --- Control ---
                self.robot.set_velocity(left_speed, right_speed)

                # --- Perception (Placeholder for next phases) ---
                lidar_ranges = self.robot.get_lidar_data()
                camera_rgb = self.robot.get_camera_image()

                # --- Step Physics ---
                p.stepSimulation(physicsClientId=self.client)
                time.sleep(TIME_STEP)  # Use the new TIME_STEP here
                step_count += 1

        except p.error:
            print("INFO: PyBullet simulation ended.")

    def disconnect(self):
        """Cleanly disconnects the PyBullet client."""
        p.disconnect(self.client)