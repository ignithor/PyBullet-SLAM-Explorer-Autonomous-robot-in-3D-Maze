from maze_generator import MazeGenerator, MAZE_SIZE, CELL_SIZE
from robot import Robot  # Import the Robot class and constants
import pybullet as p
import pybullet_data
import numpy as np
import time
from control_module import ControlModule  # <-- 1. IMPORTED CONTROL MODULE

# --- Physics Constants ---
WALL_RESTITUTION = 0.0
WALL_FRICTION = 0.8
SOLVER_ITERATIONS = 50
TIME_STEP = 1. / 480.0


class SimulationManager:
    """
    Initializes the PyBullet environment, loads the 3D maze walls,
    and manages the robot's lifecycle and the simulation loop.
    """

    def __init__(self):
        # 1. Initialize PyBullet
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Set stable physics parameters
        p.setTimeStep(TIME_STEP)
        p.setPhysicsEngineParameter(numSolverIterations=SOLVER_ITERATIONS)

        self.robot = None

        # 2. Instantiate and generate the logical maze
        self.maze_logic = MazeGenerator(MAZE_SIZE)
        self.maze_logic.generate()

        self._load_environment()
        self._load_robot()
        self._setup_camera() # Sets the initial camera view

    def _load_environment(self):
        """Loads the floor and 3D walls into the PyBullet environment."""

        # Load the ground plane (floor) and apply dynamics fix
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(plane_id, -1, restitution=WALL_RESTITUTION, lateralFriction=WALL_FRICTION,
                         physicsClientId=self.client)

        # Load the walls
        walls_to_build = self.maze_logic.get_walls_to_build()
        wall_color = [0.2, 0.2, 0.8, 1]  # Blue walls

        for pos, half_extents, orientation in walls_to_build:
            collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                        halfExtents=half_extents,
                                                        physicsClientId=self.client)
            visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                  halfExtents=half_extents,
                                                  rgbaColor=wall_color,
                                                  physicsClientId=self.client)
            
            wall_body_id = p.createMultiBody(baseMass=0,
                                              baseCollisionShapeIndex=collision_shape_id,
                                              baseVisualShapeIndex=visual_shape_id,
                                              basePosition=pos,
                                              baseOrientation=orientation,
                                              physicsClientId=self.client)

            # Apply dynamics fix to walls
            p.changeDynamics(wall_body_id,
                             linkIndex=-1,
                             restitution=WALL_RESTITUTION,
                             lateralFriction=WALL_FRICTION,
                             physicsClientId=self.client)

        print(f"INFO: Loaded {len(walls_to_build)} wall segments into PyBullet.")

    def _load_robot(self):
        """Initializes the Robot class instance and loads its model."""
        robot_start_pos = [CELL_SIZE / 2, CELL_SIZE / 2, 0.1]
        self.robot = Robot(self.client, start_pos=robot_start_pos)
        print(f"INFO: Robot loaded at position {robot_start_pos}.")

    def _setup_camera(self):
        """
        Sets the camera to your original working view (45-degree angle).
        """
        center_x = MAZE_SIZE * CELL_SIZE / 2
        center_y = MAZE_SIZE * CELL_SIZE / 2

        # --- CAMERA REVERTED TO YOUR SETTINGS ---
        p.resetDebugVisualizerCamera(cameraDistance=MAZE_SIZE * CELL_SIZE * 0.8,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[center_x, center_y, 0],
                                     physicsClientId=self.client)

    def run_simulation(self):
        """
        Main simulation loop.
        Drives the robot using ControlModule.
        """
        
        # --- 2. Initialize the Control Module ---
        exploration_controller = ControlModule(self.robot)

        try:
            # Run for 300 seconds (5 minutes)
            duration_steps = 300 * int(1.0 / TIME_STEP)
            step_count = 0

            while p.isConnected() and step_count < duration_steps:
                
                # --- 3. LET THE CONTROLLER DRIVE THE ROBOT ---
                # The step() method reads sensors and sets velocity
                exploration_controller.step()

                # --- Step Physics ---
                p.stepSimulation(physicsClientId=self.client)
                time.sleep(TIME_STEP)
                step_count += 1

        except p.error:
            print("INFO: PyBullet simulation ended.")

    def disconnect(self):
        """Cleanly disconnects the PyBullet client."""
        if self.client is not None:
            p.disconnect(self.client)

# --- Main Execution (to allow running this file directly) ---
if __name__ == "__main__":
    print("--- Starting 3D Maze Robot Simulation ---")
    sim = None
    try:
        sim = SimulationManager()
        sim.run_simulation()

    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        
    finally:
        if sim is not None:
            sim.disconnect()
            print("--- Simulation Ended and Disconnected ---")