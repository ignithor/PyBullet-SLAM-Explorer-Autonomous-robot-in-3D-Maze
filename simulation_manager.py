import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt

# --- Custom Imports ---
from maze_generator import MazeGenerator, MAZE_SIZE, CELL_SIZE
from robot import Robot
from control_module import ControlModule  # Assuming control.py was renamed to control_module.py
from mapping import Slam                  # Assuming slam.py was renamed to mapping.py

# --- Physics Constants ---
WALL_RESTITUTION = 0.0
WALL_FRICTION = 0.8
SOLVER_ITERATIONS = 50
TIME_STEP = 1. / 240.0  # Standard PyBullet step

# --- Mapping Constants ---
MAP_UPDATE_RATE = 50  # Update plot every 50 steps
# Define map size based on maze size + margin
MAP_SIZE_M = (MAZE_SIZE * CELL_SIZE) + 10.0


class SimulationManager:
    """
    Initializes the PyBullet environment, loads the 3D maze walls,
    manages the robot's lifecycle, runs the control loop, and performs SLAM.
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

        # 3. Load Environment and Robot
        self._load_environment()
        self._load_robot()

        # 4. Initialize SLAM
        # We use the calculated MAP_SIZE_M to ensure the map fits the maze
        self.slam = Slam(map_size_m=MAP_SIZE_M, map_resolution=0.1)
        self.robot_path = [] # Trace for visualization
        print("INFO: SLAM module initialized.")

        # 5. Initialize Visualization (Matplotlib)
        self.fig, self.ax, self.im, self.path_plot = self._setup_plot()

        # 6. Setup Camera (Your specific 45-degree view)
        self._setup_camera()

    def _load_environment(self):
        """Loads the floor and 3D walls into the PyBullet environment."""
        # Load the ground plane
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(plane_id, -1, restitution=WALL_RESTITUTION, lateralFriction=WALL_FRICTION,
                         physicsClientId=self.client)

        # Load the walls
        walls_to_build = self.maze_logic.get_walls_to_build()
        wall_color = [0.2, 0.2, 0.8, 1]  # Blue walls

        for pos, half_extents, orientation in walls_to_build:
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client)
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=wall_color, physicsClientId=self.client)
            
            wall_id = p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=col_id,
                                        baseVisualShapeIndex=vis_id,
                                        basePosition=pos,
                                        baseOrientation=orientation,
                                        physicsClientId=self.client)

            p.changeDynamics(wall_id, -1, restitution=WALL_RESTITUTION, lateralFriction=WALL_FRICTION, physicsClientId=self.client)

        print(f"INFO: Loaded {len(walls_to_build)} wall segments into PyBullet.")

    def _load_robot(self):
        """Initializes the Robot class instance and loads its model."""
        # Start in the middle of the first cell
        robot_start_pos = [CELL_SIZE / 2, CELL_SIZE / 2, 0.1]
        self.robot = Robot(self.client, start_pos=robot_start_pos)
        print(f"INFO: Robot loaded at position {robot_start_pos}.")

    def _setup_camera(self):
        """
        Sets the camera to your original working view (45-degree angle).
        """
        center_x = MAZE_SIZE * CELL_SIZE / 2
        center_y = MAZE_SIZE * CELL_SIZE / 2

        p.resetDebugVisualizerCamera(cameraDistance=MAZE_SIZE * CELL_SIZE * 0.8,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[center_x, center_y, 0],
                                     physicsClientId=self.client)

    def _setup_plot(self):
        """
        Sets up the Matplotlib real-time visualization.
        """
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Get initial map
        initial_map = self.slam.get_map_probabilities()
        
        # Setup the image plot
        im = ax.imshow(
            initial_map,
            cmap='gray_r',
            origin='lower',
            vmin=0.0,
            vmax=1.0,
            extent=[
                self.slam.origin_m,
                self.slam.origin_m + self.slam.map_size_m,
                self.slam.origin_m,
                self.slam.origin_m + self.slam.map_size_m
            ]
        )
        
        ax.set_title("SLAM Occupancy Grid")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        path_plot, = ax.plot([], [], 'r-', linewidth=0.5, label='Path')
        ax.legend(loc='upper right')
        
        return fig, ax, im, path_plot

    def run_simulation(self):
        """
        Main simulation loop.
        Drives the robot using ControlModule and performs SLAM.
        """
        
        # --- Initialize Controller ---
        controller = ControlModule(self.robot)

        try:
            # Run for 300 seconds (5 minutes)
            duration_steps = 300 * int(1.0 / TIME_STEP)
            step_count = 0

            print("INFO: Starting Simulation Loop...")

            while p.isConnected() and step_count < duration_steps:
                
                # --- 1. Control & Sensing ---
                # controller.step() moves the robot AND returns sensor data
                lidar_data, camera_data = controller.step()

                # --- 2. Get Ground Truth Pose for SLAM ---
                # (In real SLAM, we would estimate this, but here we use the true pose)
                pos, quat = p.getBasePositionAndOrientation(self.robot.robot_id, physicsClientId=self.client)
                yaw = p.getEulerFromQuaternion(quat)[2]
                robot_pos_xy = [pos[0], pos[1]]

                # --- 3. Update SLAM ---
                self.slam.update(robot_pos_xy, yaw, lidar_data)
                self.robot_path.append(robot_pos_xy)

                # --- 4. Update Visualization (Throttled) ---
                if step_count % MAP_UPDATE_RATE == 0:
                    prob_map = self.slam.get_map_probabilities()
                    self.im.set_data(prob_map)
                    
                    # Update path
                    path_arr = np.array(self.robot_path)
                    if len(path_arr) > 0:
                        self.path_plot.set_data(path_arr[:, 0], path_arr[:, 1])
                    
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()

                # --- 5. Step Physics ---
                p.stepSimulation()
                time.sleep(TIME_STEP)
                step_count += 1

        except p.error:
            print("INFO: PyBullet simulation ended.")
        except KeyboardInterrupt:
            print("INFO: Simulation stopped by user.")

    def disconnect(self):
        """Cleanly disconnects the PyBullet client."""
        if self.client is not None:
            p.disconnect(self.client)
        plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting 3D Maze Robot Simulation ---")
    sim = None
    try:
        sim = SimulationManager()
        sim.run_simulation()

    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if sim is not None:
            sim.disconnect()
            print("--- Simulation Ended and Disconnected ---")