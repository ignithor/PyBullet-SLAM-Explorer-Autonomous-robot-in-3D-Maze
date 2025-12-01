import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt

# --- Custom Imports ---
from maze_generator import MazeGenerator, MAZE_SIZE, CELL_SIZE
from robot import Robot
from control_module import ControlModule
from mapping import Slam
from perception_module import DuckDetector  # <-- NEW IMPORT

# --- Physics Constants ---
WALL_RESTITUTION = 0.0
WALL_FRICTION = 0.8
SOLVER_ITERATIONS = 50
TIME_STEP = 1. / 240.0

# --- Mapping Constants ---
MAP_UPDATE_RATE = 50
MAP_SIZE_M = (MAZE_SIZE * CELL_SIZE) + 10.0


class SimulationManager:
    """
    Manages the simulation, robot, SLAM, and AI Perception.
    """

    def __init__(self):
        # 1. Initialize PyBullet
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(TIME_STEP)
        p.setPhysicsEngineParameter(numSolverIterations=SOLVER_ITERATIONS)

        # 2. Setup Environment
        self.maze_logic = MazeGenerator(MAZE_SIZE)
        self.maze_logic.generate()
        self._load_environment()
        
        # 3. Load Robot
        self._load_robot()

        # 4. Load Duck
        self._load_duck()

        # 5. Initialize Modules
        self.slam = Slam(map_size_m=MAP_SIZE_M, map_resolution=0.1)
        self.robot_path = []
        
        # Initialize AI Perception
        self.detector = DuckDetector()
        
        # 6. Visualization
        self.fig, self.ax, self.im, self.path_plot = self._setup_plot()
        self._setup_camera()

    def _load_environment(self):
        # Load Floor
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(plane_id, -1, restitution=WALL_RESTITUTION, lateralFriction=WALL_FRICTION)

        # Load Walls
        walls_to_build = self.maze_logic.get_walls_to_build()
        wall_color = [0.2, 0.2, 0.8, 1]

        for pos, half_extents, orientation in walls_to_build:
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=wall_color)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=pos, baseOrientation=orientation)

        print(f"INFO: Loaded {len(walls_to_build)} wall segments.")

    def _load_robot(self):
        robot_start_pos = [CELL_SIZE / 2, CELL_SIZE / 2, 0.1]
        self.robot = Robot(self.client, start_pos=robot_start_pos)

    def _load_duck(self):
        """
        Loads a yellow duck into the maze at a specific location.
        """
        # Place duck in cell (1, 0) - ensure this coordinate exists in your maze size
        duck_x = 1.5 * CELL_SIZE
        duck_y = 0.5 * CELL_SIZE
        duck_z = 0.3 # Slightly above ground
        
        # Use duck_vhacd.urdf from pybullet_data
        try:
            self.duck_id = p.loadURDF("duck_vhacd.urdf", 
                                      basePosition=[duck_x, duck_y, duck_z], 
                                      baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                                      globalScaling=8.0) # Scale up so it's visible
            print(f"INFO: Yellow Duck placed at [{duck_x}, {duck_y}]")
        except Exception:
            print("WARNING: Could not load duck_vhacd.urdf. Make sure pybullet_data is correct.")

    def _setup_camera(self):
        center_x = MAZE_SIZE * CELL_SIZE / 2
        center_y = MAZE_SIZE * CELL_SIZE / 2
        p.resetDebugVisualizerCamera(cameraDistance=MAZE_SIZE * CELL_SIZE * 0.8, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[center_x, center_y, 0])

    def _setup_plot(self):
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        initial_map = self.slam.get_map_probabilities()
        im = ax.imshow(initial_map, cmap='gray_r', origin='lower', vmin=0.0, vmax=1.0, 
                       extent=[self.slam.origin_m, self.slam.origin_m + self.slam.map_size_m, self.slam.origin_m, self.slam.origin_m + self.slam.map_size_m])
        path_plot, = ax.plot([], [], 'r-', linewidth=0.5)
        return fig, ax, im, path_plot

    def run_simulation(self):
        controller = ControlModule(self.robot)
        
        # Calculate steps for 5 seconds
        steps_per_5_sec = int(5.0 / TIME_STEP)
        
        try:
            step_count = 0
            while p.isConnected():
                
                # 1. Control & Sensing
                # lidar_data is an array, camera_packet is (width, height, rgb_pixels)
                lidar_data, camera_packet = controller.step()

                # Unpack the camera tuple to get the actual RGB array
                _, _, camera_rgb = camera_packet

                # 2. SLAM Update
                pos, quat = p.getBasePositionAndOrientation(self.robot.robot_id)
                yaw = p.getEulerFromQuaternion(quat)[2]
                self.slam.update([pos[0], pos[1]], yaw, lidar_data)
                self.robot_path.append([pos[0], pos[1]])

                # 3. AI Perception (Every 5 seconds)
                if step_count % steps_per_5_sec == 0:
                    print("\n--- Analysing Visual Scene ---")
                    # Pass only the RGB array to the detector
                    label, conf, all_probs = self.detector.detect(camera_rgb.astype(np.uint8))
                    
                    print(f"Prediction: {label.upper()} ({conf:.2f})")
                    if label == "a yellow duck" and conf > 0.6:
                        print(">>> DUCK DETECTED! <<<")
                
                # 4. Visualization Update
                if step_count % MAP_UPDATE_RATE == 0:
                    self.im.set_data(self.slam.get_map_probabilities())
                    path_arr = np.array(self.robot_path)
                    if len(path_arr) > 0:
                        self.path_plot.set_data(path_arr[:, 0], path_arr[:, 1])
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()

                p.stepSimulation()
                time.sleep(TIME_STEP)
                step_count += 1

        except p.error:
            pass
        except KeyboardInterrupt:
            pass

    def disconnect(self):
        if self.client is not None:
            p.disconnect(self.client)
        plt.close()

if __name__ == "__main__":
    sim = SimulationManager()
    try:
        sim.run_simulation()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sim.disconnect()