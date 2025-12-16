import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt

from maze_generator import MazeGenerator, MAZE_SIZE, CELL_SIZE
from robot import Robot
from control_module import ControlModule
from mapping import Slam
from perception_module import DuckDetector
from exploration import FrontierExplorer

# --- Physics Constants ---
WALL_RESTITUTION = 0.0
WALL_FRICTION = 0.8
SOLVER_ITERATIONS = 50
TIME_STEP = 1. / 240.0

# --- Mapping Constants ---
MAP_UPDATE_RATE = 50
MAP_SIZE_M = (MAZE_SIZE * CELL_SIZE) + 10.0

# --- State Constants ---
STATE_EXPLORE = 0
STATE_STOP = 1
STATE_PLAN = 2
STATE_RETURN = 3
STATE_FINISHED = 4

class SimulationManager:
    def __init__(self):
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(TIME_STEP)
        p.setPhysicsEngineParameter(numSolverIterations=SOLVER_ITERATIONS)

        self.maze_logic = MazeGenerator(MAZE_SIZE)
        self.maze_logic.generate()
        self._load_environment()
        
        self.robot_start_pos = [CELL_SIZE / 2, CELL_SIZE / 2, 0.1]
        self._load_robot()
        
        # Load Objects
        self._load_duck()
        self._load_soccer_ball()
        self._load_teddy_bear()

        self.slam = Slam(map_size_m=MAP_SIZE_M, map_resolution=0.1)
        self.robot_path = []
        
        self.detector = DuckDetector()
        self.explorer = FrontierExplorer()
        
        self.current_state = STATE_EXPLORE
        
        self.fig, self.ax, self.im, self.path_plot, self.plan_plot, self.robot_marker = self._setup_plot()
        self._setup_camera()

    def _load_environment(self):
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(plane_id, -1, restitution=WALL_RESTITUTION, lateralFriction=WALL_FRICTION)

        walls_to_build = self.maze_logic.get_walls_to_build()
        wall_color = [0.2, 0.2, 0.8, 1]

        for pos, half_extents, orientation in walls_to_build:
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=wall_color)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=pos, baseOrientation=orientation)

        print(f"INFO: Loaded {len(walls_to_build)} wall segments.")

    def _load_robot(self):
        self.robot = Robot(self.client, start_pos=self.robot_start_pos)

    def _load_duck(self):
        # Place duck far enough to ensure exploration
        duck_x = 3.5 * CELL_SIZE
        duck_y = 2.5 * CELL_SIZE
        duck_z = 0.3 # Slightly above ground
        
        # Use duck_vhacd.urdf from pybullet_data
        try:
            self.duck_id = p.loadURDF("duck_vhacd.urdf", 
                                      basePosition=[duck_x, duck_y, duck_z], 
                                      baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                                      globalScaling=8.0)
            print(f"INFO: Yellow Duck placed at [{duck_x}, {duck_y}]")
        except Exception:
            print("WARNING: Could not load duck_vhacd.urdf.")

    def _load_soccer_ball(self):
        """Loads a soccer ball as a distractor object."""
        # Place soccer ball in a different location (e.g., cell 1, 2)
        ball_x = 1.5 * CELL_SIZE
        ball_y = 3.5 * CELL_SIZE
        ball_z = 0.5
        try:
            self.ball_id = p.loadURDF("soccerball.urdf", 
                                      basePosition=[ball_x, ball_y, ball_z], 
                                      globalScaling=0.5) # Slightly larger to be visible
            print(f"INFO: Soccer Ball placed at [{ball_x}, {ball_y}]")
        except Exception:
            print("WARNING: Could not load soccerball.urdf.")
            
    def _load_teddy_bear(self):
        """Loads a teddy bear as a third object."""
        # Place teddy bear in cell (2, 3)
        bear_x = 2.3 * CELL_SIZE
        bear_y = 0.5 * CELL_SIZE
        bear_z = 0.0
        try:
            self.bear_id = p.loadURDF("teddy_vhacd.urdf", 
                                      basePosition=[bear_x, bear_y, bear_z], 
                                      baseOrientation=p.getQuaternionFromEuler([1.57,0,0]),
                                      globalScaling=8.0) 
            print(f"INFO: Teddy Bear placed at [{bear_x}, {bear_y}]")
        except Exception:
            print("WARNING: Could not load teddy_vhacd.urdf.")
    
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
        
        path_plot, = ax.plot([], [], 'r-', linewidth=0.5, label='Traveled')
        plan_plot, = ax.plot([], [], 'b--', linewidth=1.0, label='Planned')
        robot_marker, = ax.plot([], [], 'go', markersize=8, label='Robot', markeredgecolor='black')
        
        ax.legend()
        return fig, ax, im, path_plot, plan_plot, robot_marker

    def run_simulation(self):
        steps_perception = 30
        
        try:
            step_count = 0
            while p.isConnected():
                
                # --- SENSORS & SLAM ---
                lidar_data = self.robot.get_lidar_data()
                width, height, camera_rgb = self.robot.get_camera_image()

                pos, quat = p.getBasePositionAndOrientation(self.robot.robot_id)
                yaw = p.getEulerFromQuaternion(quat)[2]
                self.slam.update([pos[0], pos[1]], yaw, lidar_data)
                self.robot_path.append([pos[0], pos[1]])

                map_probs = self.slam.get_map_probabilities()
                
                # --- FINITE STATE MACHINE ---
                left_vel, right_vel = 0, 0

                # 1. STATE EXPLORE
                if self.current_state == STATE_EXPLORE:
                    # Drive using frontier explorer
                    left_vel, right_vel = self.explorer.get_control_command(
                        robot_pose=(pos[0], pos[1], yaw),
                        map_probs=map_probs,
                        map_origin=self.slam.origin_m,
                        map_resolution=self.slam.resolution
                    )
                    
                    # Periodic Detection (Only in Explore mode)
                    if step_count % steps_perception == 0:
                        print("\n--- Analysing Visual Scene ---")
                        label, conf, _ = self.detector.detect(camera_rgb.astype(np.uint8))
                        print(f"Prediction: {label.upper()} ({conf:.2f})")
                        
                        if label == "a yellow duck" and conf > 0.6:
                            print(">>> DUCK DETECTED! INITIATING STOP & RETURN SEQUENCE <<<")
                            self.current_state = STATE_STOP
                        elif label == "a soccer ball" and conf > 0.6:
                            print(">>> SOCCER BALL DETECTED! (Ignoring, looking for duck...) <<<")
                        elif label == "a teddy bear" and conf > 0.6:
                            print(">>> TEDDY BEAR DETECTED! (Ignoring...) <<<")

                # 2. STATE STOP
                elif self.current_state == STATE_STOP:
                    left_vel, right_vel = 0, 0 # Hard Stop
                    time.sleep(1.0) 
                    print("INFO: Robot stopped. Calculating path home...")
                    self.current_state = STATE_PLAN

                # 3. STATE PLAN
                elif self.current_state == STATE_PLAN:
                    home_x, home_y = self.robot_start_pos[0], self.robot_start_pos[1]
                    self.explorer.set_return_target(home_x, home_y)
                    self.current_state = STATE_RETURN
                    print("INFO: Path planning complete. Returning to base.")

                # 4. STATE RETURN
                elif self.current_state == STATE_RETURN:
                    left_vel, right_vel = self.explorer.get_control_command(
                        robot_pose=(pos[0], pos[1], yaw),
                        map_probs=map_probs,
                        map_origin=self.slam.origin_m,
                        map_resolution=self.slam.resolution
                    )
                    if left_vel == 0 and right_vel == 0:
                        print("\n>>> MISSION ACCOMPLISHED: ROBOT RETURNED HOME <<<")
                        self.current_state = STATE_FINISHED

                # 5. STATE FINISHED
                elif self.current_state == STATE_FINISHED:
                    left_vel, right_vel = 0, 0

                # --- ACTUATION ---
                self.robot.set_velocity(left_vel, right_vel)

                # --- VISUALIZATION ---
                if step_count % MAP_UPDATE_RATE == 0:
                    self.im.set_data(self.slam.get_map_probabilities())
                    path_arr = np.array(self.robot_path)
                    if len(path_arr) > 0:
                        self.path_plot.set_data(path_arr[:, 0], path_arr[:, 1])
                    
                    if self.explorer.current_path:
                        plan_arr = np.array(self.explorer.current_path)
                        self.plan_plot.set_data(plan_arr[:, 0], plan_arr[:, 1])
                    else:
                        self.plan_plot.set_data([], [])
                    
                    self.robot_marker.set_data([pos[0]], [pos[1]])
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()

                p.stepSimulation()
                # time.sleep(TIME_STEP) 
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