import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt

from maze_generator import MazeGenerator
from robot import Robot
from mapping import Slam
from perception_module import DuckDetector
from exploration import FrontierExplorer
from ekf import EKF
from particle_filter import ParticleFilter
import config as cfg
from raw_odometry import RawOdometry


class SimulationManager:
    def __init__(self):
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(cfg.TIME_STEP)
        p.setPhysicsEngineParameter(numSolverIterations=cfg.SOLVER_ITERATIONS)

        self.maze_logic = MazeGenerator(cfg.MAZE_SIZE)
        self.maze_logic.generate()
        self._load_environment()
        
        self.robot_start_pos = [cfg.CELL_SIZE / 2, cfg.CELL_SIZE / 2, 0.1]
        self._load_robot()
        
        # Load Objects
        self._load_duck()
        self._load_soccer_ball()
        self._load_teddy_bear() 

        self.slam = Slam(map_size_m=cfg.MAP_SIZE_M, map_resolution=0.1)
        self.robot_path = []
        
        self.detector = DuckDetector()
        self.explorer = FrontierExplorer()
        
        # --- INITIALIZE ESTIMATOR ---
        # Get initial ground truth just for seeding the filter
        _, quat = p.getBasePositionAndOrientation(self.robot.robot_id)
        start_yaw = p.getEulerFromQuaternion(quat)[2]
        
        if cfg.USE_PARTICLE_FILTER:
            print("INFO: Initializing Particle Filter...")
            self.estimator = ParticleFilter(self.robot_start_pos, start_yaw, cfg.TIME_STEP, num_particles=100)
        elif cfg.USE_EKF:
            print("INFO: Initializing Extended Kalman Filter (EKF)...")
            self.estimator = EKF(self.robot_start_pos, start_yaw, cfg.TIME_STEP)
        else:
            print("INFO: Initializing Raw Odometry...")
            self.estimator = RawOdometry(self.robot_start_pos, start_yaw, cfg.TIME_STEP)
        
        self.current_state = cfg.STATE_EXPLORE
        
        # Track found object location
        self.found_duck_pos = None
        
        # Updated to catch the extra ground truth marker
        self.fig, self.ax, self.im, self.path_plot, self.plan_plot, self.robot_marker, self.duck_marker, self.gt_marker = self._setup_plot()
        self._setup_camera()

    def _load_environment(self):
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(plane_id, -1, restitution=cfg.WALL_RESTITUTION, lateralFriction=cfg.WALL_FRICTION)

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
        try:
            self.duck_id = p.loadURDF(cfg.DUCK_URDF_PATH, 
                                      basePosition=cfg.DUCK_POSITION, 
                                      baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                                      globalScaling=8.0)
            print(f"INFO: Yellow Duck placed at [{cfg.DUCK_POSITION[0]}, {cfg.DUCK_POSITION[1]}]")
        except Exception:
            print("WARNING: Could not load duck_vhacd.urdf.")

    def _load_soccer_ball(self):
        """Loads a soccer ball as a distractor object."""
        try:
            self.ball_id = p.loadURDF(cfg.SOCCERBALL_URDF_PATH, 
                                      basePosition=cfg.SOCCER_BALL_POSITION, 
                                      globalScaling=0.5) # Slightly larger to be visible
            print(f"INFO: Soccer Ball placed at [{cfg.SOCCER_BALL_POSITION[0]}, {cfg.SOCCER_BALL_POSITION[1]}]")
        except Exception:
            print("WARNING: Could not load soccerball.urdf.")

    def _load_teddy_bear(self):
        """Loads a teddy bear as a third object."""
        try:
            self.bear_id = p.loadURDF(cfg.TEDDY_URDF_PATH, 
                                      basePosition=cfg.TEDDY_BEAR_POSITION, 
                                      baseOrientation=p.getQuaternionFromEuler([1.57,0,0]),
                                      globalScaling=8.0) 
            print(f"INFO: Teddy Bear placed at [{cfg.TEDDY_BEAR_POSITION[0]}, {cfg.TEDDY_BEAR_POSITION[1]}]")
        except Exception:
            print("WARNING: Could not load teddy_vhacd.urdf.")

    def _setup_camera(self):
        center_x = cfg.MAZE_SIZE * cfg.CELL_SIZE / 2
        center_y = cfg.MAZE_SIZE * cfg.CELL_SIZE / 2
        p.resetDebugVisualizerCamera(cameraDistance=cfg.MAZE_SIZE * cfg.CELL_SIZE * 0.8, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[center_x, center_y, 0])

    def _setup_plot(self):
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        initial_map = self.slam.get_map_probabilities()
        im = ax.imshow(initial_map, cmap='gray_r', origin='lower', vmin=0.0, vmax=1.0, 
                       extent=[self.slam.origin_m, self.slam.origin_m + self.slam.map_size_m, self.slam.origin_m, self.slam.origin_m + self.slam.map_size_m])
        
        path_plot, = ax.plot([], [], 'r-', linewidth=0.5, label='Traveled')
        plan_plot, = ax.plot([], [], 'b--', linewidth=1.0, label='Planned')
        
        # Estimator Estimate (Green Circle)
        if cfg.USE_PARTICLE_FILTER:
            label_text = 'PF Est.' 
        elif cfg.USE_EKF:
            label_text = 'EKF Est.'
        else:
            label_text = 'Est.'
        robot_marker, = ax.plot([], [], 'go', markersize=8, label=label_text, markeredgecolor='black')
        
        # Ground Truth (Red Cross)
        gt_marker, = ax.plot([], [], 'rx', markersize=8, label='Ground Truth', markeredgewidth=2)
        
        duck_marker, = ax.plot([], [], 'y*', markersize=15, label='Duck', markeredgecolor='black')
        
        ax.legend(loc='upper right')
        return fig, ax, im, path_plot, plan_plot, robot_marker, duck_marker, gt_marker

    def run_simulation(self):
        steps_perception = 30
        
        try:
            step_count = 0
            while p.isConnected():
                
                # --- SENSORS ---
                lidar_data = self.robot.get_lidar_data()
                width, height, camera_rgb = self.robot.get_camera_image()
                
                # --- ESTIMATOR: PREDICTION (ODOMETRY) ---              
                wl, wr = self.robot.get_wheel_velocity()
                v_wheel = (cfg.WHEEL_RADIUS / 2.0) * (wl + wr)
                omega_wheel = (cfg.WHEEL_RADIUS / cfg.TRACK_WIDTH) * (wr - wl)
                
                # SIMULATE IMU (Velocimeter): Check actual body velocity
                lin_vel, ang_vel = p.getBaseVelocity(self.robot.robot_id)
                actual_v = np.linalg.norm(lin_vel[:2]) # Linear speed magnitude
                
                # STUCK DETECTION LOGIC:
                is_stuck = (abs(v_wheel) > 0.1) and (actual_v < 0.02)
                
                if is_stuck:
                    # print("DEBUG: Slip detected! Zeroing Estimator input.")
                    v_input = 0.0
                    omega_input = 0.0 
                else:
                    v_input = v_wheel
                    omega_input = omega_wheel

                if cfg.USE_PARTICLE_FILTER or cfg.USE_EKF:
                    self.estimator.predict(v_input, omega_input)
                    # --- ESTIMATOR: CORRECTION (COMPASS) ---
                    true_yaw = self.robot.get_compass_reading()
                    self.estimator.update_compass(true_yaw)
                else:
                    self.estimator.update(v_input, omega_input)
                
                
                # --- GET ESTIMATED POSE ---
                # Handle API differences between EKF and PF
                if cfg.USE_PARTICLE_FILTER:
                    est_pose = self.estimator.get_estimate()
                elif cfg.USE_EKF:
                    est_pose = self.estimator.get_pose()
                else:
                    est_pose = self.estimator.get_pose()
                
                est_x, est_y, est_yaw = est_pose

                # --- GET GROUND TRUTH POSE (For Visualization Only) ---
                gt_pos, gt_quat = p.getBasePositionAndOrientation(self.robot.robot_id)
                gt_x, gt_y = gt_pos[0], gt_pos[1]
                gt_yaw = p.getEulerFromQuaternion(gt_quat)[2]
                
                # Select which one to use for SLAM and Control
                if cfg.USE_PERFECT_POSE:
                    current_x, current_y, current_yaw = gt_x, gt_y, gt_yaw
                else:
                    current_x, current_y, current_yaw = est_x, est_y, est_yaw

                # --- SLAM & MAPPING using ESTIMATED POSE ---
                self.slam.update([current_x, current_y], current_yaw, lidar_data)
                self.robot_path.append([current_x, current_y])

                map_probs = self.slam.get_map_probabilities()
                
                # --- FINITE STATE MACHINE ---
                left_vel, right_vel = 0, 0

                # 1. STATE EXPLORE
                if self.current_state == cfg.STATE_EXPLORE:
                    left_vel, right_vel = self.explorer.get_control_command(
                        robot_pose=(current_x, current_y, current_yaw), 
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
                            self.current_state = cfg.STATE_STOP
                            
                            # Estimate Duck Location based on LiDAR
                            # Ray 0 is Center/Front. Check rays +/- 30 deg (Indices 0-3 and 33-35 for 36 rays)
                            num_rays = len(lidar_data) # Should be 36
                            front_indices = [0, 1, 2, 3, num_rays-3, num_rays-2, num_rays-1]
                            
                            # Find the closest obstacle in the camera's FOV
                            min_dist = float('inf')
                            closest_ray_idx = -1
                            
                            for idx in front_indices:
                                if lidar_data[idx] < 9.5: 
                                    if lidar_data[idx] < min_dist:
                                        min_dist = lidar_data[idx]
                                        closest_ray_idx = idx
                            
                            if closest_ray_idx != -1:
                                angle_step = 2 * np.pi / num_rays
                                ray_angle = closest_ray_idx * angle_step
                                dx = est_x + min_dist * np.cos(est_yaw + ray_angle)
                                dy = est_y + min_dist * np.sin(est_yaw + ray_angle)
                                self.found_duck_pos = (dx, dy)
                                print(f"DEBUG: Duck location estimated at [{dx:.2f}, {dy:.2f}] based on LiDAR.")
                            else:
                                # Fallback if LiDAR didn't catch it (unlikely if camera did)
                                print("DEBUG: Camera saw duck but LiDAR didn't. Using fallback.")
                                dx = current_x + 0.8 * np.cos(current_yaw)
                                dy = current_y + 0.8 * np.sin(current_yaw)
                                self.found_duck_pos = (dx, dy)
                        
                        elif label == "a soccer ball" and conf > 0.6:
                            print(">>> SOCCER BALL DETECTED! (Ignoring...) <<<")                                    
                        elif label == "a teddy bear" and conf > 0.6:
                            print(">>> TEDDY BEAR DETECTED! (Ignoring...) <<<")

                # 2. STATE STOP
                elif self.current_state == cfg.STATE_STOP:
                    left_vel, right_vel = 0, 0 # Hard Stop
                    time.sleep(1.0) 
                    print("INFO: Robot stopped. Calculating path home...")
                    self.current_state = cfg.STATE_PLAN
                # 3. STATE PLAN
                elif self.current_state == cfg.STATE_PLAN:
                    home_x, home_y = self.robot_start_pos[0], self.robot_start_pos[1]
                    self.explorer.set_return_target(home_x, home_y)
                    self.current_state = cfg.STATE_RETURN
                    print("INFO: Path planning complete. Returning to base.")

                # 4. STATE RETURN
                elif self.current_state == cfg.STATE_RETURN:
                    left_vel, right_vel = self.explorer.get_control_command(
                        robot_pose=(current_x, current_y, current_yaw), 
                        map_probs=map_probs,
                        map_origin=self.slam.origin_m,
                        map_resolution=self.slam.resolution
                    )
                    # Use a small threshold to detect if returned
                    if current_x < 1.6 and current_y < 1.6:
                        print("\n>>> MISSION ACCOMPLISHED: ROBOT RETURNED HOME <<<")
                        self.current_state = cfg.STATE_FINISHED

                # 5. STATE FINISHED
                elif self.current_state == cfg.STATE_FINISHED:
                    left_vel, right_vel = 0, 0

                # --- ACTUATION ---
                self.robot.set_velocity(left_vel, right_vel)

                # --- VISUALIZATION ---
                if step_count % cfg.MAP_UPDATE_RATE == 0:
                    self.im.set_data(self.slam.get_map_probabilities())
                    path_arr = np.array(self.robot_path)
                    if len(path_arr) > 0:
                        self.path_plot.set_data(path_arr[:, 0], path_arr[:, 1])
                    
                    if self.explorer.current_path:
                        plan_arr = np.array(self.explorer.current_path)
                        self.plan_plot.set_data(plan_arr[:, 0], plan_arr[:, 1])
                    else:
                        self.plan_plot.set_data([], [])
                    
                    # Update Estimates
                    self.robot_marker.set_data([current_x], [current_y])
                    # Update Ground Truth
                    self.gt_marker.set_data([gt_x], [gt_y])
                    
                    if self.found_duck_pos:
                        self.duck_marker.set_data([self.found_duck_pos[0]], [self.found_duck_pos[1]])
                    
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