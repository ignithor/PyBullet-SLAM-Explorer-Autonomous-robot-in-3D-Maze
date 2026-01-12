import os
import numpy as np

# --- Simulation Settings ---
GUI_MODE = True
TIME_STEP = 1. / 240.0
SOLVER_ITERATIONS = 50
WALL_RESTITUTION = 0.0
WALL_FRICTION = 0.8
WALL_THICKNESS = 0.1
WALL_HEIGHT = 1.0


# --- Maze Generation ---
MAZE_SIZE = 5 
CELL_SIZE = 3.0 

# --- Robot Physical Properties ---
# Paths
ROBOT_URDF_PATH = os.path.join(os.getcwd(), "urdf", "simple_two_wheel_car.urdf")
DUCK_URDF_PATH = "duck_vhacd.urdf"
SOCCERBALL_URDF_PATH = "soccerball.urdf"
TEDDY_URDF_PATH = "teddy_vhacd.urdf"

# Joints & Motors
LEFT_WHEEL_JOINT_INDEX = 0
RIGHT_WHEEL_JOINT_INDEX = 1
MAX_MOTOR_FORCE = 150.0

# Dimensions (Estimates for Kinematics)
WHEEL_RADIUS = 0.05
TRACK_WIDTH = 0.3
ROBOT_RADIUS_M = 1.0 # For path planning safety

# --- Sensors ---
# LiDAR
LIDAR_RAYS = 36
LIDAR_RANGE = 10.0
LIDAR_Z = 0.25
LIDAR_START_OFFSET = 0.28

# Camera
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FOV = 60
CAMERA_ASPECT = CAMERA_WIDTH / CAMERA_HEIGHT
CAMERA_NEAR = 0.1
CAMERA_FAR = 10.0
CAMERA_X_OFFSET = 0.0
CAMERA_Z_OFFSET = 0.3

# --- SLAM / Mapping ---
MAP_RESOLUTION = 0.1
MAP_SIZE_M = (MAZE_SIZE * CELL_SIZE) + 10.0
MAP_ORIGIN_OFFSET = -5.0
MAP_UPDATE_RATE = 50 # Visual update frequency

# Occupancy Grid Probabilities
LOG_ODDS_HIT = 3.5
LOG_ODDS_FREE = 0.05
LOG_ODDS_CLAMP = 100.0

# --- Localization (EKF/PF) ---
# Set to True to use perfect PyBullet coordinates (Debugging).
USE_PERFECT_POSE = True
# Set to True to use Particle Filter, False for EKF
USE_PARTICLE_FILTER = True

# EKF Settings
EKF_PROCESS_NOISE = [0.05, 0.05, 0.1] # x, y, theta std dev
EKF_MEASURE_NOISE = [0.01] # compass std dev

# PF Settings
PF_NUM_PARTICLES = 100
PF_ODOM_NOISE = [0.02, 0.02, 0.05]
PF_COMPASS_NOISE = 0.05

# --- Exploration / Control ---
MAX_LINEAR_SPEED = 25.0
LOOKAHEAD_DIST = 1.0
REACHED_THRESHOLD_M = 0.4
SAFETY_MARGIN_CELLS = 2

# Stuck Recovery
STUCK_CHECK_INTERVAL = 50
STUCK_DIST_THRESHOLD = 0.1
RECOVERY_DURATION = 40

# PID Gains
KP_ANGULAR = 6.0
KI_ANGULAR = 0.01
KD_ANGULAR = 1.0

# --- Perception ---
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DETECTION_LABELS = ["a yellow duck", "a soccer ball", "a teddy bear", "a wall", "an empty floor", "a robot"]
DETECTION_CONFIDENCE_THRESHOLD = 0.6
PERCEPTION_INTERVAL_STEPS = int(5.0 / TIME_STEP)

# --- State Machine States ---
STATE_EXPLORE = 0
STATE_STOP = 1
STATE_PLAN = 2
STATE_RETURN = 3
STATE_FINISHED = 4

# --- Objects positions ---
DUCK_POSITION = [1.5 * CELL_SIZE, 2.5 * CELL_SIZE, 0.3]
SOCCER_BALL_POSITION = [1.5 * CELL_SIZE, 3.5 * CELL_SIZE, 0.5]
TEDDY_BEAR_POSITION = [2.3 * CELL_SIZE, 0.5 * CELL_SIZE, 0.0]