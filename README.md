# Active Semantic SLAM: Autonomous 3D Maze Explorer

A modular autonomous navigation stack for mobile robots in unknown environments, built from scratch in Python using the PyBullet physics engine.

This project simulates a differential-drive robot navigating a procedurally generated 3D maze. The robot starts with no prior map, autonomously explores to build an occupancy grid, detects semantic targets via a vision-language model (CLIP), and plans a kinematically feasible path to return to its starting point.

---

## Key Features

### Autonomous Navigation Stack

- **Active Exploration**
  - Frontier-based exploration strategy
  - Nearest-frontier heuristic to maximize information gain

- **Directional Path Planning**
  - Custom Directional A* (Hybrid A* Lite) in state space \((x, y, theta)\)
  - Generates smooth, drivable “U-curves”
  - Respects non-holonomic constraints of a differential-drive robot

- **Adaptive Control**
  - Pure Pursuit controller for path following
  - PID angular velocity controller
  - Dynamic lookahead for stable tracking at higher speeds

### Probabilistic Localization (Hybrid)

- **Particle Filter (MCL)**
  - Monte Carlo Localization with ~100 particles
  - Robust to non-Gaussian errors (e.g., collisions)

- **Extended Kalman Filter (EKF)**
  - Fuses wheel odometry with PF estimate
  - Produces a smooth, low-noise pose for control

- **Fault Tolerance**
  - Simulated IMU-based stuck detection
  - Blind recovery maneuvers when wedged against obstacles

### Semantic Perception

- **Zero-Shot Detection (CLIP)**
  - Vision-language model for natural-language prompts  
    Examples: `"a yellow duck"`, `"a soccer ball"`, `"a teddy bear"`

- **Sensor Fusion**
  - 2D camera detections fused with LiDAR rays
  - Estimates 3D metric coordinates of detected objects on the occupancy map

### Mapping & Simulation

- **LiDAR-based SLAM**
  - Custom occupancy grid mapping with log-odds updates
  - Tuned “sticky walls” parameters to reduce map flicker at long range

- **Procedural Environments**
  - Perfect mazes (no loops) of arbitrary size
  - Generated with Recursive Backtracking

---

## Installation & Dependencies

### Requirements

- Python **3.8+**
- Recommended: virtual environment (e.g. `venv`, `conda`)

### 1. Clone the Repository

```bash
git clone https://github.com/ignithor/PyBullet-SLAM-Explorer-Autonomous-robot-in-3D-Maze
cd PyBullet-SLAM-Explorer-Autonomous-robot-in-3D-Maze
```

### 2. Install Required Packages

```bash
# Core physics & math
pip install pybullet numpy matplotlib scipy

# Computer vision & AI
pip install opencv-contrib-python torch torchvision transformers pillow
```

(You may need a compatible version of PyTorch for your OS/CUDA setup; refer to the official PyTorch installation guide if necessary.)

---

## How to Run

Start the full autonomous mission:

```bash
python src/main.py
```

---

## What You Will See

### PyBullet Window (3D Simulation)

- Robot model in a 3D maze
- Maze walls (blue)
- Objects (Duck, Ball, Bear) placed in the environment

### Matplotlib Window (Mapping View)

- Live **occupancy grid map**
- **Green circle**: Estimated robot pose (PF or EKF)
- **Red cross**: Ground-truth robot pose
- **Yellow star**: Detected target location
- **Blue line**: Current planned path

---

## Mission Logic (Finite State Machine)

The robot autonomously transitions through:

1. **EXPLORE**  
   - Frontier-based exploration to map unknown space.

2. **STOP & IDENTIFY**  
   - Robot stops when CLIP confirms the target (e.g., yellow duck).

3. **PLAN**  
   - Computes the shortest A* path back to the origin \((0, 0)\).

4. **RETURN**  
   - Follows the planned path to return home.

---

## Configuration (`config.py`)

All hyperparameters are centralized in `config.py`. Common options:

- `USE_EKF` or `USE_PF`
  - `True`: Use Particle Filter or EKF
  - `False`: Use raw odometry (to visualize drift)

- `USE_PERFECT_POSE`
  - `True`: Use “God mode” ground-truth pose (useful for debugging logic)
  - `False`: Use estimated pose

- `ROBOT_RADIUS_M`
  - Safety margin for the planner (increases distance from walls)

- `MAX_LINEAR_SPEED`
  - Maximum forward speed of the robot

- `MAP_UPDATE_RATE`
  - Frequency of map plotting (higher = faster visual updates)

Adjust these values to experiment with different navigation, mapping, and localization behaviors.

---

## Project Structure

- `simulation_manager.py`  
  Main orchestrator: synchronizes physics, sensors, mapping, and mission logic.

- `exploration.py`  
  Finite State Machine, frontier detection, and path-following logic.

- `path_planner.py`  
  Directional A* (Hybrid A* Lite) implementation.

- `particle_filter.py` & `ekf.py`  
  Particle Filter and EKF localization modules.

- `slam.py`  
  Occupancy grid mapping with log-odds updates.

- `perception_module.py`  
  CLIP-based semantic perception and vision-language interface.

- `config.py`  
  Global configuration and hyperparameters.

---

## Author

- Paul Keyvan Hoang-Long Pham Dang