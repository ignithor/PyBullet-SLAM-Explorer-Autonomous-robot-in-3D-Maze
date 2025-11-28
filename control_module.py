# import numpy as np

# # --- Navigation Constants ---
# TARGET_DISTANCE = 0.5    # Target distance to maintain from the wall (in meters)
# MAX_LINEAR_SPEED = 800.0   # Maximum linear speed for the wheels (rad/s)
# P_GAIN = 1.5             # Proportional gain for steering (P in PID)
# LIDAR_RAYS = 36          # Should match the value in robot.py

# class ControlModule:
#     """
#     Implements the Wall Following exploration strategy (Right-Hand Rule).
#     The robot tries to maintain a constant distance from the wall on its right.
#     """
#     def __init__(self, robot_instance):
#         self.robot = robot_instance
#         print("INFO: ControlModule (Right Wall Following) initialized.")

#     def calculate_steering(self, lidar_ranges):
#         """
#         Uses LiDAR data to compute the required steering velocities.
#         """
        
#         # 1. Identify the range index corresponding to the RIGHT side.
#         # 270 degrees = 3/4 * LIDAR_RAYS
#         right_index = LIDAR_RAYS * 3 // 4 
        
#         # Use an average of a few rays around the 270-degree mark
#         right_rays_indices = [
#             (right_index - 2) % LIDAR_RAYS,
#             (right_index - 1) % LIDAR_RAYS, 
#             right_index, 
#             (right_index + 1) % LIDAR_RAYS,
#             (right_index + 2) % LIDAR_RAYS
#         ]
                             
#         # Filter out max range readings and take the minimum distance
#         relevant_ranges = [lidar_ranges[i] for i in right_rays_indices if lidar_ranges[i] < self.robot.LIDAR_MAX_RANGE]
        
#         if not relevant_ranges:
#             # No wall detected on the right -> turn RIGHT sharply
#             current_distance = 2 * TARGET_DISTANCE 
#         else:
#             # Use the shortest distance in the sector
#             current_distance = np.min(relevant_ranges)

#         # 2. Calculate the Error
#         # Positive error: too far from the wall -> steer RIGHT
#         # Negative error: too close to the wall -> steer LEFT
#         error = current_distance - TARGET_DISTANCE
        
#         # 3. Proportional Control (Steering)
#         steering_adjustment = error * P_GAIN

#         # 4. Determine Final Speeds
        
#         # If no wall is detected on the right, prioritize turning sharply RIGHT:
#         if current_distance > (TARGET_DISTANCE + 0.1): 
#             # Turn Right (L_speed > R_speed)
#             left_speed = MAX_LINEAR_SPEED * 1.0 
#             right_speed = MAX_LINEAR_SPEED * 0.5
        
#         # If a wall is close, follow the wall:
#         else:
#             base_speed = MAX_LINEAR_SPEED * 0.8
            
#             # Apply proportional control:
#             # If error > 0 (too far): L_speed increases, R_speed decreases (turn RIGHT)
#             # If error < 0 (too close): L_speed decreases, R_speed increases (turn LEFT)
#             left_speed = base_speed + steering_adjustment
#             right_speed = base_speed - steering_adjustment
            
#         # 5. Clamp Speeds 
#         left_speed = np.clip(left_speed, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED)
#         right_speed = np.clip(right_speed, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED)

#         return left_speed, right_speed
        
#     def step(self):
#         """
#         Performs one control step: reads sensors, calculates command, and executes command.
#         """
#         # 1. Sense
#         lidar_ranges = self.robot.get_lidar_data()
        
#         # 2. Plan/Control
#         left_vel, right_vel = self.calculate_steering(lidar_ranges)
        
#         # 3. Act
#         self.robot.set_velocity(left_vel, right_vel)
        
#         camera_rgb = self.robot.get_camera_image()
        
#         return lidar_ranges, camera_rgb



import numpy as np

# --- Navigation Constants ---
MAX_LINEAR_SPEED = 20.0   # Reduced from 800.0 for safety (unless your sim requires 800)
LIDAR_RAYS = 36           # Should match the value in robot.py

class ControlModule:
    """
    A simple control module that drives the robot straight
    while strictly returning sensor data for analysis.
    """
    def __init__(self, robot_instance):
        self.robot = robot_instance
        print("INFO: ControlModule (Straight Motion) initialized.")

    def calculate_steering(self, lidar_ranges):
        """
        Ignores LiDAR data and sets wheels to equal velocities to go straight.
        """
        # Set both wheels to the same speed to drive straight
        # Adjust 0.5 to 1.0 to change speed
        left_speed = MAX_LINEAR_SPEED * 1.0 
        right_speed = MAX_LINEAR_SPEED * 1.0

        return left_speed, right_speed
        
    def step(self):
        """
        Performs one control step: reads sensors, calculates command, and executes command.
        Returns the sensor data as requested.
        """
        # 1. Sense
        lidar_ranges = self.robot.get_lidar_data()
        
        # 2. Plan/Control (Just go straight)
        left_vel, right_vel = self.calculate_steering(lidar_ranges)
        
        # 3. Act
        self.robot.set_velocity(left_vel, right_vel)
        
        # 4. Capture Camera Data
        camera_rgb = self.robot.get_camera_image()
        
        # Return the data as requested
        return lidar_ranges, camera_rgb