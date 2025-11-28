import pybullet as p
import numpy as np

class ControlModule:
    """
    Control module that enables manual control of the robot using PyBullet GUI sliders.
    It reads the slider values and applies them as wheel velocities.
    """
    def __init__(self, robot_instance):
        self.robot = robot_instance
        
        # --- Initialize Sliders ---
        # We create the sliders here so they appear when the module is instantiated.
        # Range: -15 rad/s to 15 rad/s (Adjust as needed)
        self.slider_left = p.addUserDebugParameter("Left Wheel Vel", -15, 15, 0)
        self.slider_right = p.addUserDebugParameter("Right Wheel Vel", -15, 15, 0)
        
        print("INFO: ControlModule (Manual Sliders) initialized.")

    def step(self):
        """
        Performs one control step:
        1. Reads the manual slider values.
        2. Sets the robot's wheel velocities.
        3. Captures and returns sensor data (LiDAR + Camera).
        """
        # 1. Read Manual Controls from Sliders
        # readUserDebugParameter returns the current value of the slider
        left_vel = p.readUserDebugParameter(self.slider_left)
        right_vel = p.readUserDebugParameter(self.slider_right)
        
        # 2. Act (Set Robot Velocity)
        self.robot.set_velocity(left_vel, right_vel)
        
        # 3. Sense (Get Sensor Data)
        # We must capture this data here to return it to the SimulationManager
        lidar_ranges = self.robot.get_lidar_data()
        camera_rgb = self.robot.get_camera_image()
        
        # Return the data as expected by the simulation loop
        return lidar_ranges, camera_rgb