import numpy as np
import config as cfg

class EKF:
    """
    Extended Kalman Filter for Differential Drive Robot.
    
    State Vector: [x, y, theta]
    Control Vector: [v, omega] (Linear and Angular velocity)
    """
    def __init__(self, start_pos, start_yaw, dt):
        self.dt = dt
        
        # Initial State [x, y, theta]
        self.mu = np.array([start_pos[0], start_pos[1], start_yaw])
        
        # Initial Covariance
        self.Sigma = np.eye(3) * 0.1
        
        # --- TUNING PARAMETERS (THE FIX) ---
        # Process Noise (R): Uncertainty in Motion
        # We increase this significantly (especially for theta) because
        # the robot slips/skids in the simulation, making odometry unreliable.
        # [x_var, y_var, theta_var]
        self.R = np.diag(cfg.EKF_PROCESS_NOISE) ** 2 
        
        # Measurement Noise (Q): Uncertainty in Sensor
        # We decrease this because the simulated Compass is nearly perfect.
        # This forces the EKF to snap the heading to the true value.
        self.Q = np.diag(cfg.EKF_MEASURE_NOISE) ** 2 

    def predict(self, v, omega):
        """
        Prediction Step (Dead Reckoning / Odometry).
        Updates state based on control inputs (v, omega).
        """
        theta = self.mu[2]
        
        # 1. State Prediction (Motion Model)
        # Using exact arc integration for better stability
        if abs(omega) < 1e-5:
            dx = v * np.cos(theta) * self.dt
            dy = v * np.sin(theta) * self.dt
        else:
            dx = (v / omega) * (np.sin(theta + omega * self.dt) - np.sin(theta))
            dy = (v / omega) * (-np.cos(theta + omega * self.dt) + np.cos(theta))
        
        dtheta = omega * self.dt
        
        self.mu += np.array([dx, dy, dtheta])
        
        # Normalize Theta to [-pi, pi]
        self.mu[2] = (self.mu[2] + np.pi) % (2 * np.pi) - np.pi
        
        # 2. Covariance Prediction (Jacobian G)
        # Jacobian of the motion model with respect to state
        G = np.eye(3)
        G[0, 2] = -v * np.sin(theta) * self.dt
        G[1, 2] =  v * np.cos(theta) * self.dt
        
        self.Sigma = G @ self.Sigma @ G.T + self.R

    def update_compass(self, z_theta):
        """
        Correction Step using a Compass/IMU.
        z_theta: The measured heading from the sensor.
        """
        predicted_theta = self.mu[2]
        
        # Innovation (Measurement Residual)
        y = z_theta - predicted_theta
        
        # Normalize residual to [-pi, pi]
        y = (y + np.pi) % (2 * np.pi) - np.pi
        
        # Jacobian H (Measurement Model)
        # We observe only theta directly: H = [0, 0, 1]
        H = np.array([[0, 0, 1]])
        
        # Residual Covariance S
        S = H @ self.Sigma @ H.T + self.Q
        
        # Kalman Gain K
        K = self.Sigma @ H.T @ np.linalg.inv(S)
        
        # Update State
        self.mu = self.mu + (K * y).flatten()
        
        # Normalize Theta again after update
        self.mu[2] = (self.mu[2] + np.pi) % (2 * np.pi) - np.pi
        
        # Update Covariance
        I = np.eye(3)
        self.Sigma = (I - K @ H) @ self.Sigma

    def get_pose(self):
        """Returns current estimate (x, y, yaw)."""
        return self.mu