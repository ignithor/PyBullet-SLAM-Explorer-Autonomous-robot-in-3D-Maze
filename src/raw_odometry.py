import numpy as np

class RawOdometry:
    """
    Computes robot pose based purely on wheel encoders (Dead Reckoning).
    No filtering, no compass correction. Subject to unbounded drift.
    """
    def __init__(self, start_pos, start_yaw, dt):
        self.dt = dt
        # State: [x, y, theta]
        self.pose = np.array([start_pos[0], start_pos[1], start_yaw])

    def update(self, v, omega):
        """
        Updates pose based on linear velocity (v) and angular velocity (omega).
        """
        x, y, theta = self.pose
        
        # Avoid division by zero for straight lines
        if abs(omega) < 1e-5:
            # Straight line motion
            dx = v * np.cos(theta) * self.dt
            dy = v * np.sin(theta) * self.dt
            dtheta = 0.0
        else:
            # Exact arc motion model (more accurate for curves)
            # R = v / omega
            # dx = R * (sin(theta + w*dt) - sin(theta))
            # dy = -R * (cos(theta + w*dt) - cos(theta))
            
            radius = v / omega
            dx = radius * (np.sin(theta + omega * self.dt) - np.sin(theta))
            dy = radius * (-np.cos(theta + omega * self.dt) + np.cos(theta))
            dtheta = omega * self.dt
        
        self.pose[0] += dx
        self.pose[1] += dy
        self.pose[2] += dtheta
        
        # Normalize Theta to [-pi, pi]
        self.pose[2] = (self.pose[2] + np.pi) % (2 * np.pi) - np.pi

    def get_pose(self):
        """Returns [x, y, yaw]"""
        return self.pose