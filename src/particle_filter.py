import numpy as np

class ParticleFilter:
    """
    Particle Filter for Differential Drive Robot Localization.
    Uses a set of particles to represent the posterior distribution of the robot pose.
    """
    def __init__(self, start_pos, start_yaw, dt, num_particles=200):
        self.dt = dt
        self.num_particles = num_particles
        
        # Initialize particles [x, y, theta] around the start position
        # Add some initial noise so they aren't all identical
        self.particles = np.zeros((num_particles, 3))
        self.particles[:, 0] = start_pos[0] + np.random.normal(0, 0.1, num_particles)
        self.particles[:, 1] = start_pos[1] + np.random.normal(0, 0.1, num_particles)
        self.particles[:, 2] = start_yaw + np.random.normal(0, 0.1, num_particles)
        
        # Initialize weights (uniform distribution)
        self.weights = np.ones(num_particles) / num_particles
        
        # --- TUNING PARAMETERS ---
        # Motion Noise (Process Noise): [x_std, y_std, theta_std]
        # High theta noise handles wheel slippage well
        self.odom_noise = [0.02, 0.02, 0.05] 
        
        # Measurement Noise (Compass): standard deviation in radians
        # Low value trusts the compass significantly
        self.compass_noise = 0.05

    def predict(self, v, omega):
        """
        Motion Model: Moves every particle according to control input (v, omega)
        plus random Gaussian noise.
        """
        # Vectorized implementation for speed
        thetas = self.particles[:, 2]
        
        # Prevent division by zero for straight lines
        # We define a small epsilon. 
        # Logic: If omega is small, use simple trig approximations.
        
        # Calculate deltas without noise
        # Note: We handle the array logic by checking if omega is effectively zero scalar
        # but since omega is a scalar input here, it's easy.
        
        if abs(omega) < 1e-5:
            dx = v * np.cos(thetas) * self.dt
            dy = v * np.sin(thetas) * self.dt
            dtheta = 0.0
        else:
            # Exact arc motion model
            dx = (v / omega) * (np.sin(thetas + omega * self.dt) - np.sin(thetas))
            dy = (v / omega) * (-np.cos(thetas + omega * self.dt) + np.cos(thetas))
            dtheta = omega * self.dt

        # Add motion to particles
        self.particles[:, 0] += dx
        self.particles[:, 1] += dy
        self.particles[:, 2] += dtheta
        
        # Add Process Noise (The "Monte Carlo" part)
        # We add random jitter to every particle to account for slippage/error
        self.particles[:, 0] += np.random.normal(0, self.odom_noise[0], self.num_particles)
        self.particles[:, 1] += np.random.normal(0, self.odom_noise[1], self.num_particles)
        self.particles[:, 2] += np.random.normal(0, self.odom_noise[2], self.num_particles)
        
        # Normalize angles to [-pi, pi]
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

    def update_compass(self, measured_yaw):
        """
        Measurement Model: Reweights particles based on how well they match
        the compass measurement.
        """
        # Calculate error between measurement and particle yaw
        yaw_diff = measured_yaw - self.particles[:, 2]
        
        # Normalize diff to [-pi, pi]
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
        
        # Gaussian Likelihood function
        # High probability if diff is close to 0, Low if far
        sigma = self.compass_noise
        likelihood = np.exp(-(yaw_diff**2) / (2 * sigma**2))
        
        # Update weights
        self.weights *= likelihood
        
        # Handle tiny weights (avoid division by zero)
        self.weights += 1.e-300
        
        # Normalize weights so they sum to 1
        self.weights /= np.sum(self.weights)
        
        # Resample if necessary
        self._resample_if_needed()

    def _resample_if_needed(self):
        """
        Low Variance Resampling: Keeps particles with high weights,
        discards those with low weights.
        """
        # Effective number of particles (N_eff)
        # Measure of how "degenerate" the weights are
        n_eff = 1.0 / np.sum(self.weights**2)
        
        # Resample if N_eff drops below half the particles
        if n_eff < self.num_particles / 2.0:
            indices = np.zeros(self.num_particles, dtype=int)
            c = self.weights[0]
            r = np.random.uniform(0, 1.0 / self.num_particles)
            i = 0
            
            for m in range(self.num_particles):
                u = r + m * (1.0 / self.num_particles)
                while u > c:
                    i = (i + 1) % self.num_particles
                    c += self.weights[i]
                indices[m] = i
            
            # Keep the selected particles
            self.particles = self.particles[indices]
            # Reset weights to uniform
            self.weights.fill(1.0 / self.num_particles)

    def get_estimate(self):
        """
        Returns the mean estimate of the robot's pose.
        """
        # Weighted mean for X and Y
        mean_x = np.average(self.particles[:, 0], weights=self.weights)
        mean_y = np.average(self.particles[:, 1], weights=self.weights)
        
        # Weighted mean for Yaw (requires handling circularity)
        sin_sum = np.sum(np.sin(self.particles[:, 2]) * self.weights)
        cos_sum = np.sum(np.cos(self.particles[:, 2]) * self.weights)
        mean_yaw = np.arctan2(sin_sum, cos_sum)
        
        return np.array([mean_x, mean_y, mean_yaw])