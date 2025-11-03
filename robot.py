import pybullet as p
import numpy as np

# --- Configuration Constants (Can be moved to a separate config file later) ---
WHEEL_RADIUS = 0.05      # Radius of the robot wheels (in meters)
WHEEL_AXLE_LENGTH = 0.2  # Distance between the two wheels (axle length)
ROBOT_MASS = 1.0         # Mass of the robot body
LIDAR_MAX_RANGE = 5.0    # Maximum range for the simulated LiDAR (in meters)
LIDAR_RAYS = 36          # Number of rays for the LiDAR
LIDAR_ANGLE = 360        # Total angular sweep of the LiDAR (in degrees)

class Robot:
    """
    Represents a simple differential drive robot in the PyBullet environment.
    Handles low-level control (velocity) and sensor simulation (LiDAR, Camera).
    """
    def __init__(self, client_id, start_pos=[0.5, 0.5, 0.1]):
        self.client = client_id
        self.start_pos = start_pos
        self.robot_id = self._load_robot_model()
        self.wheel_joints = self._find_wheel_joints()
        self.camera_setup = self._setup_camera_parameters()
        
        print(f"INFO: Robot (Differential Drive) initialized with ID {self.robot_id}.")

    def _load_robot_model(self):
        """
        Loads the robot model (a simplified structure: base + 2 motorized wheels).
        Since URDF/SDF creation is complex, we use PyBullet's built-in creation function 
        to define a simple box base with two hinge joints for wheels.
        """
        
        # 1. Base Shape (Simplified Box)
        base_half_extents = [0.15, 0.1, 0.05]
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=base_half_extents, physicsClientId=self.client)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=base_half_extents, rgbaColor=[1, 0, 0, 1], physicsClientId=self.client)
        
        base_id = p.createMultiBody(baseMass=ROBOT_MASS, 
                                    baseCollisionShapeIndex=col_shape, 
                                    baseVisualShapeIndex=vis_shape, 
                                    basePosition=self.start_pos, 
                                    physicsClientId=self.client)

        # In a full differential model, you'd add joints and wheels here. 
        # For simplicity and focusing on control, we often treat the single body (base_id) 
        # as the robot and apply forces/velocities directly to it, or use a pre-built URDF.
        # Here, we keep the simple base ID and simulate differential control via velocity.
        
        return base_id

    def _find_wheel_joints(self):
        """Placeholder for finding wheel joint IDs if a full URDF was used."""
        # Since we used a simple single base, this returns empty.
        # In a full model, this would return [left_wheel_joint_id, right_wheel_joint_id]
        return []

    def set_velocity(self, left_vel, right_vel):
        """
        Implements low-level control: sets the target angular velocity for the wheels.
        
        Since we are using a simplified base without explicit wheel joints,
        this function calculates the resulting linear and angular velocity 
        and applies it directly to the base. This simplifies the physics setup 
        at the cost of some realism.
        
        Args:
            left_vel (float): Target angular velocity of the left wheel (rad/s).
            right_vel (float): Target angular velocity of the right wheel (rad/s).
        """
        
        # Calculate Linear Velocity (forward movement)
        linear_vel = WHEEL_RADIUS * (right_vel + left_vel) / 2.0
        
        # Calculate Angular Velocity (turning rate)
        angular_vel = WHEEL_RADIUS * (right_vel - left_vel) / WHEEL_AXLE_LENGTH
        
        # Get current orientation to apply local velocity
        _, orientation = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        
        # Convert quaternion to rotation matrix (or just use vectors)
        rot_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        
        # Apply linear velocity in the robot's forward (X) direction
        forward_vec = rot_matrix[:, 0] * linear_vel
        
        # Apply the velocities
        p.resetBaseVelocity(self.robot_id, 
                            linearVelocity=forward_vec, 
                            angularVelocity=[0, 0, angular_vel],
                            physicsClientId=self.client)

    # --- Sensor Simulation ---

    def _setup_camera_parameters(self, width=64, height=64, fov=60):
        """Defines parameters for the simulated camera sensor."""
        aspect = width / height
        near_val = 0.01
        far_val = 10.0
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_val, far_val, physicsClientId=self.client)
        return {
            'width': width,
            'height': height,
            'proj_matrix': projection_matrix,
            'fov': fov,
            'near': near_val,
            'far': far_val
        }

    def get_camera_image(self):
        """
        Simulates the robot's camera view using p.getCameraImage.
        Returns the RGB image array (and optionally depth/segmentation).
        """
        # Get robot position and orientation
        pos, quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        
        # Define the position of the camera relative to the robot's base (e.g., a bit above the center)
        camera_offset_z = 0.1
        camera_pos = [pos[0], pos[1], pos[2] + camera_offset_z]
        
        # Define the target point for the camera (look straight ahead)
        # Calculate the forward vector based on current orientation
        rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        forward_vec = rot_matrix[:, 0]
        
        target_pos = [camera_pos[i] + forward_vec[i] for i in range(3)]
        
        # View Matrix (defines camera position/orientation)
        view_matrix = p.computeViewMatrix(cameraEyePosition=camera_pos, 
                                          cameraTargetPosition=target_pos, 
                                          cameraUpVector=[0, 0, 1], # Z is up in PyBullet
                                          physicsClientId=self.client)
        
        # Capture the image
        img_data = p.getCameraImage(width=self.camera_setup['width'],
                                    height=self.camera_setup['height'],
                                    viewMatrix=view_matrix,
                                    projectionMatrix=self.camera_setup['proj_matrix'],
                                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                    physicsClientId=self.client)
        
        # Extract and return the RGB image (discard alpha channel)
        rgb_array = np.reshape(img_data[2], (self.camera_setup['height'], self.camera_setup['width'], 4))[:, :, :3]
        return rgb_array.astype(np.uint8)


    def get_lidar_data(self):
        """
        Simulates a 360-degree 2D LiDAR sensor using p.rayTestBatch.
        Returns an array of distances (ranges) to obstacles.
        """
        pos, quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        
        # Calculate the rotation angles for all rays
        angles = np.linspace(0, 2 * np.pi, LIDAR_RAYS, endpoint=False)
        
        # Convert robot orientation quaternion to Euler angles to get Yaw (Z rotation)
        _, _, yaw = p.getEulerFromQuaternion(quat)
        
        start_points = []
        end_points = []
        
        # Define ray start and end points in world coordinates
        for angle in angles:
            # Angle in world frame (robot yaw + relative ray angle)
            world_angle = yaw + angle
            
            # Start point (slightly above the ground)
            start_point = [pos[0], pos[1], 0.05]
            
            # End point (LIDAR_MAX_RANGE away)
            end_point = [
                pos[0] + LIDAR_MAX_RANGE * np.cos(world_angle),
                pos[1] + LIDAR_MAX_RANGE * np.sin(world_angle),
                0.05
            ]
            
            start_points.append(start_point)
            end_points.append(end_point)
        
        # Perform the ray casting
        ray_results = p.rayTestBatch(start_points, end_points, physicsClientId=self.client)
        
        distances = []
        # ray_results format: [hitFraction, hitPosition, hitNormal, objectUniqueId, linkIndex]
        for result in ray_results:
            hit_fraction = result[0]
            # Distance is hitFraction * MaxRange
            distance = hit_fraction * LIDAR_MAX_RANGE
            distances.append(distance)
            
        return np.array(distances)