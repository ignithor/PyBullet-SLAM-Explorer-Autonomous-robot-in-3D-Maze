import pybullet as p
import numpy as np
import os

# --- Robot Constants ---
ROBOT_URDF_PATH = os.path.join(os.getcwd(), "urdf", "simple_two_wheel_car.urdf")

LEFT_WHEEL_JOINT_INDEX = 0
RIGHT_WHEEL_JOINT_INDEX = 1

MAX_MOTOR_FORCE = 350.0

# --- Physical Dimensions (Estimated for EKF) ---
# These must match your URDF geometry for Odometry to be accurate!
WHEEL_RADIUS = 0.05  # meters
TRACK_WIDTH = 0.3    # meters (distance between wheels)

# --- LiDAR Constants ---
LIDAR_RAYS = 36
LIDAR_RANGE = 10.0
# Z height of the LiDAR (keep higher than wheel height)
LIDAR_Z = 0.25
LIDAR_START_OFFSET = 0.0

# --- Camera Constants ---
CAMERA_WIDTH = 320  # Width in pixels
CAMERA_HEIGHT = 240 # Height in pixels
CAMERA_FOV = 60       # Field of View (degrees)
CAMERA_ASPECT = CAMERA_WIDTH / CAMERA_HEIGHT
CAMERA_NEAR = 0.1     # Near clip plane
CAMERA_FAR = 10.0     # Far clip plane
# Camera position relative to robot's center (base_link)
CAMERA_X_OFFSET = 0.0  # Slightly in front
CAMERA_Z_OFFSET = 0.3   # Above the LiDAR


class Robot:
    """
    Robot class with LiDAR and Camera implementation.
    """

    def __init__(self, client_id, start_pos=[0.5, 0.5, 0.1]):
        self.client = client_id
        self.start_pos = start_pos

        # Store debug lines so we can clear them each frame
        # Initialize this first to prevent errors if init fails early
        self.debug_lines = []

        self.robot_id = self._load_robot()

        self.wheel_joints = [LEFT_WHEEL_JOINT_INDEX, RIGHT_WHEEL_JOINT_INDEX]
        self.LIDAR_MAX_RANGE = 10.0
        # Enable simple motor control
        p.setJointMotorControlArray(
            self.robot_id,
            self.wheel_joints,
            p.VELOCITY_CONTROL,
            forces=[MAX_MOTOR_FORCE, MAX_MOTOR_FORCE],
            physicsClientId=self.client
        )

        # Calculate projection matrix (only once)
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=CAMERA_FOV,
            aspect=CAMERA_ASPECT,
            nearVal=CAMERA_NEAR,
            farVal=CAMERA_FAR,
            physicsClientId=self.client
        )

        print("INFO: Robot initialized.")

    # -------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------
    def _load_robot(self):
        if not os.path.exists(ROBOT_URDF_PATH):
            raise FileNotFoundError(f"URDF not found: {ROBOT_URDF_PATH}")

        return p.loadURDF(
            ROBOT_URDF_PATH,
            basePosition=self.start_pos,
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=False,
            physicsClientId=self.client
        )

    # -------------------------------------------------------------
    # Movement
    # -------------------------------------------------------------
    def set_velocity(self, left_vel, right_vel):
        """
        Sets velocity for the differential drive wheels.
        """
        p.setJointMotorControl2(
            self.robot_id,
            LEFT_WHEEL_JOINT_INDEX,
            p.VELOCITY_CONTROL,
            targetVelocity=left_vel,
            force=MAX_MOTOR_FORCE,
            physicsClientId=self.client
        )

        p.setJointMotorControl2(
            self.robot_id,
            RIGHT_WHEEL_JOINT_INDEX,
            p.VELOCITY_CONTROL,
            targetVelocity=right_vel,
            force=MAX_MOTOR_FORCE,
            physicsClientId=self.client
        )

    def get_wheel_velocity(self):
        """
        Simulates Wheel Encoders.
        Returns the actual angular velocity of (left_wheel, right_wheel) in rad/s.
        """
        # getJointState returns: (pos, vel, reaction_forces, applied_torque)
        left_state = p.getJointState(self.robot_id, LEFT_WHEEL_JOINT_INDEX, physicsClientId=self.client)
        right_state = p.getJointState(self.robot_id, RIGHT_WHEEL_JOINT_INDEX, physicsClientId=self.client)
        
        return left_state[1], right_state[1]

    def get_compass_reading(self):
        """
        Simulates a Compass/IMU.
        Returns current yaw angle. In a real scenario, this would have noise added.
        """
        _, quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        yaw = p.getEulerFromQuaternion(quat)[2]
        return yaw

    def get_lidar_data(self):
        """
        Clean + correct 2D LiDAR using PyBullet ray tests.
        """
        for line_id in self.debug_lines:
            p.removeUserDebugItem(line_id, physicsClientId=self.client)
        self.debug_lines.clear()

        # Get robot pose
        pos, quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        yaw = p.getEulerFromQuaternion(quat)[2]

        angles = np.linspace(0, 2 * np.pi, LIDAR_RAYS, endpoint=False)

        start_points, end_points = [], []

        for angle in angles:
            world_angle = yaw + angle

            # Start a bit outside the body
            start = [
                pos[0] + np.cos(world_angle) * LIDAR_START_OFFSET,
                pos[1] + np.sin(world_angle) * LIDAR_START_OFFSET,
                LIDAR_Z
            ]

            end = [
                start[0] + np.cos(world_angle) * LIDAR_RANGE,
                start[1] + np.sin(world_angle) * LIDAR_RANGE,
                LIDAR_Z
            ]

            start_points.append(start)
            end_points.append(end)

        results = p.rayTestBatch(start_points, end_points, physicsClientId=self.client)

        distances = []

        for i, result in enumerate(results):
            hit_object = result[0]
            hit_fraction = result[2]
            hit_position = result[3]

            if hit_object == -1:
                # No hit
                distance = LIDAR_RANGE
                line_end = end_points[i]
                color = [1, 0, 0] # Red
            else:
                # Hit
                distance = hit_fraction * LIDAR_RANGE
                line_end = hit_position
                color = [0, 1, 0] # Green

            distances.append(distance)

            # Draw debug line
            line_id = p.addUserDebugLine(
                start_points[i], line_end, color, 1, 0.05, physicsClientId=self.client
            )
            self.debug_lines.append(line_id)

        return np.array(distances)

    # -------------------------------------------------------------
    # Camera
    # -------------------------------------------------------------
    def get_camera_image(self): # <-- RENAMED THIS METHOD
        """
        Get the camera image data.
        """
        # Get robot base pose
        pos, quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        
        # Get rotation matrix from quaternion
        rot_matrix = p.getMatrixFromQuaternion(quat)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Camera offset (in robot's frame)
        local_cam_pos = np.array([CAMERA_X_OFFSET, 0, CAMERA_Z_OFFSET])
        
        # Target point (look straight ahead from camera, in robot's frame)
        # Look 1m in front of the camera
        local_cam_target = np.array([CAMERA_X_OFFSET + 1.0, 0, CAMERA_Z_OFFSET]) 
        
        # "Up" vector (in robot's frame, Z-axis)
        local_cam_up = np.array([0, 0, 1])

        # Transform positions/vectors to world coordinates
        # np.dot(rot_matrix, vector) is safer than rot_matrix.dot(vector)
        world_cam_pos = pos + np.dot(rot_matrix, local_cam_pos)
        world_cam_target = pos + np.dot(rot_matrix, local_cam_target)
        world_cam_up = np.dot(rot_matrix, local_cam_up)

        # Calculate view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=world_cam_pos,
            cameraTargetPosition=world_cam_target,
            cameraUpVector=world_cam_up,
            physicsClientId=self.client
        )

        # Get the image
        img_data = p.getCameraImage(
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL, # Use the main hardware renderer
            physicsClientId=self.client
        )
        
        # Extract RGB data
        # img_data[0] = width
        # img_data[1] = height
        # img_data[2] = RGB pixels (array of (height, width, 4) RGBA bytes)
        # img_data[3] = depth buffer
        # img_data[4] = segmentation buffer
        
        width = img_data[0]
        height = img_data[1]
        rgb_pixels = np.array(img_data[2]).reshape(height, width, 4)
        
        # Return the RGB image (dropping the Alpha channel)
        return width, height, rgb_pixels[:, :, :3]