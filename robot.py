import pybullet as p
import numpy as np
import os  # Required for handling file paths

# --- Configuration Constants for the Mobile Robot (Adjust as needed) ---

# !!! CRITICAL: UPDATE THIS PATH !!!
# You must adjust this path to point to the actual URDF file in your cloned repository.
# Example assumption based on common repo structures:
# If you cloned 'PybulletRobotics' into your project directory:
ROBOT_URDF_PATH = os.path.join(os.getcwd(), 'urdf', 'simple_two_wheel_car.urdf')

# Motor/Joint Configuration (MUST be verified against the URDF structure)
# These indices are placeholders; check the URDF for the actual index numbers
# of the Left and Right wheel joints.
LEFT_WHEEL_JOINT_INDEX = 0
RIGHT_WHEEL_JOINT_INDEX = 1
MAX_MOTOR_FORCE = 50.0  # Force used in setJointMotorControl2

# Sensor Constants (Can remain as is if you are satisfied with the simulation)
LIDAR_MAX_RANGE = 5.0
LIDAR_RAYS = 36


class Robot:
    """
    Represents the external differential drive robot loaded via URDF.
    Handles low-level control (velocity) and sensor simulation (LiDAR, Camera).
    """

    def __init__(self, client_id, start_pos=[0.5, 0.5, 0.1]):
        self.client = client_id
        self.start_pos = start_pos
        self.robot_id = self._load_robot_model()
        self.wheel_joints = self._find_wheel_joints()
        self.camera_setup = self._setup_camera_parameters()

        # Ensure the wheels are set up to be controllable by motors
        p.setJointMotorControlArray(self.robot_id, self.wheel_joints, p.VELOCITY_CONTROL, forces=[0, 0],
                                    physicsClientId=self.client)

        print(f"INFO: External Robot (URDF) initialized with ID {self.robot_id}.")

    def _load_robot_model(self):
        """
        Loads the robot model from the specified URDF path.
        """
        # Ensure the URDF path exists
        if not os.path.exists(ROBOT_URDF_PATH):
            raise FileNotFoundError(
                f"CRITICAL: Robot URDF not found at path: {ROBOT_URDF_PATH}. Please update ROBOT_URDF_PATH.")

        # Load the URDF model
        robot_id = p.loadURDF(ROBOT_URDF_PATH,
                              basePosition=self.start_pos,
                              baseOrientation=[0, 0, 0, 1],
                              useFixedBase=False,
                              physicsClientId=self.client)

        return robot_id

    def _find_wheel_joints(self):
        """
        Identifies and returns the PyBullet joint IDs for the left and right motorized wheels.
        This relies on the constants defined above.
        """
        # If the URDF is well-structured, the joints are usually the first few indices.
        # Verify LEFT_WHEEL_JOINT_INDEX and RIGHT_WHEEL_JOINT_INDEX above!
        return [LEFT_WHEEL_JOINT_INDEX, RIGHT_WHEEL_JOINT_INDEX]

    def set_velocity(self, left_vel, right_vel):
        """
        Implements low-level differential control using PyBullet's setJointMotorControl2.

        Args:
            left_vel (float): Target angular velocity of the left wheel (rad/s).
            right_vel (float): Target angular velocity of the right wheel (rad/s).
        """
        if not self.wheel_joints:
            # Fallback for simple body if joint IDs were not found
            # You should remove this once you verify the joint indices
            print("WARNING: Wheel joints not defined. Using simplified body control.")
            return

        left_wheel_id, right_wheel_id = self.wheel_joints[0], self.wheel_joints[1]

        # Set motor control for the left wheel
        p.setJointMotorControl2(self.robot_id,
                                left_wheel_id,
                                p.VELOCITY_CONTROL,
                                targetVelocity=left_vel,
                                force=MAX_MOTOR_FORCE,
                                physicsClientId=self.client)

        # Set motor control for the right wheel
        p.setJointMotorControl2(self.robot_id,
                                right_wheel_id,
                                p.VELOCITY_CONTROL,
                                targetVelocity=right_vel,
                                force=MAX_MOTOR_FORCE,
                                physicsClientId=self.client)

    # --- Sensor Simulation (Retained but potentially requires adjustment for placement) ---

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
        Simulates the robot's camera view. Ensure the camera position is outside the URDF geometry.
        """
        pos, quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)

        # Adjust these offsets based on the actual URDF's dimensions!
        CAMERA_OFFSET_FORWARD = 0.15
        CAMERA_OFFSET_Z = 0.1
        TARGET_OFFSET_FORWARD = 0.5
        UP_VECTOR = [0, 0, 1]

        # Calculate camera position (Eye)
        camera_pos = [
            pos[0] + rot_matrix[0, 0] * CAMERA_OFFSET_FORWARD,
            pos[1] + rot_matrix[1, 0] * CAMERA_OFFSET_FORWARD,
            pos[2] + CAMERA_OFFSET_Z
        ]

        # Calculate target position
        target_pos = [
            pos[0] + rot_matrix[0, 0] * TARGET_OFFSET_FORWARD,
            pos[1] + rot_matrix[1, 0] * TARGET_OFFSET_FORWARD,
            pos[2] + CAMERA_OFFSET_Z
        ]

        view_matrix = p.computeViewMatrix(cameraEyePosition=camera_pos,
                                          cameraTargetPosition=target_pos,
                                          cameraUpVector=UP_VECTOR,
                                          physicsClientId=self.client)

        img_data = p.getCameraImage(width=self.camera_setup['width'],
                                    height=self.camera_setup['height'],
                                    viewMatrix=view_matrix,
                                    projectionMatrix=self.camera_setup['proj_matrix'],
                                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                    physicsClientId=self.client)

        rgb_array = np.reshape(img_data[2], (self.camera_setup['height'], self.camera_setup['width'], 4))[:, :, :3]
        return rgb_array.astype(np.uint8)

    def get_lidar_data(self):
        """
        Simulates a 360-degree 2D LiDAR sensor using p.rayTestBatch.
        """
        pos, quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)

        angles = np.linspace(0, 2 * np.pi, LIDAR_RAYS, endpoint=False)
        _, _, yaw = p.getEulerFromQuaternion(quat)

        start_points = []
        end_points = []

        # Define ray start and end points
        for angle in angles:
            world_angle = yaw + angle
            start_point = [pos[0], pos[1], 0.05]
            end_point = [
                pos[0] + LIDAR_MAX_RANGE * np.cos(world_angle),
                pos[1] + LIDAR_MAX_RANGE * np.sin(world_angle),
                0.05
            ]

            start_points.append(start_point)
            end_points.append(end_point)

        ray_results = p.rayTestBatch(start_points, end_points, physicsClientId=self.client)

        distances = []
        for result in ray_results:
            hit_fraction = result[0]
            distance = hit_fraction * LIDAR_MAX_RANGE
            distances.append(distance)

        return np.array(distances)