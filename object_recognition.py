import pybullet as p
import pybullet_data
import time
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

physicsClient = p.connect(p.GUI)  # GUI mode
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

planeId = p.loadURDF("plane.urdf")


duck1 = p.loadURDF("duck_vhacd.urdf", [2, 0, 0], p.getQuaternionFromEuler([0, 90, 0]), globalScaling=10)
duck2 = p.loadURDF("duck_vhacd.urdf", [-2, 2, 0], p.getQuaternionFromEuler([0, 90, 0]), globalScaling=10)
duck3 = p.loadURDF("duck_vhacd.urdf", [0, -2, 0], p.getQuaternionFromEuler([0, 90, 0]), globalScaling=10)

robotStartPos = [0, 0, 0.2]
robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF("r2d2.urdf", robotStartPos, robotStartOrientation)

width = 320
height = 240
fov = 60
near = 0.1
far = 10


for step in range(10):
    
    # Camera attached to robot
    pos, orn = p.getBasePositionAndOrientation(robotId)
    rot_matrix = p.getMatrixFromQuaternion(orn)
    dir_vec = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]  # forward vector
    
    camEye = [pos[0], pos[1], pos[2]+0.5]           # raise camera above robot
    camTarget = [pos[0]+2*dir_vec[0], pos[1]+2*dir_vec[1], pos[2]+0.2]  # look further forward
    
    viewMatrix = p.computeViewMatrix(camEye, camTarget, [0,0,1])
    projectionMatrix = p.computeProjectionMatrixFOV(fov, width/height, near, far)
    
    # Get camera image
    width_img, height_img, rgbImg, depthImg, segImg = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    # Convert to proper RGB array
    rgb_array = np.array(rgbImg, dtype=np.uint8).reshape((height_img, width_img, 4))[:, :, :3]
    
    if step % 5 == 0:  # show every 5 steps
        plt.imshow(rgb_array)
        plt.axis('off')
        plt.show()
    print(f"Captured image at step {step}")


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Convert the last camera frame to PIL Image
image = Image.fromarray(rgb_array)

# Define candidate labels
labels = [ "a yellow duck", "a wall",]

inputs = processor(text=labels, images=[np.array(image)], return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

# Show probabilities
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob.item():.4f}")

predicted_label = labels[probs.argmax()]
print("Predicted label:", predicted_label)