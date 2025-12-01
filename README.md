# PyBullet-SLAM-Explorer-Autonomous-robot-in-3D-Maze

This project implements a software simulation of an autonomous mobile robot navigating a procedurally generated 3D maze environment. Built using Python and the PyBullet physics engine, the simulation integrates key concepts in robotics and artificial intelligence, including Simultaneous Localization and Mapping (SLAM), Deep Learning-based Object Recognition, and Sensor Fusion.

The robot is equipped with simulated LiDAR and Camera sensors. It builds a real-time 2D occupancy grid map of the unknown environment while simultaneously scanning the visual scene to detect a specific target (a yellow duck) using the CLIP vision-language model.

## Features

- 3D Physics Simulation: Realistic differential-drive robot dynamics and collision detection using PyBullet.

- Procedural Environment: Dynamic generation of "perfect" mazes (no loops) of variable sizes using the Recursive Backtracking algorithm.

- Lidar-based SLAM: Implementation of Occupancy Grid Mapping with Log-Odds probabilistic updates to construct a reliable map from noisy sensor data.

- AI Perception: Integration of OpenAI's CLIP model for zero-shot object detection (detecting "a yellow duck" vs "a wall").

- Modular Architecture: Clean separation of concerns with distinct modules for robot control, mapping, perception, and simulation management.

- Real-time Visualization: Live plotting of the SLAM map and robot trajectory using Matplotlib.

- Manual Control: GUI-based slider control for human-in-the-loop testing and data gathering.


## Run the code 

To start the simulation do 

```python main.py```

## Controls

Once the simulation starts, two windows will appear:

- PyBullet GUI: Shows the 3D robot and maze.

  Use the sliders on the side (or press F1 to toggle the menu) labelled "Left Wheel Vel" and "Right Wheel Vel" to drive the robot manually.

- Matplotlib Window: Shows the live generated map.

  White: Free space.

  Black: Walls/Obstacles.

  Gray: Unknown area.

## Object Detection

Drive the robot around the maze. Every 5 seconds, the system captures an image and analyzes it. If the robot faces the target object (the yellow duck), the terminal will output:

 DUCK DETECTED!


## Dependencies

Tested with Python version 3.13.9

pip install pybullet # pybullet-3.2.7

conda install numpy

conda install matplotlib

pip install opencv-contrib-python 

pip install scipy 

pip install ffmpeg-python 

pip install transformers pillow pytorch
