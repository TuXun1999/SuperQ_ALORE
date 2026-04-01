"""
The program to load the grasp poses & calculate the associated inverse kinmetics
for the robot
"""
import json
import numpy as np
grasp_poses = json.load(open("./SuperQ_GRASP/grasp_poses.json", "r"))
grasp_poses = np.array(grasp_poses)

# TODO: Find a better way to select grasp poses
# Now, only select the one on the back of the chair
x_norm = 10000
for grasp_pose in grasp_poses:
    if abs(grasp_pose[0, 3]) < x_norm:
        x_norm = abs(grasp_pose[0, 3])
        selected_grasp_pose = grasp_pose
print("Selected grasp pose: " + str(selected_grasp_pose))