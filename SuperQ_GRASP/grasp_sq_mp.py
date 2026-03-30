import numpy as np
import os


import sys
sys.path.append(os.getcwd())

import trimesh
import open3d as o3d

import argparse
from SuperQ_GRASP.superquadrics import *
from utils.mesh_process import *
from utils.image_process import *

from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

# Necessary Packages for sq parsing
from SuperQ_GRASP.Marching_Primitives.sq_split import sq_predict_mp
from SuperQ_GRASP.Marching_Primitives.MPS import add_mp_parameters
from SuperQ_GRASP.Marching_Primitives.mesh2sdf_convert import mesh2sdf_csv
'''
Main Program of the whole grasp pose prediction module
using Marching Primitives to split the target object into sq's
'''

## The program to decompose the target mesh into several superquadrics
def preprocess_mesh(mesh_filename, csv_filename, options):
    """
    The function to the given mesh model
    Task:
        Decompose the target mesh model into primitives, if needed
    """
    # Split the mesh into primitives
    print("Splitting the Target Mesh (Marching Primitives)")
    # Split the target object into several primitives using Marching Primitives
    sq_predict = sq_predict_mp(csv_filename, options)
    normalize_stats = [1.0, 0.0]
    sq_vertices_original, sq_transformation = read_sq_mp(\
        sq_predict, norm_scale=1.0, norm_d=0.0)
    
    # Store the results in the local folder
    suffix = os.path.split(mesh_filename)[1].split(".")[0]
    stored_stats_filename = "./SuperQ_GRASP/Marching_Primitives/sq_data/" + suffix + ".p"
    store_mp_parameters(stored_stats_filename, \
                    sq_vertices_original, sq_transformation, normalize_stats)

    return stored_stats_filename
    
    

def grasp_pose_eval_gripper(mesh, sq_closest, grasp_poses, gripper_attr, \
                            csv_filename, visualization = False):
    '''
    The function to evaluate the predicted grasp poses on the target mesh
    
    Input:
    sq_closest: the target superquadric (the closest superquadric to the camera)
    grasp_poses: predicted grasp poses based on the superquadrics
    gripper_attr: attributes of the gripper
    Output: 
    bbox_cands, grasp_cands: meshes used in open3d for visualization purpose
    grasp_pose: the VALID grasp poses in world frame 
    (frame convention: the gripper's arm is along the positive x direction;
    the gripper's opening is along the z direction)
    '''
    if gripper_attr["Type"] == "Parallel":
        ## For parallel grippers, evaluate it based on antipodal metrics
        # Extract the attributes of the gripper
        gripper_width = gripper_attr["Width"]
        gripper_length = gripper_attr["Length"]
        gripper_thickness = gripper_attr["Thickness"]

        # Key points on the gripper
        num_sample = 20
        arm_end = np.array([gripper_length, 0, 0])
        center = np.array([0, 0, 0])
        elbow1 = np.array([0, 0, gripper_width/2])
        elbow2 = np.array([0, 0, -gripper_width/2])
        tip1 = np.array([-gripper_length, 0, gripper_width/2])
        tip2 = np.array([-gripper_length, 0, -gripper_width/2])
        
        # Construct the gripper
        vis_width = 0.004
        arm = o3d.geometry.TriangleMesh.create_cylinder(radius=vis_width, height=gripper_length)
        arm_rot = np.array([  [0.0000000,  0.0000000,  1.0000000],
        [0.0000000,  1.0000000,  0.0000000],
        [-1.0000000,  0.0000000,  0.0000000]])
        arm.rotate(arm_rot)
        arm.translate((gripper_length/2, 0, 0))
        hand = o3d.geometry.TriangleMesh.create_box(width=vis_width, depth=gripper_width, height=vis_width)
        hand.translate((0, 0, -gripper_width/2))
        finger1 = o3d.geometry.TriangleMesh.create_box(width=vis_width, depth=gripper_length, height=vis_width)
        finger2 = o3d.geometry.TriangleMesh.create_box(width=vis_width, depth=gripper_length, height=vis_width)
        finger_rot = np.array([  [0.0000000,  0.0000000,  -1.0000000],
        [0.0000000,  1.0000000,  0.0000000],
        [1.0000000,  0.0000000,  0.0000000]])
        finger1.rotate(finger_rot)
        finger2.rotate(finger_rot)
        finger1.translate((-gripper_length/2, 0, 0))
        finger2.translate((-gripper_length/2, 0, 0))
        finger1.translate((0, 0, gripper_width/2 - gripper_length/2))
        finger2.translate((0, 0, -gripper_width/2 - gripper_length/2))

        gripper = arm
        gripper += hand
        gripper += finger1
        gripper += finger2
        ## Part I: collision test preparation
        # Sample several points on the gripper
        gripper_part1 = np.linspace(arm_end, center, num_sample)
        gripper_part2 = np.linspace(elbow1, tip1, num_sample)
        gripper_part3 = np.linspace(elbow2, tip2, num_sample)
        gripper_part4 = np.linspace(elbow1, elbow2, num_sample)
        gripper_points_sample = np.vstack((gripper_part1, gripper_part2, gripper_part3, gripper_part4))

        # Add the thickness
        gripper_point_sample1 = copy.deepcopy(gripper_points_sample)
        gripper_point_sample1[:, 1] = -gripper_thickness/2
        gripper_point_sample2 = copy.deepcopy(gripper_points_sample)
        gripper_point_sample2[:, 1] = gripper_thickness/2

        # Stack all points together (points for collision test)
        gripper_points_sample = np.vstack((gripper_points_sample, gripper_point_sample1, gripper_point_sample2))
        
        ## Part II: collision test & antipodal test
        print("Evaluating Grasp Qualities....")
        grasp_cands = [] # All the grasp candidates
        bbox_cands = [] # Closing region of the gripper
        grasp_poses_world = []
        # Construct the grasp poses at the specified locations,
        # and add them to the visualizer optionally
        for grasp_pose in grasp_poses:
            # Find the grasp pose in the world frame (converted from sq local frame)
            grasp_pose = np.matmul(sq_closest["transformation"], grasp_pose)
            
            # Sample points for collision test
            gripper_points_vis_sample = np.vstack(\
                (gripper_points_sample.T, np.ones((1, gripper_points_sample.shape[0]))))
            gripper_points_vis_sample = np.matmul(grasp_pose, gripper_points_vis_sample)
            
            if visualization:
                # Transform the associated points for visualization or collision testing to the correct location
                grasp_pose_mesh = copy.deepcopy(gripper)
                grasp_pose_mesh = grasp_pose_mesh.rotate(R=grasp_pose[:3, :3], center=(0, 0, 0))
                grasp_pose_mesh = grasp_pose_mesh.translate(grasp_pose[0:3, 3])
                
            # Do the necessary testing jobs
            antipodal_res, bbox = antipodal_test(mesh, grasp_pose, gripper_attr, 5, np.pi/6)
            # collision_res, _, _ = collision_test_local(mesh, gripper_points_sample, \
            #                 grasp_pose, gripper_attr, 0.05 * gripper_width, scale = 1.5)
            collision_res = collision_test(mesh, gripper_points_vis_sample[:-1].T, threshold=0.03 * gripper_width)
            # collision_res = collision_test_sdf(csv_filename, gripper_points_vis_sample[:-1].T, threshold=0.05 * gripper_width)
            # Collision Test
            if collision_res:
                if visualization:
                    grasp_pose_mesh.paint_uniform_color((1, 0, 0))
            else: # Antipodal test
                if antipodal_res == True:
                    grasp_poses_world.append(grasp_pose)
                    if visualization:
                        bbox_cands.append(bbox)
                        grasp_pose_mesh.paint_uniform_color((0, 1, 0))
                else:
                    if visualization:
                        # Color them into yellow (no collision, but still invalid)
                        # grasp_pose_mesh.paint_uniform_color((235/255, 197/255, 28/255))
                        # Color them into red instead (stricter)
                        grasp_pose_mesh.paint_uniform_color((1, 0, 0))
            if visualization:
                grasp_cands.append(grasp_pose_mesh)
    

    return bbox_cands, grasp_cands, grasp_poses_world

def lie_algebra(gripper_pose):
    '''
    Convert a SE(3) pose to se(3)
    '''
    translation = gripper_pose[0:3, 3]
    omega = R.from_matrix(gripper_pose[:3, :3]).as_rotvec()
    x, y, z = omega
    theta = np.linalg.norm(omega)
    omega_hat = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])
    coeff = 1 - (theta * np.cos(theta/2))/(2*np.sin(theta/2))
    V_inv = np.eye(3) - (1/2) * omega_hat + (coeff / theta ** 2) * (omega_hat@omega_hat)
    tp = V_inv@translation.flatten()
    vw = np.hstack((tp, omega))
    assert vw.shape[0] == 6
    return vw

def record_gripper_pose_sq(gripper_pose, sq_parameters, filename="./record.txt"):
    '''
    The function to record the gripper pose and the associated selected index of sq
    from the demos
    '''
    f = open(filename, "a")
    vw = lie_algebra(gripper_pose)
    sq_xyz = sq_parameters["location"]
    line = str(vw[0]) + ", " + str(vw[1]) + ", " + str(vw[2]) + ", " + \
        str(vw[3]) + ", " + str(vw[4]) + ", " + str(vw[5]) + ", " + \
        str(sq_xyz[0]) + ", " + str(sq_xyz[1]) + ", " + str(sq_xyz[2]) + "\n"
    f.write(line)
    f.close()

def predict_grasp_pose_sq(mesh, csv_filename, \
                          normalize_stats, stored_stats_filename, \
                            gripper_attr, args):
    '''
    Input:
    mesh: the mesh of the target object
    csv_filename: name of the file storing the corresponding csv values
    normalize_stats: stats in normalizing the mesh (used by mesh2sdf)
    stored_stats_filename: pre-stored stats of the splitted superquadrics
    gripper_attr: dict of the attributes of gripper
    args: user arguments

    Output:
    grasp_poses_camera: the grasp poses in the camera frame 
    Equivalently, the relative transformations between the camera and the grasp poses
    '''
    ##################
    ## Part I: Split the mesh into several superquadrics
    ##################
    ## Read the parameters of the superquadrics
    os.path.isfile(stored_stats_filename)
    print("Reading pre-stored Superquadric Parameters...")
    sq_vertices_original, sq_transformation, normalize_stats = read_mp_parameters(\
                        stored_stats_filename)
        
    # Convert sq_verticies_original into a numpy array
    sq_vertices = np.array(sq_vertices_original).reshape(-1, 3)
    sq_centers = []
    for val in sq_transformation:
        sq_center = val["transformation"][0:3 , 3]
        sq_centers.append(sq_center)
    sq_centers = np.array(sq_centers)
    # Compute the convex hull
    pc_sq_centers= o3d.geometry.PointCloud()
    pc_sq_centers.points = o3d.utility.Vector3dVector(sq_centers)
    hull, hull_indices = pc_sq_centers.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    sq_associated = []
    
    # Iteratively find the grasp poses on the iterated superquadrics
    for idx in hull_indices:
        # NOTE: Originally, SuperQ_GRASP only finds grasp poses on the 
        # closest superquadric to the camera, which explains why the variable
        # is named as "sq_closest". 
        # 
        # But in the current implementation, we will iteratively 
        # find grasp poses on the superquadrics associated with the 
        # vertices on the convex hull.
        sq_closest = sq_transformation[idx]
        

        #######
        # Part II: Determine the grasp candidates on the selected sq and visualize them
        #######
        # Predict grasp poses around the target superquadric in LOCAL frame
        grasp_poses = grasp_pose_predict_sq_closest(sq_closest, gripper_attr, sample_number=50)
        # Evaluate the grasp poses w.r.t. the target mesh in WORLD frame
        bbox_cands, grasp_cands, grasp_poses_world = \
            grasp_pose_eval_gripper(mesh, sq_closest, grasp_poses, gripper_attr, \
                                    csv_filename, args.visualization)
        if args.visualization:
            sq_associated.append(sq_closest["points"])
        
    ## Postlogue - Visualization
    if args.visualization:
        
        # Construct a point cloud representing the reconstructed object mesh
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sq_vertices)
        # Visualize the super-ellipsoids
        pcd.paint_uniform_color((0.0, 0.5, 0))

        
        # Plot out the fundamental frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        frame.scale(20/64, [0, 0, 0])


        # Create the window to display everything
        vis= o3d.visualization.Visualizer()
        vis.create_window()
       
        
        vis.add_geometry(mesh)
        vis.add_geometry(pcd)
        
        for pcd_associated in sq_associated:
            vis.add_geometry(pcd_associated) 

        # vis.add_geometry(frame)

        vis.add_geometry(hull_ls)
        
        for grasp_cand in grasp_cands:
            vis.add_geometry(grasp_cand)
        for bbox_cand in bbox_cands:
            vis.add_geometry(bbox_cand)

        vis.run()

        # Close all windows
        vis.destroy_window()

        # Print out the validation results
        print("*******************")
        print("** Grasp pose Prediction Result: ")
        print("Selected Point in Space: ")
        print("Number of valid grasp poses predicted: " + str(len(grasp_poses_world)))
        print("*******************")

    return np.array(grasp_poses_world)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Implementation of SuperQ_GRASP to \
                predict grasp poses on top of the objects"
    )
    
    parser.add_argument(
        "--mesh_name",
        default = "target_obj.obj",
        help="The name of the mesh model to use"
    )
    """
    Arguments for Mesh Decomposition
    """
    # (split the target mesh into several superquadrics using Marching Primitives)
    parser.add_argument("--marching_cubes_res", \
                        default=256, type=int, \
                            help="Sets the resolution for the marching cubes grid.")
    parser.add_argument("--marching_cubes_density_thresh", \
                        default=2.5, type=float, \
                        help="Sets the density threshold for marching cubes.")
    parser.add_argument(
        '--grid_resolution', type=int, default=100,
        help='Set the resolution of the voxel grids in the order of x, y, z, e.g. 64 means 100^3.'
    )
    parser.add_argument(
        '--level', type=float, default=2,
        help='Set watertighting thicken level. By default 2'
    )

    parser.add_argument(
        '--decompose', action = 'store_true', help="Whether to re-decompose the mesh into superquadrics"
    )
    parser.add_argument(
        '--store', action = 'store_true', help="Whether to store the re-decomposed result"
    )
    
    
    # Visualization in open3d
    parser.add_argument(
        '--visualization', action = 'store_true', help="Whether to visualize the grasp poses"
    )
    add_mp_parameters(parser)
    parser.set_defaults(normalize=False)
    parser.set_defaults(decompose=False)
    parser.set_defaults(store=True)
    parser.set_defaults(visualization=True)


    ##########
    # Part I: Read mesh 
    ##########

    # Read the mesh file
    mesh_filename= os.path.join(parser.parse_args().mesh_name)
    mesh = o3d.io.read_triangle_mesh(mesh_filename)
    suffix = os.path.split(mesh_filename)[1].split(".")[0]
    
    # Read the csv file containing the sdf
    # (Theoretically, the csv file should have the same name as the mesh file,
    # but with the suffix changed to .csv; and they should be in the same folder)
    csv_filename = os.path.split(mesh_filename)[0] + "/" + os.path.split(mesh_filename)[1].split(".")[0] + ".csv"
    if not os.path.isfile(csv_filename):
        print("Converting mesh into SDF...")
        # If the csv file has not been generated, generate one
        _ = mesh2sdf_csv(mesh_filename, options = parser.parse_args())
    
    # Read the pre-stored decomposition results (if any)
    stored_stats_filename = "./SuperQ_GRASP/Marching_Primitives/sq_data/" + suffix + ".p"
    if not os.path.isfile(stored_stats_filename) or parser.parse_args().decompose:
        stored_stats_filename = preprocess_mesh(mesh, csv_filename)


    # The normalization is never used in SuperQ_GRASP, so just set it at the default value
    normalize_stats = [1.0, 0.0]
    
    ###############
    ## Part II: Predict Grasp poses
    ###############
    
    # Attributes of gripper
    gripper_width = 0.09
    gripper_length = 0.09 
    gripper_thickness = 0.089
    gripper_attr = {"Type": "Parallel", 
                    "Length": gripper_length,
                    "Width": gripper_width, 
                    "Thickness": gripper_thickness}
        
    predict_grasp_pose_sq(mesh, csv_filename, \
                          normalize_stats, 
                          stored_stats_filename, 
                          gripper_attr, 
                          parser.parse_args())