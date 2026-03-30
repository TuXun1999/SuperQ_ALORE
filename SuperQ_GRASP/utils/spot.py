#################################################################################
# Date: 03/11/2025
# Author of this Header: Xun Tu
#
# All helper functions for SPOT commands
###############################################################################
from scipy.optimize import fsolve

# Messages
from bosdyn.api import (robot_command_pb2,
                        geometry_pb2,
                        image_pb2,
                        mobility_command_pb2, 
                        basic_command_pb2,
                        manipulation_api_pb2,
                        arm_command_pb2)

from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.api.arm_command_pb2 import (ArmCommand, ArmCartesianCommand)
from bosdyn.api.basic_command_pb2 import (ArmDragCommand)
from bosdyn.api.robot_command_pb2 import (RobotCommand )
from bosdyn.api.synchronized_command_pb2 import (SynchronizedCommand)

from bosdyn.client import math_helpers, frame_helpers
from bosdyn.client.math_helpers import SE2Pose as bdSE2Pose
from bosdyn.client.math_helpers import SE3Pose as bdSE3Pose
from bosdyn.client.math_helpers import Quat    as bdQuat



# Clients
from bosdyn.client.manipulation_api_client import ManipulationApiClient

from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, 
                                         ODOM_FRAME_NAME, 
                                         HAND_FRAME_NAME,
                                         VISION_FRAME_NAME,
                                         GRAV_ALIGNED_BODY_FRAME_NAME,
                                         get_se2_a_tform_b,
                                         get_vision_tform_body,
                                         get_a_tform_b)

from bosdyn.client.robot_command import (block_until_arm_arrives, block_for_trajectory_cmd)
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_state import RobotStateClient

from bosdyn.client.robot_command import RobotCommandBuilder as CmdBuilder

# Others
from bosdyn.util import seconds_to_duration

import numpy as np
import time
from scipy.spatial.transform import Rotation as R


'''
Helper functions
'''
def pose_np_to_bd(pose:np.array, se3=False):
    '''Change input np array into bd pose'''
    if pose.shape == (3,3):
        pose = bdSE2Pose.from_matrix(pose)
        if se3: pose = pose.get_closest_se3_transform()
    else: 
        pose = bdSE3Pose.from_matrix(pose)
    return pose    

def pose_bd_to_vectors(pose:bdSE3Pose):
    '''Change into bd pose into vectors (pos, rot)'''
    pos = [pose.position.x, pose.position.y, pose.position.z]
    rot = [pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z]
    return pos, rot

def pose_bd_interpolate_linear(pose1:bdSE3Pose, pose2:bdSE3Pose, segment:int):
    '''Interpolate between two poses (linear)'''
    pos1, rot1 = pose_bd_to_vectors(pose1)
    pos2, rot2 = pose_bd_to_vectors(pose2)
    pos_seq = np.linspace(pos1, pos2, segment+1, endpoint=True)
    rot_seq = np.linspace(rot1, rot2, segment+1, endpoint=True)
    pose_seq = []
    for i in range(1, segment+1):
        w, x, y, z = rot_seq[i]
        pose_seq.append(bdSE3Pose(x=pos_seq[i][0], y=pos_seq[i][1], z=pos_seq[i][2], \
                    rot=bdQuat(w=w, x=x, y=y, z=z)))
    return pose_seq
def pose_bd_interpolate_rotZ(pose1:bdSE3Pose, rot_axis_origin, angle, segment:int):
    '''Interpolate poses along the rotation axis'''
    pose_initial = pose1.to_matrix()
    pose_seq = []
    # Find the current rotation transformation matrix
    K = np.eye(4)
    K[0:3, 3] = rot_axis_origin
    for i in range(1, segment+1):
        rot_angle = angle * i / segment
        rot = R.from_quat([0, 0, np.sin(rot_angle/2), np.cos(rot_angle/2)])
        rot_tran = np.eye(4)
        rot_tran[0:3, 0:3] = rot.as_matrix()
        pose_traj = K @ rot_tran @ np.linalg.inv(K) @ pose_initial
        pose_traj = bdSE3Pose.from_matrix(pose_traj)
        pose_seq.append(pose_traj)
    return pose_seq

def offset_pose(pose:bdSE3Pose, distance, axis):
    '''Calculate the offset along the direction defined by axis'''
    all_axis = {'x':0, 'y':1, 'z':2}
    dir = np.eye(3)[all_axis[axis]] 
    offset = np.eye(4)
    offset[0:3, 3] = -dir * distance
    offset = bdSE3Pose.from_matrix(offset)
    return pose * offset
def transform_bd_pose(robot_state_client, pose, reference_frame:str, target_frame:str):
    '''Transform the pose from one frame to another'''
    Tf_tree = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    if isinstance(pose, bdSE2Pose):
        T = get_se2_a_tform_b(Tf_tree, target_frame, reference_frame)
    elif isinstance(pose, bdSE3Pose):
        T = get_a_tform_b(Tf_tree, target_frame, reference_frame)
    else:
        raise ValueError("pose must be either bdSE2Pose or bdSE3Pose")
    return T * pose

def to_bd_se3(robot_state_client, pose, ref, se3=True, ref_frame=ODOM_FRAME_NAME):
    '''Changes input pose into a bd se3 pose.'''
    if isinstance(pose, np.ndarray):
        pose = pose_np_to_bd(pose, se3=se3)
    pose = transform_bd_pose(robot_state_client, pose, ref, ref_frame)
    return pose

def get_gripper_initial_pose(y_offset, frame_name = "odom"):
    '''Get the initial pose of the gripper w/o any offset defined by y_offset'''
    return transform_bd_pose(\
        bdSE3Pose(0, y_offset, 0, bdQuat()), HAND_FRAME_NAME, frame_name)
'''
By assuming a relative fixed transformation between gripper & body
Predict one given the other
'''
def predict_gripper_pose(robot_state_client, spot_target_pose, \
                         target_ref_frame_name = GRAV_ALIGNED_BODY_FRAME_NAME):
    # spot_target_pose: the target pose for the spot to move to
    # target_ref_frame_name: the reference frame where the target pose is specified
    Tf_tree = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    g = get_a_tform_b(Tf_tree, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)

    if (target_ref_frame_name != GRAV_ALIGNED_BODY_FRAME_NAME):
        T = get_a_tform_b(Tf_tree, GRAV_ALIGNED_BODY_FRAME_NAME, target_ref_frame_name)
        # In body frame, how the spot changes its orientation
        spot_target_pose_body = T * spot_target_pose

        # The gripper tareget pose in current body frame
        gripper_target_pose = spot_target_pose_body * g

        # Convert the gripper target pose back in the reference frame of spot_target_pose
        gripper_target_pose = T.inverse() * gripper_target_pose
    else:
        gripper_target_pose = spot_target_pose * g

    return gripper_target_pose

def predict_body_pose(robot_state_client, gripper_target_pose, \
                         target_ref_frame_name = HAND_FRAME_NAME):
    # spot_target_pose: the target pose for the spot to move to
    # target_ref_frame_name: the reference frame where the target pose is specified
    Tf_tree = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    g = get_a_tform_b(Tf_tree, HAND_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

    if (target_ref_frame_name != HAND_FRAME_NAME):
        T = get_a_tform_b(Tf_tree, HAND_FRAME_NAME, target_ref_frame_name)
        # In hand frame, how the gripper changes its orientation
        spot_target_pose_body = T * gripper_target_pose

        # The gripper tareget pose in body frame
        spot_target_pose =  spot_target_pose_body * g

        # Convert the gripper target pose back in the reference frame of spot_target_pose
        spot_target_pose = T.inverse() * spot_target_pose
    else:
        spot_target_pose = gripper_target_pose * g

    return spot_target_pose

def robot_command(robot_command_client, command_proto, end_time_secs=None, timesync_endpoint=None):
    """Generic blocking function for sending commands to robots.

    Args:
        command_proto: robot_command_pb2 object to send to the robot.  Usually made with RobotCommandBuilder
        end_time_secs: (optional) Time-to-live for the command in seconds
        timesync_endpoint: (optional) Time sync endpoint
    """
    try:
        id = robot_command_client.robot_command(
            lease=None,
            command=command_proto,
            end_time_secs=end_time_secs,
            timesync_endpoint=timesync_endpoint,
        )
        return True, "Success", id
    except Exception as e:
        return False, str(e), None






'''
Move the arm to the target pose in the reference frame
using follow_arm_to command
'''
def follow_arm_to(robot_command_client, pose, reference_frame):

    pose = to_bd_se3(pose, reference_frame)

    # Create the arm & body command.
    arm_command = CmdBuilder.arm_pose_command(
        pose.x, pose.y, pose.z, 
        pose.rot.w, pose.rot.x, pose.rot.y, pose.rot.z, 
        reference_frame, 
        2.0)
    follow_arm_command = CmdBuilder.follow_arm_command()
    command = CmdBuilder.build_synchro_command(follow_arm_command, 
                                                arm_command)

    # Send the request
    cmd_client = robot_command_client
    move_command_id = cmd_client.robot_command(command)
    print('Moving arm to position.')

    block_until_arm_arrives(cmd_client, move_command_id, 6.0)
    print('Succeeded')
    return True

'''
Send the trajectory command to the robot
'''
def trajectory_cmd(
        goal_x,
        goal_y,
        goal_heading,
        cmd_duration,
        robot_state_client,
        robot_command_client,
        reference_frame=BODY_FRAME_NAME,
        frame_name="odom",
        build_on_command=None,
        trajectory_params = None,
    ):
        """Send a trajectory motion command to the robot.

        Args:
            goal_x: Position X coordinate in meters
            goal_y: Position Y coordinate in meters
            goal_heading: Pose heading in radians
            cmd_duration: Time-to-live for the command in seconds.
            frame_name: frame_name to be used to calc the target position. 'odom' or 'vision'
            precise_position: if set to false, the status STATUS_NEAR_GOAL and STATUS_AT_GOAL will be equivalent. If
            true, the robot must complete its final positioning before it will be considered to have successfully
            reached the goal.
            reference_frame: The frame in which the goal is represented

        Returns: (bool, str) tuple indicating whether the command was successfully sent, and a message
        """
        
        if frame_name not in [ODOM_FRAME_NAME, VISION_FRAME_NAME]: 
            raise ValueError("frame_name must be 'vision' or 'odom'")
        print("Recieved Pose Trajectory Command.")
        
        end_time = time.time() + cmd_duration

        T_in_ref = bdSE2Pose(x=goal_x, y=goal_y, angle=goal_heading)
        T_in_target = transform_bd_pose(robot_state_client, T_in_ref, reference_frame, frame_name)

        if trajectory_params == None:
            trajectory_params = RobotCommandBuilder.mobility_params()
        
        response = robot_command(
                robot_command_client,
                RobotCommandBuilder.synchro_se2_trajectory_point_command(
                    goal_x=T_in_target.x,
                    goal_y=T_in_target.y,
                    goal_heading=T_in_target.angle,
                    frame_name=frame_name,
                    params=trajectory_params,
                    build_on_command=build_on_command,
                ),
                end_time_secs=end_time,
            )

        return response[0], response[1], response[2]


'''
Estimate the object pose w.r.t the gripper/hand
'''
def estimate_obj_pose_hand(bbox, image_response, distance):
    ## Estimate the target object pose (indicated by the bounding box) in hand frame
    bbox_center = [int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3])/2)]
    pick_x, pick_y = bbox_center
    # Obtain the camera inforamtion
    camera_info = image_response.source
    
    w = camera_info.cols
    h = camera_info.rows
    fl_x = camera_info.pinhole.intrinsics.focal_length.x
    k1= camera_info.pinhole.intrinsics.skew.x
    cx = camera_info.pinhole.intrinsics.principal_point.x
    fl_y = camera_info.pinhole.intrinsics.focal_length.y
    k2 = camera_info.pinhole.intrinsics.skew.y
    cy = camera_info.pinhole.intrinsics.principal_point.y

    pinhole_camera_proj = np.array([
        [fl_x, 0, cx, 0],
        [0, fl_y, cy, 0],
        [0, 0, 1, 0]
    ])
    pinhole_camera_proj = np.float32(pinhole_camera_proj) # Converted into float type
    # Calculate the object's pose in hand camera frame
    initial_guess = [1, 1, 10]
    def equations(vars):
        x, y, z = vars
        eq = [
            pinhole_camera_proj[0][0] * x + pinhole_camera_proj[0][1] * y + pinhole_camera_proj[0][2] * z - pick_x * z,
            pinhole_camera_proj[1][0] * x + pinhole_camera_proj[1][1] * y + pinhole_camera_proj[1][2] * z - pick_y * z,
            x * x + y * y + z * z - distance * distance
        ]
        return eq

    root = fsolve(equations, initial_guess)
    # Correct the frame conventions in hand frame & pinhole model
    # pinhole model: z-> towards object, x-> rightward, y-> downward
    # hand frame in SPOT: x-> towards object, y->rightward
    result = bdSE3Pose(x=root[2], y=-root[0], z=-root[1], rot=bdQuat(w=1, x=0, y=0, z=0))
    return result

'''
Helper function compute the stand location and yaw
'''
def compute_stand_location_and_yaw(vision_tform_target, robot_state_client,
                                distance_margin):

    # Compute drop-off location:
    #   Draw a line from Spot to the person
    #   Back up 2.0 meters on that line
    vision_tform_robot = frame_helpers.get_a_tform_b(
        robot_state_client.get_robot_state(
        ).kinematic_state.transforms_snapshot, frame_helpers.VISION_FRAME_NAME,
        frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME)


    # Compute vector between robot and person
    robot_rt_person_ewrt_vision = [
        vision_tform_robot.x - vision_tform_target.x,
        vision_tform_robot.y - vision_tform_target.y,
        vision_tform_robot.z - vision_tform_target.z
    ]


    # Compute the unit vector.
    if np.linalg.norm(robot_rt_person_ewrt_vision) < 0.01:
        robot_rt_person_ewrt_vision_hat = vision_tform_robot.transform_point(1, 0, 0)
    else:
        robot_rt_person_ewrt_vision_hat = robot_rt_person_ewrt_vision / np.linalg.norm(
            robot_rt_person_ewrt_vision)


    # Starting at the person, back up meters along the unit vector.
    drop_position_rt_vision = [
        vision_tform_target.x +
        robot_rt_person_ewrt_vision_hat[0] * distance_margin,
        vision_tform_target.y +
        robot_rt_person_ewrt_vision_hat[1] * distance_margin,
        vision_tform_target.z +
        robot_rt_person_ewrt_vision_hat[2] * distance_margin
    ]


    # We also want to compute a rotation (yaw) so that we will face the person when dropping.
    # We'll do this by computing a rotation matrix with X along
    #   -robot_rt_person_ewrt_vision_hat (pointing from the robot to the person) and Z straight up:
    xhat = -robot_rt_person_ewrt_vision_hat
    zhat = [0.0, 0.0, 1.0]
    yhat = np.cross(zhat, xhat)
    mat = np.matrix([xhat, yhat, zhat]).transpose()
    heading_rt_vision = math_helpers.Quat.from_matrix(mat).to_yaw()

    return drop_position_rt_vision, heading_rt_vision

'''
Helper function to get the mobility params under the velocity limit
'''
def get_walking_params(max_linear_vel, max_rotation_vel):
    max_vel_linear = geometry_pb2.Vec2(x=max_linear_vel, y=max_linear_vel)
    max_vel_se2 = geometry_pb2.SE2Velocity(linear=max_vel_linear,
                                        angular=max_rotation_vel)
    vel_limit = geometry_pb2.SE2VelocityLimit(max_vel=max_vel_se2)
    params = RobotCommandBuilder.mobility_params()
    params.vel_limit.CopyFrom(vel_limit)
    return params


def move_gripper(grasp_pose_obj, robot_state_client, command_client):
    '''
    Move the gripper to the desired pose, given in the local frame of HAND_frame
    Input:
    grasp_pose_obj: the pose of the gripper in the current hand frame
    robot_state_client: the state client of the robot
    command_client: the client to send the commands
    '''

    # Transform the desired pose from the moving body frame to the odom frame.
    robot_state = robot_state_client.get_robot_state()
    odom_T_hand = frame_helpers.get_a_tform_b(\
                robot_state.kinematic_state.transforms_snapshot,
                frame_helpers.ODOM_FRAME_NAME, \
                frame_helpers.HAND_FRAME_NAME)
    odom_T_hand = odom_T_hand.to_matrix()

    # Build the SE(3) pose of the desired hand position in the moving body frame.
    odom_T_target = odom_T_hand@grasp_pose_obj
    odom_T_target = math_helpers.SE3Pose.from_matrix(odom_T_target)
    
    # duration in seconds
    seconds = 5.0

    # Create the arm command.
    arm_command = RobotCommandBuilder.arm_pose_command(
        odom_T_target.x, odom_T_target.y, odom_T_target.z, odom_T_target.rot.w, odom_T_target.rot.x,
        odom_T_target.rot.y, odom_T_target.rot.z, \
            frame_helpers.ODOM_FRAME_NAME, seconds)

    # Tell the robot's body to follow the arm
    follow_arm_command = RobotCommandBuilder.follow_arm_command()

    # Combine the arm and mobility commands into one synchronized command.
    command = RobotCommandBuilder.build_synchro_command(follow_arm_command, arm_command)

    # Send the request
    move_command_id = command_client.robot_command(command)
    print('Moving arm to position.')

    block_until_arm_arrives(command_client, move_command_id, 30.0)


'''
Move the robot to the desired pose, while gripper attached to the object
No force within the arm joints
'''
def drag_arm_to(robot_state_client, robot_command_client, pose:bdSE3Pose, reference_frame, duration_sec=15.0):
    '''Commands the robot to position the robot at the commanded
    pose while leaving the arm compliant thus allowing it to drag
    objects.
    Args:
        pose: The desired pose of the robot.
        reference_frame: The frame in which the pose is defined.
    returns:
            '''
    print(f'Building drag command.')
    robot_cmd = RobotCommand(
        synchronized_command = SynchronizedCommand.Request(
            arm_command = ArmCommand.Request(
                arm_drag_command = ArmDragCommand.Request()
            )
        )
    )
    # Set the claw to apply force        
    robot_cmd = CmdBuilder.claw_gripper_close_command(robot_cmd) 

    pose = pose.get_closest_se2_transform()
    print(f'Sending Robot Command.')
    succeeded, e, id = trajectory_cmd(
        goal_x=pose.x, goal_y=pose.y,
        goal_heading=pose.angle,
        cmd_duration=duration_sec, 
        robot_state_client=robot_state_client,
        reference_frame=reference_frame,
        build_on_command=robot_cmd
    )
    if succeeded:
        block_for_trajectory_cmd(robot_command_client,
                                    cmd_id=id, 
                                    feedback_interval_secs=0.5, 
                                    timeout_sec=duration_sec,
                                    logger=None)
    else: 
        print('Failed to send trajectory command.')
        print(e)
        return False
    return True


'''
Move the obstacle to the pose, defined for the gripper
'''
def drag_arm_impedance(robot_state_client, robot_command_client, \
                       gripper_target_pose:bdSE3Pose, spot_target_pose:bdSE3Pose,
                                reference_frame = BODY_FRAME_NAME, duration_sec=15.0):
    '''Commands the robot arm to perform an impedance command
    which allows it to move heavy objects or perform surface 
    interaction.
    This will force the robot to stand and support the arm. 
    Args:
        pose: The desired end-effector pose.
        reference_frame: The frame in which the pose is defined.
        duration_sec: The max duration given for the command to conclude.
    Returns:
        True if the command succeeded, False otherwise.
    TODO:
        - Try to apply and retain force on the gripper
        - If no object is in hand, raise an error
        - Try work with stiffness
        - Try make the relative pose between the arm and the robot constant
        
    Reference: Spot sdk python examples: arm_impedance_control.py'''

    
    print(f'Building Impedance Cmd')

    '''
    Part I: Build up the arm impedance control cmd 
    '''
    stand_command = get_body_assist_stance_command()
    
    gripper_arm_cmd = robot_command_pb2.RobotCommand()
    gripper_arm_cmd.CopyFrom(stand_command)  # Make sure we keep adjusting the body for the arm
    impedance_cmd = gripper_arm_cmd.synchronized_command.arm_command.arm_impedance_command
    print("Start building impedance cmd")
    # Set up our root frame; task frame, and tool frame are set by default
    impedance_cmd.root_frame_name = reference_frame

    # Set up stiffness and damping matrices. Note: if these values are set too high,
    # the arm can become unstable. Currently, these are the max stiffness and
    # damping values that can be set.

    # NOTE: Max stiffness: [500, 500, 500, 60, 60, 60]
    #      Max damping: [2.5, 2.5, 2.5, 1.0, 1.0, 1.0]
    impedance_cmd.diagonal_stiffness_matrix.CopyFrom(
        geometry_pb2.Vector(values=[500, 500, 500, 60, 60, 60]))
    impedance_cmd.diagonal_damping_matrix.CopyFrom(
        geometry_pb2.Vector(values=[2.0, 2.0, 2.0, 0.55, 0.55, 0.55]))

    # Set up our `desired_tool` trajectory. This is where we want the tool to be with respect
    # to the task frame. The stiffness we set will drag the tool towards `desired_tool`.
    traj = impedance_cmd.task_tform_desired_tool
    pt1 = traj.points.add()
    pt1.time_since_reference.CopyFrom(seconds_to_duration(5.0))
    pt1.pose.CopyFrom(gripper_target_pose.to_proto())


    print("Build up the arm impedance command")



    # Set the claw to apply force        
    gripper_arm_cmd = CmdBuilder.claw_gripper_close_command(gripper_arm_cmd) 
    # NOTE: in some places more claw pressure helps. The command below
    #       fails. Need to find alternatives.
    # robot_cmd.gripper_command.claw_gripper_command.maximum_torque = 8

    '''
    Part II: send the trajectory command based on the arm&gripper command
    '''
    '''
        # Execute the impedance command
    cmd_id = self.spot._robot_command_client.robot_command(gripper_arm_cmd)
    succeeded = block_until_arm_arrives(self.spot._robot_command_client, 
                                        cmd_id, self._log,
                                        timeout_sec=duration_sec)
    return succeeded
    '''
    drag_trajectory_params = get_walking_params(0.6, 1)

    # drag_trajectory_params.vel_limit.min_vel.linear.x *= 0.8
    # drag_trajectory_params.vel_limit.min_vel.linear.y *= 0.8
    pose = spot_target_pose.get_closest_se2_transform()
    print(f'Sending the drag_arm_impedance command.')
    succeeded, _, id = trajectory_cmd(
        goal_x=pose.x, goal_y=pose.y,
        goal_heading=pose.angle,
        cmd_duration=duration_sec, 
        robot_state_client=robot_state_client,
        robot_command_client=robot_command_client,
        reference_frame=reference_frame,
        build_on_command=gripper_arm_cmd,
        trajectory_params = drag_trajectory_params
    )
    if succeeded:
        block_for_trajectory_cmd(robot_command_client,
                                    cmd_id=id, 
                                    feedback_interval_secs=0.5, 
                                    timeout_sec=duration_sec,
                                    logger=None)
    else: 
        print('Failed to send trajectory command.')
        return False
    return True

def get_body_assist_stance_command():
    '''A assistive stance is used when manipulating heavy
    objects or interacting with the environment. 
    
    Returns: A body assist stance command'''
    body_control = spot_command_pb2.BodyControlParams(
        body_assist_for_manipulation=spot_command_pb2.BodyControlParams.
        BodyAssistForManipulation(enable_hip_height_assist=True, enable_body_yaw_assist=False))
    
    
    stand_command = CmdBuilder.synchro_stand_command(
        params=spot_command_pb2.MobilityParams(body_control=body_control))
    return stand_command



def joint_mobility_arm_cmd(self, pose, reference_frame):
    '''Command the arm to a certain pose and request that the 
    body follows. This does not expect there to be any contact.'''
    raise NotImplementedError('Joint-level API is not currently supported by the program.')


def move_heavy_object(robot_state_client, robot_command_client, \
                      gripper_target_pose, spot_target_pose, reference_frame, \
                        target_frame=ODOM_FRAME_NAME):
    '''Commands the robot to move an object to a desired pose.
    This involves dragging the object near the goal pose, then
    moving the arm to properly position it.
    
    Ideally this would be done using the ArmImpedanceCommand, but
    this is not currently supported by the robot.
    Instead ArmDragCommand has to first be used followed by 
    Arm impedance command.'''
    print('Moving a heavy object.')
    gripper_pose = to_bd_se3(robot_state_client, gripper_target_pose, reference_frame, ref_frame=target_frame)
    spot_pose = to_bd_se3(robot_state_client, spot_target_pose, reference_frame, ref_frame=target_frame)
    print("===Target Location===")
    print(gripper_pose)
    print(spot_pose)
    # Drag the object to the desired location
    drag_arm_impedance(robot_state_client, robot_command_client, gripper_pose, \
                    spot_pose, target_frame, duration_sec=10.0)
    time.sleep(4)




