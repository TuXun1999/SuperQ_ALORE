
import time
import cv2
from typing import Optional
import numpy as np
from requests import options
from scipy import ndimage
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation as R
from tkinter import *
import os 
import math
import apriltag
"""Import Boston Dynamics libraries"""
import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.api import manipulation_api_pb2, gripper_camera_param_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.robot_command import RobotCommandBuilder as CmdBuilder
from google.protobuf import wrappers_pb2 as wrappers
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.gripper_camera_param import GripperCameraParamClient
from bosdyn.client.robot_command import \
    RobotCommandBuilder, RobotCommandClient, \
        block_until_arm_arrives, block_for_trajectory_cmd, blocking_stand
from bosdyn.client.math_helpers import SE3Pose, Quat
from bosdyn.client.math_helpers import SE2Pose as bdSE2Pose
from bosdyn.client.math_helpers import SE3Pose as bdSE3Pose
import bosdyn.client.math_helpers as math_helpers
from bosdyn.client import frame_helpers
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, 
                                         VISION_FRAME_NAME, 
                                         HAND_FRAME_NAME,
                                         ODOM_FRAME_NAME,
                                         GRAV_ALIGNED_BODY_FRAME_NAME,
                                         get_se2_a_tform_b,
                                         get_vision_tform_body,
                                         get_a_tform_b,
                                         get_odom_tform_body)

import bosdyn.api.power_pb2 as PowerServiceProto
import bosdyn.api.robot_state_pb2 as robot_state_proto
import bosdyn.client.util
from bosdyn.api import arm_command_pb2, geometry_pb2, robot_command_pb2, \
    world_object_pb2, synchronized_command_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.async_tasks import AsyncPeriodicQuery
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.time_sync import TimeSyncError
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.util import duration_str, format_metric, secs_to_hms, seconds_to_duration
from bosdyn.api.graph_nav import map_pb2, map_processing_pb2, recording_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.map_processing import MapProcessingServiceClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
import torch
from PIL import Image
# Hyperparameters for SPOT
VELOCITY_CMD_DURATION = 0.5  # seconds
COMMAND_INPUT_RATE = 0.1
VELOCITY_HAND_NORMALIZED = 0.5  # normalized hand velocity [0,1]
VELOCITY_ANGULAR_HAND = 1.0  # rad/sec

ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180,
    'hand_color_image': 0
}

# Hyperparameters for RL for SPOT
DEFAULT_X = 2.0
DEFAULT_ANGLE = np.pi

"""
Helper functions
"""
def pixel_format_type_strings():
    names = image_pb2.Image.PixelFormat.keys()
    return names[1:]


def pixel_format_string_to_enum(enum_string):
    return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)


def hand_image_resolution(gripper_param_client, resolution):
    camera_mode = None
    if resolution is not None:
        if resolution == '640x480':
            camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_640_480
        elif resolution == '1280x720':
            camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_1280_720
        elif resolution == '1920x1080':
            camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_1920_1080
        elif resolution == '3840x2160':
            camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_3840_2160
        elif resolution == '4096x2160':
            camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_4096_2160
        elif resolution == '4208x3120':
            camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_4208_3120

    request = gripper_camera_param_pb2.GripperCameraParamRequest(
        params=gripper_camera_param_pb2.GripperCameraParams(camera_mode=camera_mode))
    response = gripper_param_client.set_camera_params(request)



class SPOT:
    def __init__(self, options):
		# Create robot object with an image client.
        sdk = bosdyn.client.create_standard_sdk('rl_policy_deployment')
        self.robot = sdk.create_robot(options.hostname)
        bosdyn.client.util.authenticate(self.robot)
        self.robot.sync_with_directory()
        self.robot.time_sync.wait_for_sync()

        self.gripper_param_client = self.robot.ensure_client(\
            GripperCameraParamClient.default_service_name)
        # Optionally set the resolution of the hand camera
        if hasattr(options, 'resolution') and 'hand_color_image' in options.image_sources:
            hand_image_resolution(self.gripper_param_client, options.resolution)
        
        if hasattr(options, 'image_service'):
            self.image_client = self.robot.ensure_client(options.image_service)
        else:
            self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
        
        self.image_source_names = [
            src.name for src in self.image_client.list_image_sources() if
            (src.image_type == image_pb2.ImageSource.IMAGE_TYPE_VISUAL and 'depth' not in src.name)
        ]
        self.image_source_names.append("hand_color_image")
        # Clients
        self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.lease_client = self.robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        
        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.world_object_client = self.robot.ensure_client(WorldObjectClient.default_service_name)
        # Verification before the formal task
        assert self.robot.has_arm(), 'Robot requires an arm to run this example.'
        # Verify the robot is not estopped and that an external application has registered and holds
        # an estop endpoint.
        assert not self.robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                        'such as the estop SDK example, to configure E-Stop.'
    
    """Section I: Fundamental functionalities"""
    def lease_alive(self):
        self._lease_alive = bosdyn.client.lease.LeaseKeepAlive(\
            self.lease_client, must_acquire=True, return_at_exit=True)
        return self._lease_alive
    def lease_return(self):
        self._lease_alive.shutdown()
        self._lease_alive = None
        return self._lease_alive
    def power_on_stand(self):
        # Power on the robot
        self.robot.power_on(timeout_sec=20)
        assert self.robot.is_powered_on(), 'Robot power on failed.'

        
        blocking_stand(self.command_client, timeout_sec=10)
        self.robot.logger.info('Robot standing.')
        # Command the robot to open its gripper
        robot_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1)

        # Send the trajectory to the robot.
        cmd_id = self.command_client.robot_command(robot_command)

        time.sleep(3)
    def get_walking_params(self, max_linear_vel, max_rotation_vel):
        max_vel_linear = geometry_pb2.Vec2(x=max_linear_vel, y=max_linear_vel)
        max_vel_se2 = geometry_pb2.SE2Velocity(linear=max_vel_linear,
                                            angular=max_rotation_vel)
        vel_limit = geometry_pb2.SE2VelocityLimit(max_vel=max_vel_se2)
        params = RobotCommandBuilder.mobility_params()
        params.vel_limit.CopyFrom(vel_limit)
        return params
    def _start_robot_command(self, desc, command_proto, end_time_secs=None):

        def _start_command():
            self.command_client.robot_command(command=command_proto,
                                                     end_time_secs=end_time_secs)

        self._try_grpc(desc, _start_command)
    def _try_grpc(self, desc, thunk):
        try:
            return thunk()
        except (ResponseError, RpcError, LeaseBaseError) as err:
            print(f'Failed {desc}: {err}')
            return None
    @property
    def robot_state(self):
        """Get latest robot state proto."""
        return self.state_client.get_robot_state()
    
    def get_desired_angle(self, xhat):
        """Compute heading based on the vector from robot to object."""
        zhat = [0.0, 0.0, 1.0]
        yhat = np.cross(zhat, xhat)
        mat = np.array([xhat, yhat, zhat]).transpose()
        return Quat.from_matrix(mat).to_yaw()
    
    def power_off(self):
        # Power off the robot
        self.robot.power_off()
        assert not self.robot.is_powered_on(), 'Robot power off failed.'
    
    
    """Section II: Image services"""
    def list_image_sources(self):
        image_sources = self.image_client.list_image_sources()
        print('Image sources:')
        for source in image_sources:
            print('\t' + source.name)
    def rotate_image(self, img, source_name):
        img = ndimage.rotate(img, ROTATION_ANGLE[source_name])
        return img
    def capture_images(self, options):
        # Capture and save images to disk
        pixel_format = pixel_format_string_to_enum(options.pixel_format)
        image_request = [
            build_image_request(source, pixel_format=pixel_format)
            for source in options.image_sources
        ]
        image_responses = self.image_client.get_image(image_request)
        images = []
        image_extensions = []
        for image in image_responses:
            num_bytes = 1  # Assume a default of 1 byte encodings.
            if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                dtype = np.uint16
                extension = '.png'
            else:
                if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                    num_bytes = 3
                elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
                    num_bytes = 4
                elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                    num_bytes = 1
                elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
                    num_bytes = 2
                dtype = np.uint8
                extension = '.jpg'

            img = np.frombuffer(image.shot.image.data, dtype=dtype)
            if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                try:
                    # Attempt to reshape array into a RGB rows X cols shape.
                    img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_bytes))
                except ValueError:
                    # Unable to reshape the image data, trying a regular decode.
                    img = cv2.imdecode(img, -1)
            else:
                img = cv2.imdecode(img, -1)

            if options.auto_rotate:
                img = self.rotate_image(img, image.source.name)

            # Append the image to the list
            images.append(img)
            image_extensions.append(extension)
        # # Save the image from the GetImage request to the current directory with the filename
        # # matching that of the image source.
        # image_saved_path = image.source.name
        # image_saved_path = image_saved_path.replace(
        #     '/', '')  # Remove any slashes from the filename the image is saved at locally.
        # cv2.imwrite(image_saved_path + custom_tag + extension, img)
        return image_responses, images, image_extensions
    
    
    """
    Section III: Base movement services
    """
    def get_base_pose_se2(self, frame_name = ODOM_FRAME_NAME):
        # The function to get the robot's base pose in SE2
        robot_state = self.robot_state
        odom_T_base = frame_helpers.get_a_tform_b(\
            robot_state.kinematic_state.transforms_snapshot, frame_name, GRAV_ALIGNED_BODY_FRAME_NAME)
        return odom_T_base.get_closest_se2_transform()
        
    
    def send_velocity_command_se2(self, vx, vy, vtheta, exec_time = 1.0, build_on_command = None):
        # The function to send the se2 synchro velocity command to the robot
        move_cmd = RobotCommandBuilder.synchro_velocity_command(\
            v_x=vx, v_y=vy, v_rot=vtheta,\
            params=self.get_walking_params(0.6, 1), \
            build_on_command=build_on_command)
        cmd_id = self.command_client.robot_command(command=move_cmd,\
                            end_time_secs=time.time() + exec_time)
        # Wait until the robot reports that it is at the goal.
        block_for_trajectory_cmd(self.command_client, cmd_id, timeout_sec=exec_time + 0.5)
    
    def send_pose_command_se2(self, x, y, theta, exec_time = 1.5, frame_name = ODOM_FRAME_NAME):
        # The function to send the pose command to move the robot to the desired pose
        move_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(\
            goal_x=x, goal_y=y, goal_heading=theta, \
                frame_name=frame_name, \
                params=self.get_walking_params(0.6, 1)
            )
        cmd_id = self.command_client.robot_command(command=move_cmd,\
                                end_time_secs = time.time() + exec_time)
        # Wait until the robot reports that it is at the goal.
        block_for_trajectory_cmd(self.command_client, cmd_id, timeout_sec=exec_time + 0.5)

    
    """Section IV: Arm services"""
    def get_hand_pose(self, reference_frame = BODY_FRAME_NAME):
        # Get the pose of the hand in the body frame
        robot_state = self.robot_state
        hand_T_body = frame_helpers.get_a_tform_b(\
            robot_state.kinematic_state.transforms_snapshot, reference_frame, HAND_FRAME_NAME)
        return hand_T_body
    def get_hand_pose_se2(self, reference_frame = BODY_FRAME_NAME):
        # Get the pose of the hand in the body frame
        robot_state = self.state_client.get_robot_state()
        hand_T_body = frame_helpers.get_a_tform_b(\
            robot_state.kinematic_state.transforms_snapshot, reference_frame, HAND_FRAME_NAME)
        return hand_T_body.get_closest_se2_transform()
    
    def open_gripper(self):
        # Open the gripper
        # Command the robot to open its gripper
        robot_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1)
        # Send the trajectory to the robot.
        cmd_id = self.command_client.robot_command(robot_command)
        time.sleep(0.5)
        
    def close_gripper(self):
        # Close the gripper
        # Command the robot to close its gripper
        robot_command = RobotCommandBuilder.claw_gripper_open_fraction_command(0)
        cmd_id = self.command_client.robot_command(robot_command)
        time.sleep(0.5)
    
    def make_arm_joint_freeze_command(self):
        return RobotCommandBuilder.arm_joint_freeze_command()
    
    def get_arm_joint_angles(self):
        # Get the current arm joint angles
        robot_state = self.state_client.get_robot_state()
        joint_angles = robot_state.kinematic_state.joint_states
        print(joint_angles)
        arm_joint_angles = []
        joint_name_list = ["arm0.sh0", "arm0.sh1", "arm0.el0", "arm0.el1", "arm0.wr0", "arm0.wr1", "arm0.f1x"]
        for joint_angle in joint_angles:
            if joint_angle.name in joint_name_list:
                arm_joint_angles.append(joint_angle.position.value)
        return arm_joint_angles
    def make_robot_arm_joint_control_command(self, arm_joint_traj):
        """ Helper function to create a RobotCommand from an ArmJointTrajectory.
            The returned command will be a SynchronizedCommand with an ArmJointMoveCommand
            filled out to follow the passed in trajectory. """

        joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_traj)
        arm_command = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=joint_move_command)
        sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)
        arm_sync_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_arm)
        return RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)
    def make_robot_arm_joint_command_displacement(self, displacement):
        """ Helper function to create a RobotCommand from an ArmJointTrajectory.
            However, the trajectory is specified as a displacement. """
        joint_angles = self.get_arm_joint_angles()
        arm_joint_target = torch.tensor(joint_angles) + displacement.squeeze()
        sh0 = arm_joint_target[0]
        sh1 = arm_joint_target[1]
        el0 = arm_joint_target[2]
        el1 = arm_joint_target[3]
        wr0 = arm_joint_target[4]
        wr1 = arm_joint_target[5]

        traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
            sh0, sh1, el0, el1, wr0, wr1)
        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[traj_point])
        
        joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_traj)
        arm_command = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=joint_move_command)
        sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)
        arm_sync_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_arm)
        return RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)
    
    def arm_joint_control(self, joint_positions):
        # Control the arm joints by specifying the desired joint positions
        sh0 = joint_positions[0]
        sh1 = joint_positions[1]
        el0 = joint_positions[2]
        el1 = joint_positions[3]
        wr0 = joint_positions[4]
        wr1 = joint_positions[5]

        traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
            sh0, sh1, el0, el1, wr0, wr1)
        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[traj_point])
        # Make a RobotCommand
        command = self.make_robot_arm_joint_control_command(arm_joint_traj)

        # Send the request
        cmd_id = self.command_client.robot_command(command)
        self.robot.logger.info('Moving arm to grasp position.')
        
    def arm_control_wasd(self):
        # Print helper messages
        print("[wasd]: Radial/Azimuthal control")
        print("[rf]: Up/Down control")
        print("[uo]: X-axis rotation control")
        print("[ik]: Y-axis rotation control")
        print("[jl]: Z-axis rotation control")
        # Use tk to read user input and adjust arm poses
        root = Tk()

        root.bind("<KeyPress>", self.arm_control)
        root.mainloop()
    
    def arm_control(self, event):
        # Control arm by sending commands
        if event.keysym =='w':
            self._move_out()
        elif event.keysym == 's':
            self._move_in()
        elif event.keysym == 'a':
            self._rotate_ccw()
        elif event.keysym == 'd':
            self._rotate_cw()
        elif event.keysym == 'r':
            self._move_up()
        elif event.keysym == 'f':
            self._move_down()
        elif event.keysym == 'i':
            self._rotate_plus_ry()
        elif event.keysym == 'k':
            self._rotate_minus_ry()
        elif event.keysym == 'u':
            self._rotate_plus_rx()
        elif event.keysym == 'o':
            self._rotate_minus_ry()
        elif event.keysym == 'j':
            self._rotate_plus_rz()
        elif event.keysym == 'l':
            self._rotate_minus_rz()
        elif event.keysym == 'g':
            self._arm_stow()
    def _arm_stow(self):
        stow = RobotCommandBuilder.arm_stow_command()

        # Issue the command via the RobotCommandClient
        stow_command_id = self.command_client.robot_command(stow)

        block_until_arm_arrives(self.command_client, stow_command_id, 3.0)
        
    def _move_out(self):
        self._arm_cylindrical_velocity_cmd_helper('move_out', v_r=VELOCITY_HAND_NORMALIZED)

    def _move_in(self):
        self._arm_cylindrical_velocity_cmd_helper('move_in', v_r=-VELOCITY_HAND_NORMALIZED)

    def _rotate_ccw(self):
        self._arm_cylindrical_velocity_cmd_helper('rotate_ccw', v_theta=VELOCITY_HAND_NORMALIZED)

    def _rotate_cw(self):
        self._arm_cylindrical_velocity_cmd_helper('rotate_cw', v_theta=-VELOCITY_HAND_NORMALIZED)

    def _move_up(self):
        self._arm_cylindrical_velocity_cmd_helper('move_up', v_z=VELOCITY_HAND_NORMALIZED)

    def _move_down(self):
        self._arm_cylindrical_velocity_cmd_helper('move_down', v_z=-VELOCITY_HAND_NORMALIZED)

    def _rotate_plus_rx(self):
        self._arm_angular_velocity_cmd_helper('rotate_plus_rx', v_rx=VELOCITY_ANGULAR_HAND)

    def _rotate_minus_rx(self):
        self._arm_angular_velocity_cmd_helper('rotate_minus_rx', v_rx=-VELOCITY_ANGULAR_HAND)

    def _rotate_plus_ry(self):
        self._arm_angular_velocity_cmd_helper('rotate_plus_ry', v_ry=VELOCITY_ANGULAR_HAND)

    def _rotate_minus_ry(self):
        self._arm_angular_velocity_cmd_helper('rotate_minus_ry', v_ry=-VELOCITY_ANGULAR_HAND)

    def _rotate_plus_rz(self):
        self._arm_angular_velocity_cmd_helper('rotate_plus_rz', v_rz=VELOCITY_ANGULAR_HAND)

    def _rotate_minus_rz(self):
        self._arm_angular_velocity_cmd_helper('rotate_minus_rz', v_rz=-VELOCITY_ANGULAR_HAND)

    def _arm_cylindrical_velocity_cmd_helper(self, desc='', v_r=0.0, v_theta=0.0, v_z=0.0):
        """ Helper function to build a arm velocity command from unitless cylindrical coordinates.

        params:
        + desc: string description of the desired command
        + v_r: normalized velocity in R-axis to move hand towards/away from shoulder in range [-1.0,1.0]
        + v_theta: normalized velocity in theta-axis to rotate hand clockwise/counter-clockwise around the shoulder in range [-1.0,1.0]
        + v_z: normalized velocity in Z-axis to raise/lower the hand in range [-1.0,1.0]

        """
        # Build the linear velocity command specified in a cylindrical coordinate system
        cylindrical_velocity = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity()
        cylindrical_velocity.linear_velocity.r = v_r
        cylindrical_velocity.linear_velocity.theta = v_theta
        cylindrical_velocity.linear_velocity.z = v_z

        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cylindrical_velocity=cylindrical_velocity,
            end_time=self.robot.time_sync.robot_timestamp_from_local_secs(time.time() +
                                                                           VELOCITY_CMD_DURATION))

        self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)

    def _arm_angular_velocity_cmd_helper(self, desc='', v_rx=0.0, v_ry=0.0, v_rz=0.0):
        """ Helper function to build a arm velocity command from angular velocities measured with respect
            to the odom frame, expressed in the hand frame.

        params:
        + desc: string description of the desired command
        + v_rx: angular velocity about X-axis in units rad/sec
        + v_ry: angular velocity about Y-axis in units rad/sec
        + v_rz: angular velocity about Z-axis in units rad/sec

        """
        # Specify a zero linear velocity of the hand. This can either be in a cylindrical or Cartesian coordinate system.
        cylindrical_velocity = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity()

        # Build the angular velocity command of the hand
        angular_velocity_of_hand_rt_odom_in_hand = geometry_pb2.Vec3(x=v_rx, y=v_ry, z=v_rz)

        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cylindrical_velocity=cylindrical_velocity,
            angular_velocity_of_hand_rt_odom_in_hand=angular_velocity_of_hand_rt_odom_in_hand,
            end_time=self.robot.time_sync.robot_timestamp_from_local_secs(time.time() +
                                                                           VELOCITY_CMD_DURATION))

        self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)

    def _arm_velocity_cmd_helper(self, arm_velocity_command, desc=''):

        # Build synchronized robot command
        robot_command = robot_command_pb2.RobotCommand()
        robot_command.synchronized_command.arm_command.arm_velocity_command.CopyFrom(
            arm_velocity_command)

        self._start_robot_command(desc, robot_command,
                                  end_time_secs=time.time() + VELOCITY_CMD_DURATION)
    
    """Extra section: AprilTag services"""
    def pregrasp_location_apriltag(self, max_attempts = 10000, option = "image_service"):
        '''Move the robot to the object labeled by apriltag.'''
        attempts = 0
        if option != "image_service" and option != "world_object_service":
            raise ValueError("Invalid option. Must be 'image_service' or 'world_object_service'.")
        
        # Attempt to detect the apriltag & move towards it
        while attempts <= max_attempts:
            detected_fiducial = False
            fiducial_rt_world = None
            if option == "world_object_service":
                # Get the first fiducial object Spot detects with the world object service.
                fiducial = self.get_fiducial_objects()
                if fiducial is not None:
                    vision_tform_fiducial = get_a_tform_b(
                        fiducial.transforms_snapshot, VISION_FRAME_NAME,
                        fiducial.apriltag_properties.frame_name_fiducial).to_proto()
                    if vision_tform_fiducial is not None:
                        detected_fiducial = True
                        fiducial_rt_world = vision_tform_fiducial.position
            else:
                # Detect the april tag in the images from Spot using the apriltag library.
                bboxes, source_name, camera_tform_body, body_tform_world, intrinsics = self.image_to_bounding_box()
                if bboxes:
                    self._previous_source = source_name
                    (tvec, _, source_name) = self.pixel_coords_to_camera_coords(
                        bboxes, intrinsics, source_name)
                    vision_tform_fiducial_position = self.compute_fiducial_in_world_frame(tvec,\
                            camera_tform_body, body_tform_world)
                    fiducial_rt_world = geometry_pb2.Vec3(x=vision_tform_fiducial_position[0],
                                                          y=vision_tform_fiducial_position[1],
                                                          z=vision_tform_fiducial_position[2])
                    detected_fiducial = True

            if detected_fiducial:
                # Go to the tag and stop within a certain distance
                self.go_to_tag(fiducial_rt_world, tag_offset = 1 - 0.3)
                break
            else:
                print('No fiducials found')

            attempts += 1  #increment attempts at finding a fiducial
    ## Find the AprilTag fiducial. 
    # Option 1: Find the AprilTag using world-object service
    def get_fiducial_objects(self):
        """Get all fiducials that Spot detects with its perception system."""
        # Get all fiducial objects (an object of a specific type).
        request_fiducials = [world_object_pb2.WORLD_OBJECT_APRILTAG]
        fiducial_objects = self.world_object_client.list_world_objects(
            object_type=request_fiducials).world_objects
        if len(fiducial_objects) > 0:
            # Return the first detected fiducial.
            return fiducial_objects[0]
        # Return none if no fiducials are found.
        return None
    # Option 2: Use the pre-built service provided by image clients
    def image_to_bounding_box(self):
        """Determine which camera source has a fiducial.
           Return the bounding box of the first detected fiducial."""
        # Iterate through all six camera sources to check for a fiducial
        for source_name in self.image_source_names:
            # Get the image from the source camera.
            img_req = build_image_request(source_name, quality_percent=100,
                                          image_format=image_pb2.Image.FORMAT_RAW)
            image_response = self.image_client.get_image([img_req])
            camera_tform_body = get_a_tform_b(image_response[0].shot.transforms_snapshot,
                                                    image_response[0].shot.frame_name_image_sensor,
                                                    BODY_FRAME_NAME)
            body_tform_world = get_a_tform_b(image_response[0].shot.transforms_snapshot,
                                                   BODY_FRAME_NAME, VISION_FRAME_NAME)

            # Camera intrinsics for the given source camera.
            intrinsics = image_response[0].source.pinhole.intrinsics
            width = image_response[0].shot.image.cols
            height = image_response[0].shot.image.rows

            # detect given fiducial in image and return the bounding box of it
            bboxes = self.detect_fiducial_in_image(image_response[0].shot.image, (width, height),
                                                   source_name)
            if bboxes:
                print(f'Found bounding box for {source_name}')
                return bboxes, source_name, camera_tform_body, body_tform_world, intrinsics
            else:
                self._tag_not_located = True
                print(f'Failed to find bounding box for {source_name}')
        return [], None, None, None, None

    def detect_fiducial_in_image(self, image, dim, source_name, visualization=True):
        """Detect the fiducial within a single image and return its bounding box."""
        image_grey = np.array(
            Image.frombytes('P', (int(dim[0]), int(dim[1])), data=image.data, decoder_name='raw'))

        #Rotate each image such that it is upright
        image_grey = self.rotate_image(image_grey, source_name)

        #Make the image greyscale to use bounding box detections
        options = apriltag.DetectorOptions(families="tag36h11")
        detector = apriltag.Detector(options)
        detections = detector.detect(image_grey)

        bboxes = []
        for i in range(len(detections)):
            # Draw the bounding box detection in the image.
            (ptA, ptB, ptC, ptD) = detections[i].corners
            bbox = np.array([ptA, ptB, ptC, ptD])
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            
            cv2.polylines(image_grey, [np.int32(bbox)], True, (1, 0, 0), 2)
            bboxes.append(bbox)

        if visualization:
            cv2.imshow(f'Fiducial Detection - {source_name}', image_grey)
            # Visualize the bboxes
            cv2.waitKey(0)
        return bboxes

    def bbox_to_image_object_pts(self, bbox):
        """Determine the object points and image points for the bounding box.
           The origin in object coordinates = top left corner of the fiducial.
           Order both points sets following: (TL,TR, BL, BR)"""
        fiducial_height_and_width = 146  #mm
        obj_pts = np.array([[0, 0], [fiducial_height_and_width, 0], [0, fiducial_height_and_width],
                            [fiducial_height_and_width, fiducial_height_and_width]],
                           dtype=np.float32)
        #insert a 0 as the third coordinate (xyz)
        obj_points = np.insert(obj_pts, 2, 0, axis=1)

        #['lb-rb-rt-lt']
        img_pts = np.array([[bbox[3][0], bbox[3][1]], [bbox[2][0], bbox[2][1]],
                            [bbox[0][0], bbox[0][1]], [bbox[1][0], bbox[1][1]]], dtype=np.float32)
        return obj_points, img_pts
    
    def make_camera_matrix(self, ints):
        """Transform the ImageResponse proto intrinsics into a camera matrix."""
        camera_matrix = np.array([[ints.focal_length.x, ints.skew.x, ints.principal_point.x],
                                  [ints.skew.y, ints.focal_length.y, ints.principal_point.y],
                                  [0, 0, 1]])
        return camera_matrix
    
    def pixel_coords_to_camera_coords(self, bbox, intrinsics, source_name):
        """Compute transformation of 2d pixel coordinates to 3d camera coordinates."""
        camera = self.make_camera_matrix(intrinsics)
        # Track a triplet of (translation vector, rotation vector, camera source name)
        best_bbox = (None, None, source_name)
        # The best bounding box is considered the closest to the robot body.
        closest_dist = float('inf')

        for i in range(len(bbox)):
            obj_points, img_points = self.bbox_to_image_object_pts(bbox[i])
            _, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera, np.zeros((5, 1)))

            dist = math.sqrt(float(tvec[0][0])**2 + float(tvec[1][0])**2 +
                             float(tvec[2][0])**2) / 1000.0
            if dist < closest_dist:
                closest_dist = dist
                best_bbox = (tvec, rvec, source_name)

        # Flag indicating if the best april tag been found/located
        self._tag_not_located = best_bbox[0] is None and best_bbox[1] is None
        return best_bbox
    
    def compute_fiducial_in_world_frame(self, tvec, camera_tform_body, body_tform_world):
        """Transform the tag position from camera coordinates to world coordinates."""
        fiducial_rt_camera_frame = np.array(
            [float(tvec[0][0]) / 1000.0,
             float(tvec[1][0]) / 1000.0,
             float(tvec[2][0]) / 1000.0])
        body_tform_fiducial = (camera_tform_body.inverse()).transform_point(
            fiducial_rt_camera_frame[0], fiducial_rt_camera_frame[1], fiducial_rt_camera_frame[2])
        fiducial_rt_world = body_tform_world.inverse().transform_point(
            body_tform_fiducial[0], body_tform_fiducial[1], body_tform_fiducial[2])
        return fiducial_rt_world
    
    ## Go to the tag by an offset
    def go_to_tag(self, fiducial_rt_world, tag_offset = 1.15):
        """Use the position of the april tag in vision world frame and command the robot."""
        # Compute the go-to point (offset by .5m from the fiducial position) and the heading at
        # this point.
        self._current_tag_world_pose, self._tag_angle_desired = self.offset_tag_pose(
            fiducial_rt_world, tag_offset)

        #Command the robot to go to the tag in kinematic odometry frame
        mobility_params = self.get_walking_params(0.5, 1.0)
        tag_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=self._current_tag_world_pose[0], goal_y=self._current_tag_world_pose[1],
            goal_heading=self._tag_angle_desired, frame_name=VISION_FRAME_NAME, params=mobility_params,
            body_height=0.0, locomotion_hint=spot_command_pb2.HINT_AUTO)
        end_time = 5.0
        
        #Issue the command to the robot
        self.command_client.robot_command(lease=None, command=tag_cmd,
                                                     end_time_secs=time.time() + end_time)
        # Feedback to check and wait until the robot is in the desired position or timeout
        start_time = time.time()
        current_time = time.time()
        while (not self.pregrasp_final_state() and current_time - start_time < end_time):
            time.sleep(.25)
            current_time = time.time()
        return
    
    def offset_tag_pose(self, object_rt_world, dist_margin=1.0):
        """Offset the go-to location of the fiducial and compute the desired heading."""
        robot_rt_world = get_vision_tform_body(self.robot_state.kinematic_state.transforms_snapshot)
        robot_to_object_ewrt_world = np.array(
            [object_rt_world.x - robot_rt_world.x, object_rt_world.y - robot_rt_world.y, 0])
        robot_to_object_ewrt_world_norm = robot_to_object_ewrt_world / np.linalg.norm(
            robot_to_object_ewrt_world)
        heading = self.get_desired_angle(robot_to_object_ewrt_world_norm)
        goto_rt_world = np.array([
            object_rt_world.x - robot_to_object_ewrt_world_norm[0] * dist_margin,
            object_rt_world.y - robot_to_object_ewrt_world_norm[1] * dist_margin
        ])
        return goto_rt_world, heading
    
    def pregrasp_final_state(self):
        """Check if the current robot state is within range of the fiducial position."""
        robot_state = get_vision_tform_body(self.robot_state.kinematic_state.transforms_snapshot)
        robot_angle = robot_state.rot.to_yaw()
        if self._current_tag_world_pose.size != 0:
            x_dist = abs(self._current_tag_world_pose[0] - robot_state.x)
            y_dist = abs(self._current_tag_world_pose[1] - robot_state.y)
            angle = abs(self._tag_angle_desired - robot_angle)
            if ((x_dist < 0.05) and (y_dist < 0.05) and (angle < 0.075)):
                return True
        return False
    """Extra section: custom long-horizon task"""
    def move_base_apriltag(self, tag_pose: SE3Pose, distance: SE3Pose):
        '''Move the robot's base to the specified AprilTag pose.'''
        pass

    def get_body_assist_stance_command(self):
        '''A assistive stance is used when manipulating heavy
        objects or interacting with the environment. 
        
        Returns: A body assist stance command'''
        body_control = spot_command_pb2.BodyControlParams(
            body_assist_for_manipulation=spot_command_pb2.BodyControlParams.
            BodyAssistForManipulation(enable_hip_height_assist=True, enable_body_yaw_assist=False))
        
        
        stand_command = CmdBuilder.synchro_stand_command(
            params=spot_command_pb2.MobilityParams(body_control=body_control))
        return stand_command
    
    def drag_arm_impedance(self, gripper_target_pose:bdSE3Pose, spot_body_velocity,
                                reference_frame = BODY_FRAME_NAME, duration_sec=4.0):
        '''
        Part I: Build up the arm impedance control cmd 
        '''
        stand_command = self.get_body_assist_stance_command()
        
        gripper_arm_cmd = robot_command_pb2.RobotCommand()
        gripper_arm_cmd.CopyFrom(stand_command)  # Make sure we keep adjusting the body for the arm
        impedance_cmd = gripper_arm_cmd.synchronized_command.arm_command.arm_impedance_command
        # Set up our root frame; task frame, and tool frame are set by default
        impedance_cmd.root_frame_name = reference_frame

        # Set up stiffness and damping matrices. 
        # Note: these are values obtained from Bosdyn's customer support
        # They claim that these values are used for the arm joint level control
        impedance_cmd.diagonal_stiffness_matrix.CopyFrom(
            geometry_pb2.Vector(values=[500, 500, 500, 60, 60, 60]))
        impedance_cmd.diagonal_damping_matrix.CopyFrom(
            geometry_pb2.Vector(values=[2.0, 2.0, 2.0, 0.55, 0.55, 0.55]))

        # Set up our `desired_tool` trajectory. This is where we want the tool to be with respect
        # to the task frame. The stiffness we set will drag the tool towards `desired_tool`.
        traj = impedance_cmd.task_tform_desired_tool
        pt1 = traj.points.add()
        pt1.time_since_reference.CopyFrom(seconds_to_duration(duration_sec))
        pt1.pose.CopyFrom(gripper_target_pose.to_proto())


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
        drag_trajectory_params = self.get_walking_params(0.3, 0.5)
        vx, vy, vtheta = spot_body_velocity
        # The function to send the se2 synchro velocity command to the robot
        move_cmd = RobotCommandBuilder.synchro_velocity_command(\
            v_x=vx, v_y=vy, v_rot=vtheta,\
                params=drag_trajectory_params, build_on_command=gripper_arm_cmd)
        cmd_id = self.command_client.robot_command(command=move_cmd,\
                            end_time_secs=time.time() + duration_sec)
        # Wait until the robot reports that it is at the goal.
        block_for_trajectory_cmd(self.command_client, cmd_id, timeout_sec=duration_sec + 2.5)
    
    def estimate_obj_distance(self, image_depth_response, bbox):
        ## Estimate the distance to the target object by estimating the depth image
        # Depth is a raw bytestream
        cv_depth = np.frombuffer(image_depth_response.shot.image.data, dtype=np.uint16)
        cv_depth = cv_depth.reshape(image_depth_response.shot.image.rows,
                                    image_depth_response.shot.image.cols)
        
        # Visualize the depth image
        # cv2.applyColorMap() only supports 8-bit; 
        # convert from 16-bit to 8-bit and do scaling
        min_val = np.min(cv_depth)
        max_val = np.max(cv_depth)
        depth_range = max_val - min_val
        depth8 = (255.0 / depth_range * (cv_depth - min_val)).astype('uint8')
        depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
        depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)


        # Save the image locally
        filename = os.path.join("images", "initial_search_depth.png")
        cv2.imwrite(filename, depth_color)
        # Convert the value into real-world distance & Find the average value within bbox
        dist_avg = 0
        dist_count = 0
        dist_scale = image_depth_response.source.depth_scale

        for i in range(int(bbox[1]), int(bbox[3])):
            for j in range((int(bbox[0])), int(bbox[2])):
                if cv_depth[i, j] != 0:
                    dist_avg += cv_depth[i, j] / dist_scale
                    dist_count += 1
        
        distance = dist_avg / dist_count
        return distance

    def estimate_obj_pose_hand(self, bbox, image_response, distance):
        ## Estimate the target object pose (indicated by the bounding box) in hand frame
        bbox_center = [int((bbox[1] + bbox[3])/2), int((bbox[0] + bbox[2])/2)]
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
        result = SE3Pose(x=root[2], y=-root[0], z=-root[1], rot=Quat(w=1, x=0, y=0, z=0))
        return result
    def correct_body(self, obj_pose_hand):
        # Find the object pose in body frame
        robot_state = self.state_client.get_robot_state()
        body_T_hand = frame_helpers.get_a_tform_b(\
                    robot_state.kinematic_state.transforms_snapshot,
                    frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME, \
                    frame_helpers.HAND_FRAME_NAME)
        body_T_obj = body_T_hand * obj_pose_hand

        # Rotate the body 
        body_T_obj_se2 = body_T_obj.get_closest_se2_transform()

        # Command the robot to rotate its body
        move_command = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(\
            0, 0, body_T_obj_se2.angle, \
                robot_state.kinematic_state.transforms_snapshot, \
                params=self.get_walking_params(0.6, 1))
        id = self.command_client.robot_command(command=move_command, \
                                               end_time_secs=time.time() + 10)
        block_for_trajectory_cmd(self.command_client,
                                    cmd_id=id, 
                                    feedback_interval_secs=5, 
                                    timeout_sec=10,
                                    logger=None)



    def move_base_arc(self, obj_pose_hand:SE3Pose, angle):
        # Find the object pose in body frame
        robot_state = self.state_client.get_robot_state()
        body_T_hand = frame_helpers.get_a_tform_b(\
                    robot_state.kinematic_state.transforms_snapshot,
                    frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME, \
                    frame_helpers.HAND_FRAME_NAME)
        body_T_obj = body_T_hand * obj_pose_hand

        # Rotate the robot base w.r.t the object pose
        body_T_obj_mat = body_T_obj.to_matrix()
        rot_mat = R.from_rotvec([0, 0, angle]).as_matrix()
        rot = np.eye(4)
        rot[0:3, 0:3] = rot_mat
        body_T_target = body_T_obj_mat @ rot @ np.linalg.inv(body_T_obj_mat)
        body_T_target = SE3Pose.from_matrix(body_T_target)
        odom_T_body = frame_helpers.get_a_tform_b(\
                    robot_state.kinematic_state.transforms_snapshot,
                    frame_helpers.ODOM_FRAME_NAME, \
                    frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME)
        odom_T_target = odom_T_body * body_T_target
        # Send the command to move the robot base
        odom_T_target_se2 = odom_T_target.get_closest_se2_transform()
        # Command the robot to open its gripper
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1)
        move_command = RobotCommandBuilder.synchro_se2_trajectory_point_command(\
            odom_T_target_se2.x, odom_T_target_se2.y, odom_T_target_se2.angle, \
                frame_name=ODOM_FRAME_NAME, \
                params=self.get_walking_params(0.6, 1),\
                build_on_command=gripper_command)
        id = self.command_client.robot_command(command=move_command, \
                                               end_time_secs=time.time() + 10)
        block_for_trajectory_cmd(self.command_client,
                                    cmd_id=id, 
                                    feedback_interval_secs=5, 
                                    timeout_sec=10,
                                    logger=None)
        print("Moving done")

    """Extra session: RL"""
    def get_RL_obs(self):
        pass
    

## Environment to deploy pretrained policy on SPOT

# TODO: might need to adjust the action & observation space
class SpotRLEnvPLAY():
    
    # Initialize the necessary attributes
    def __init__(self, robot):
        self.robot = robot

    
    def reset(self):
        """TODO: figure out how to write reset on policy deployment..."""
        pass
    
    def step(self, action):
        """TODO: Temporarily only support the movement of SE(2) velocity"""
        joint_action = action[:, 3:10]
        velocity = action[:, 0:3]
        velocity = velocity.squeeze()
        arm_joint_movement = self.robot.make_robot_arm_joint_command_displacement(joint_action)
        
        arm_joint_movement = self.robot.make_arm_joint_freeze_command()
        self.robot.send_velocity_command_se2(\
            vx = velocity[0], vy = velocity[1], vtheta = velocity[2],\
            build_on_command = arm_joint_movement, exec_time=0.5)
        
    def update_robot(self, robot):
        # Update the robot to use
        self.robot = robot
        
    def close(self):
        # Close the robot connection
        self.robot.power_off()
