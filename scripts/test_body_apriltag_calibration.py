"""Real SDK messages/transforms, mocked services; never connect to a robot."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from bosdyn.api import world_object_pb2, robot_command_pb2
from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME
from bosdyn.client.math_helpers import SE3Pose, Quat
from bosdyn.client.robot_command import RobotCommandBuilder


def load_methods(clock, wait):
    path = Path(__file__).with_name("robot.py")
    tree = ast.parse(path.read_text())
    spot = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SPOT")
    spot.body = [n for n in spot.body if isinstance(n, ast.FunctionDef) and n.name in {
        "get_apriltag_pose_hand", "gripper_camera_look_down_on_back", "get_body_T_AprilTag",
    }]
    ns = dict(np=np, math=math, time=clock, world_object_pb2=world_object_pb2,
              robot_command_pb2=robot_command_pb2, get_a_tform_b=get_a_tform_b,
              BODY_FRAME_NAME=BODY_FRAME_NAME, RobotCommandBuilder=RobotCommandBuilder,
              block_until_arm_arrives=wait)
    exec(compile(ast.Module(body=[spot], type_ignores=[]), str(path), "exec"), ns)
    return ns["SPOT"]()


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.elapsed = 0.0
        def sleep(seconds):
            self.elapsed += seconds
        clock = SimpleNamespace(time=lambda: 100.0 + self.elapsed,
                                monotonic=lambda: self.elapsed, sleep=sleep)
        self.wait = Mock(return_value=True)
        self.spot = load_methods(clock, self.wait)
        self.spot.robot = Mock()
        self.spot.robot.is_powered_on.return_value = True
        self.spot.robot.is_estopped.return_value = False
        converter = self.spot.robot.time_sync.get_robot_time_converter.return_value
        converter.local_seconds_from_robot_timestamp.side_effect = lambda t: t.seconds + t.nanos / 1e9
        self.spot.activate_thread = None
        self.spot.joint_api_interface = SimpleNamespace(started_streaming=False)
        self.spot.command_client = Mock()
        self.spot.command_client.robot_command.return_value = 42
        self.spot.state_client = Mock()
        self.spot.image_client = Mock()
        self.spot.image_client.get_image_from_sources.return_value = [
            SimpleNamespace(shot=SimpleNamespace(frame_name_image_sensor="hand_camera"))]
        self.camera_T_tag = SE3Pose(0.1, -0.05, 0.4, Quat.from_roll(0.2))
        self.body_T_camera = SE3Pose(-0.1, 0.2, 0.6, Quat.from_pitch(1.2))
        self.obj = world_object_pb2.WorldObject()
        self.obj.acquisition_time.seconds = 100
        props = self.obj.apriltag_properties
        props.tag_id = 7
        props.frame_name_camera = "hand_camera"
        props.frame_name_fiducial = "fiducial_7"
        props.fiducial_pose_status = world_object_pb2.AprilTagProperties.STATUS_OK
        props.dimensions.x = props.dimensions.y = 0.2
        for child, parent, pose in (
            ("vision", "", SE3Pose(0, 0, 0, Quat())),
            ("body", "vision", SE3Pose(5, 2, 0.5, Quat.from_yaw(0.7))),
            ("hand_camera", "body", self.body_T_camera),
            ("fiducial_7", "hand_camera", self.camera_T_tag),
        ):
            edge = self.obj.transforms_snapshot.child_to_parent_edge_map[child]
            edge.parent_frame_name = parent
            edge.parent_tform_child.CopyFrom(pose.to_proto())
        self.spot.world_object_client = Mock()
        self.spot.world_object_client.list_world_objects.return_value = SimpleNamespace(world_objects=[self.obj])

    def test_camera_pose_and_same_snapshot_body_calibration(self):
        pose = self.spot.get_apriltag_pose_hand(7, 0.2)
        np.testing.assert_allclose(pose.to_matrix(), self.camera_T_tag.to_matrix(), atol=1e-12)
        body_T_tag = self.spot.get_body_T_AprilTag(7, 0.2)
        expected = self.body_T_camera * self.camera_T_tag
        np.testing.assert_allclose(body_T_tag.to_matrix(), expected.to_matrix(), atol=1e-12)
        self.assertIs(self.spot.body_T_AprilTag, body_T_tag)
        self.assertEqual(self.spot.body_apriltag_id, 7)
        self.spot.state_client.get_robot_state.assert_not_called()
        self.spot.command_client.robot_command.assert_not_called()

    def test_wrong_tag_source_quality_and_stale_data_are_rejected(self):
        for case in ("id", "source", "ambiguous", "unknown", "high_error", "stale", "future", "missing_time"):
            with self.subTest(case=case):
                self.setUp()
                props = self.obj.apriltag_properties
                if case == "id": props.tag_id = 8
                if case == "source": props.frame_name_camera = "frontleft_fisheye"
                if case == "ambiguous": props.fiducial_pose_status = 2
                if case == "unknown": props.fiducial_pose_status = 0
                if case == "high_error": props.fiducial_pose_status = 3
                if case == "stale": self.obj.acquisition_time.seconds = 99
                if case == "future": self.obj.acquisition_time.seconds = 110
                if case == "missing_time": self.obj.ClearField("acquisition_time")
                with self.assertRaises(TimeoutError):
                    self.spot.get_apriltag_pose_hand(7, 0.2, timeout_sec=0.2)

    def test_missing_service_detection_times_out_without_motion(self):
        self.spot.world_object_client.list_world_objects.return_value.world_objects = []
        with self.assertRaisesRegex(TimeoutError, "hand_color_image"):
            self.spot.get_apriltag_pose_hand(7, 0.2, timeout_sec=0.2)
        self.spot.command_client.robot_command.assert_not_called()

    def test_size_mismatch_and_failed_recalibration_clear_cache(self):
        self.spot.body_T_AprilTag = self.body_T_camera
        self.obj.apriltag_properties.dimensions.x = 0.146
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.spot.get_body_T_AprilTag(7, 0.2)
        self.assertIsNone(self.spot.body_T_AprilTag)
        self.assertIsNone(self.spot.body_apriltag_id)

    def test_missing_body_frame_rejects_calibration(self):
        snapshot = self.obj.transforms_snapshot
        snapshot.child_to_parent_edge_map["hand_camera"].parent_frame_name = "vision"
        del snapshot.child_to_parent_edge_map["body"]
        with self.assertRaisesRegex(RuntimeError, "body-to-camera"):
            self.spot.get_body_T_AprilTag(7, 0.2)
        self.assertIsNone(self.spot.body_T_AprilTag)

    def test_invalid_detection_arguments_never_request_images(self):
        for args in ((-1, 0.2), (True, 0.2), (7, 0), (7, float("nan"))):
            with self.assertRaises(ValueError):
                self.spot.get_apriltag_pose_hand(*args)
        self.spot.image_client.get_image_from_sources.assert_not_called()

    def test_move_waits_and_preserves_gripper(self):
        joints = [0, -1, 1.5, 0, 0.5, 0]  # Synthetic test input, NOT a recommended viewing pose.
        self.assertEqual(self.spot.gripper_camera_look_down_on_back(joints), 42)
        cmd = self.spot.command_client.robot_command.call_args.args[0]
        sync = cmd.synchronized_command
        self.assertTrue(sync.mobility_command.HasField("stand_request"))
        self.assertFalse(sync.HasField("gripper_command"))
        move = sync.arm_command.arm_joint_move_command
        self.assertAlmostEqual(move.trajectory.maximum_velocity.value, 0.4)
        self.wait.assert_called_once_with(self.spot.command_client, 42, 10.0)

    def test_invalid_arm_pose_never_commands_robot(self):
        for joints in ([0] * 7, [float("nan")] * 6, [10] * 6):
            with self.assertRaises(ValueError):
                self.spot.gripper_camera_look_down_on_back(joints)
        self.spot.command_client.robot_command.assert_not_called()

    def test_power_estop_and_relic_guards(self):
        for case in ("off", "estop", "relic", "streaming"):
            with self.subTest(case=case):
                self.setUp()
                if case == "off": self.spot.robot.is_powered_on.return_value = False
                if case == "estop": self.spot.robot.is_estopped.return_value = True
                if case == "relic": self.spot.activate_thread = Mock()
                if case == "streaming": self.spot.joint_api_interface.started_streaming = True
                with self.assertRaises(RuntimeError):
                    self.spot.gripper_camera_look_down_on_back([0, -1, 1.5, 0, 0.5, 0])
                self.spot.command_client.robot_command.assert_not_called()

    def test_arm_failure_and_interrupt_request_arm_stop(self):
        for interrupt in (False, True):
            with self.subTest(interrupt=interrupt):
                self.setUp()
                if interrupt:
                    self.wait.side_effect = KeyboardInterrupt
                else:
                    self.wait.return_value = False
                with self.assertRaises(KeyboardInterrupt if interrupt else RuntimeError):
                    self.spot.gripper_camera_look_down_on_back([0, -1, 1.5, 0, 0.5, 0])
                stop = self.spot.command_client.robot_command.call_args.args[0]
                self.assertTrue(stop.synchronized_command.arm_command.HasField("arm_stop_command"))


if __name__ == "__main__":
    unittest.main()
