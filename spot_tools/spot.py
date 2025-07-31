import atexit
import json
import threading
import time
import uuid
from pathlib import Path
from typing import TypeVar, Type, Generic, Dict, List, Tuple, Optional, Union

import cv2
import numpy as np
from bosdyn.api.arm_command_pb2 import ArmCommand, ArmCartesianCommand
from bosdyn.api.geometry_pb2 import FrameTreeSnapshot
from bosdyn.api.image_pb2 import Image, ImageRequest
from bosdyn.api.robot_command_pb2 import RobotCommand
from bosdyn.api.synchronized_command_pb2 import SynchronizedCommand
from bosdyn.api.trajectory_pb2 import SE3TrajectoryPoint, SE3Trajectory
from bosdyn.client import create_standard_sdk, BaseClient, frame_helpers, math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.math_helpers import SE3Pose, Quat
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, block_until_arm_arrives
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.util import seconds_to_duration
from google.protobuf.wrappers_pb2 import BoolValue, DoubleValue

from spot_tools.common import SpotImageSource, process_image_response

T = TypeVar('T', bound=BaseClient)


class _SpotClients(Generic[T]):
    def __init__(self, spot: 'Spot'):
        self._robot = spot.robot
        self._clients: Dict[str, T] = {}
        self._lock = threading.Lock()  # make sure this object is thread-safe
        self.always_print_clients = True

    def __getitem__(self, client_type: Type[T]) -> T:
        # type checking on parameter input
        if not issubclass(client_type, BaseClient):
            raise TypeError(f'item parameter must be a BaseClient type, instead got {type(client_type)}')

        # get the default service name of the client
        key = self._get_default_service_name(client_type)

        # either create or return the correct client
        with self._lock:
            try:
                return self._clients[key]
            except KeyError:
                return self._add_client(client_type)

    @staticmethod
    def _get_default_service_name(client_type: Type[T]) -> str:
        key = getattr(client_type, 'default_service_name', None)
        if key is None:
            raise ValueError(
                f"Can't find or create type {client_type}, as it doesn't have a default service name"
            )
        return key

    def _add_client(self, client_type: Type[T]) -> T:
        print(f'New robot client added of type {client_type}')
        key = self._get_default_service_name(client_type)
        client = self._clients.setdefault(key, self._robot.ensure_client(key))  # type: ignore
        if self.always_print_clients:
            self.print_clients()
        return client  # type: ignore

    def print_clients(self) -> None:
        print('Current clients:')
        for client_name, client in self._clients.items():
            print(f'\tName: "{client_name}" Type: {type(client)}')
        print()


class _SpotGripper:
    def __init__(self, clients: _SpotClients):
        self._clients = clients

    def get_hand_pose(self) -> SE3Pose:
        robot_state = self._clients[RobotStateClient].get_robot_state()
        hand_pos = frame_helpers.get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot,
            frame_helpers.VISION_FRAME_NAME,
            frame_helpers.HAND_FRAME_NAME
        )
        return hand_pos

    def is_object_in_gripper(self, minimum_grip: float) -> bool:
        robot_state = self._clients[RobotStateClient].get_robot_state()
        grip_amount = robot_state.manipulator_state.gripper_open_percentage / 100.0
        print(f'Gripper at {grip_amount:.2f}, minimum_grip={minimum_grip}')
        return grip_amount >= minimum_grip

    def open_percentage(self, percentage: float, blocking: bool = True) -> None:
        command_client = self._clients[RobotCommandClient]
        open_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(percentage)
        cmd_id = command_client.robot_command(open_cmd)
        if blocking:
            block_until_arm_arrives(command_client, cmd_id)

    def open(self, blocking: bool = True) -> None:
        self.open_percentage(1.0, blocking)

    def close(self, blocking: bool = True) -> None:
        self.open_percentage(0.0, blocking)


class _SpotArm:
    def __init__(self, clients: _SpotClients):
        self._clients = clients

    def stow(self, blocking: bool = True) -> None:
        command_client = self._clients[RobotCommandClient]
        robot_state_client = self._clients[RobotStateClient]

        print(('Blocking' if blocking else 'Non-blocking') + ' stow requested...', end='', flush=True)

        robot_state = robot_state_client.get_robot_state()
        if robot_state.manipulator_state.is_gripper_holding_item:
            print(' Releasing object...', end='', flush=True)
            cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
            command_client.robot_command(cmd)
            time.sleep(0.8)

        stow_cmd = RobotCommandBuilder.arm_stow_command()
        grip_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(0.0)
        sync_cmd = RobotCommandBuilder.build_synchro_command(grip_cmd, stow_cmd)

        cmd_id = command_client.robot_command(sync_cmd)
        if blocking:
            block_until_arm_arrives(command_client, cmd_id)

    def ready(self, blocking: bool = True) -> None:
        command_client = self._clients[RobotCommandClient]
        cmd_id = command_client.robot_command(RobotCommandBuilder.arm_ready_command())
        if blocking:
            block_until_arm_arrives(command_client, cmd_id)

    def pose(
        self, pos_x: float, pos_y: float, pos_z: float,
        rot_w: float, rot_x: float, rot_y: float, rot_z: float,
        gripper_open: Optional[float] = None,
        seconds: int = 1, disable_velocity_limiting: bool = False,
        blocking: bool = True
    ) -> None:
        command_client = self._clients[RobotCommandClient]
        hand_pose = SE3Pose(pos_x, pos_y, pos_z, rot=Quat(rot_w, rot_x, rot_y, rot_z)).to_proto()
        duration = seconds_to_duration(seconds)
        traj_point = SE3TrajectoryPoint(pose=hand_pose, time_since_reference=duration)
        hand_traj = SE3Trajectory(points=[traj_point])

        disable_cmds = {
            'disable_velocity_limiting': BoolValue(value=True),
            'maximum_acceleration': DoubleValue(value=20.0),
            'max_linear_velocity': DoubleValue(value=20.0),
            'max_angular_velocity': DoubleValue(value=20.0)
        } if disable_velocity_limiting else {}

        arm_req = ArmCartesianCommand.Request(
            root_frame_name=frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME,
            pose_trajectory_in_task=hand_traj,
            **disable_cmds
        )
        sync = SynchronizedCommand.Request(arm_command=ArmCommand.Request(arm_cartesian_command=arm_req))
        robot_cmd = RobotCommand(synchronized_command=sync)

        if gripper_open is not None:
            robot_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(
                gripper_open, build_on_command=robot_cmd)

        cmd_id = command_client.robot_command(robot_cmd)
        if blocking:
            block_until_arm_arrives(command_client, cmd_id)

    POSE_TUPLE = Tuple[float, float, float, float, float, float, float]

    def pose_many(
        self, positions: List[POSE_TUPLE],
        seconds_between: int = 1, disable_velocity_limiting: bool = False,
        blocking: bool = True
    ) -> None:
        command_client = self._clients[RobotCommandClient]
        traj_points: List[SE3TrajectoryPoint] = []

        for i, pose in enumerate(positions):
            hand_pose = SE3Pose(*pose[:3], rot=Quat(*pose[3:])).to_proto()
            tsr = seconds_to_duration(i * seconds_between)
            traj_points.append(SE3TrajectoryPoint(pose=hand_pose, time_since_reference=tsr))

        hand_traj = SE3Trajectory(points=traj_points)
        disable_cmds = {
            'disable_velocity_limiting': BoolValue(value=True),
            'maximum_acceleration': DoubleValue(value=20.0),
            'max_linear_velocity': DoubleValue(value=20.0),
            'max_angular_velocity': DoubleValue(value=20.0)
        } if disable_velocity_limiting else {}

        arm_req = ArmCartesianCommand.Request(
            root_frame_name=frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME,
            pose_trajectory_in_task=hand_traj,
            **disable_cmds
        )
        sync = SynchronizedCommand.Request(arm_command=ArmCommand.Request(arm_cartesian_command=arm_req))
        robot_cmd = RobotCommand(synchronized_command=sync)

        cmd_id = command_client.robot_command(robot_cmd)
        if blocking:
            block_until_arm_arrives(command_client, cmd_id)


class _SpotBody:
    def __init__(self, clients: _SpotClients):
        self._clients = clients

    def compute_stand_location_and_yaw(
        self, vision_tform_target: SE3Pose, distance_margin: float
    ) -> Tuple[Tuple[float, float, float], float]:
        robot_state = self._clients[RobotStateClient].get_robot_state()
        vision_to_body = frame_helpers.get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot,
            frame_helpers.VISION_FRAME_NAME,
            frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME
        )

        vec = np.array([
            vision_to_body.x - vision_tform_target.x,
            vision_to_body.y - vision_tform_target.y,
            vision_to_body.z - vision_tform_target.z
        ])
        norm = np.linalg.norm(vec)
        if norm < 0.01:
            vec_hat = vision_to_body.transform_point(1, 0, 0)
        else:
            vec_hat = vec / norm

        drop = (
            vision_tform_target.x + vec_hat[0] * distance_margin,
            vision_tform_target.y + vec_hat[1] * distance_margin,
            vision_tform_target.z + vec_hat[2] * distance_margin
        )

        xhat = -vec_hat
        zhat = np.array([0.0, 0.0, 1.0])
        yhat = np.cross(zhat, xhat)
        mat = np.vstack([xhat, yhat, zhat]).T
        yaw = math_helpers.Quat.from_matrix(mat).to_yaw()
        return drop, yaw


class _SpotImage:
    def __init__(self, clients: _SpotClients):
        self._clients = clients

    def get_image_from_source(
        self, image_source: SpotImageSource,
        color_image: bool = True, image_quality: int = 100
    ) -> np.ndarray:
        client = self._clients[ImageClient]
        pf = Image.PixelFormat.PIXEL_FORMAT_RGB_U8 if color_image else Image.PixelFormat.PIXEL_FORMAT_GREYSCALE_U8
        req = ImageRequest(
            image_source_name=image_source,
            image_format=Image.Format.FORMAT_JPEG,
            pixel_format=pf,
            quality_percent=image_quality
        )
        resp = client.get_image([req])[0]
        return process_image_response(resp)


class Spot:
    def __init__(
        self,
        authentication_file: Union[str, Path],
        ip: str = '192.168.80.3',
        client_name: Optional[str] = None,
        should_acquire_lease: bool = False
    ):
        self.sdk = create_standard_sdk(client_name or str(uuid.uuid4()))
        self.robot = self.sdk.create_robot(ip)

        auth = json.loads(Path(authentication_file).read_bytes())
        self.robot.authenticate(auth['username'], auth['password'])
        self.robot.time_sync.wait_for_sync()

        self.clients = _SpotClients(self)
        self._lease_keep_alive: Optional[LeaseKeepAlive] = None
        if should_acquire_lease:
            lease_client = self.clients[LeaseClient]
            lease_client.take()
            self._lease_keep_alive = LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)
            atexit.register(self._lease_keep_alive.shutdown)

        if self.robot.has_arm():
            self.gripper = _SpotGripper(self.clients)
            self.arm = _SpotArm(self.clients)
        self.body = _SpotBody(self.clients)
        self.image = _SpotImage(self.clients)

    def get_transforms_snapshot(self) -> FrameTreeSnapshot:
        return self.robot.get_frame_tree_snapshot()

    def get_parent_tform_frame(self, frame_name: str) -> Optional[SE3Pose]:
        snapshot = self.get_transforms_snapshot()
        edge = snapshot.child_to_parent_edge_map.get(frame_name)
        if edge is None:
            return None
        return SE3Pose.from_proto(edge.parent_tform_child)


def main() -> None:
    spot = Spot(authentication_file='../authentication.json')
    img = spot.image.get_image_from_source(SpotImageSource.LEFT, color_image=False)
    cv2.imshow('Spot', img)
    cv2.waitKey(1)


if __name__ == '__main__':
    main()
