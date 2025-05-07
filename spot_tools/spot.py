import json
import threading
import time
import uuid
from pathlib import Path
from typing import TypeVar, Type

import numpy as np
from bosdyn.client import create_standard_sdk, BaseClient, frame_helpers, math_helpers
from bosdyn.client.math_helpers import SE3Pose
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, block_until_arm_arrives
from bosdyn.client.robot_state import RobotStateClient

T = TypeVar('T', bound=BaseClient)


class _SpotClients[T]:
    def __init__(self, spot: 'Spot'):
        self._robot = spot.robot
        self._clients: dict[str, T] = {}
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
                f'Can\'t find or create type {type(client_type)}, as it doesn\'t have a default service name'
            )
        return key

    def _add_client(self, client_type: Type[T]) -> T:
        print(f'New robot client added of type {client_type}')
        key = self._get_default_service_name(client_type)
        client = self._clients.setdefault(key, self._robot.ensure_client(key))
        if self.always_print_clients:
            self.print_clients()
        return client

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

    def is_holding_object(self, minimum_grip: float) -> bool:
        # grab necessary clients
        robot_state_client = self._clients[RobotStateClient]

        # request robot state and look in the manipulator_state section
        robot_state = robot_state_client.get_robot_state()
        grip_amount = robot_state.manipulator_state.gripper_open_percentage / 100
        print(f'Gripper at {grip_amount:.2f}, {minimum_grip=}')

        # True if the gripper is more open than the input minimum
        return grip_amount >= minimum_grip


class _SpotArm:
    def __init__(self, clients: _SpotClients):
        self._clients = clients

    def stow(self, blocking: bool = True) -> None:
        # grab necessary clients
        command_client = self._clients[RobotCommandClient]
        robot_state_client = self._clients[RobotStateClient]

        if blocking:
            print('Blocking stow requested...', end='', flush=True)
        else:
            print('Non-blocking stow requested...', end='', flush=True)

        # if robot is holding something, it must drop it before it can stow its arm
        robot_state = robot_state_client.get_robot_state()
        if robot_state.manipulator_state.is_gripper_holding_item:
            print('Releasing object...', end='', flush=True)
            open_gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
            command_client.robot_command(open_gripper_command)
            time.sleep(0.8)

        # create actual command
        stow_cmd = RobotCommandBuilder.arm_stow_command()
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(0.0)
        synchro_command = RobotCommandBuilder.build_synchro_command(gripper_command, stow_cmd)

        # execute command, blocking if requested
        cmd_id = command_client.robot_command(synchro_command)
        if blocking:
            block_until_arm_arrives(command_client, cmd_id)
        print('Finished')


class _SpotBody:
    def __init__(self, clients: _SpotClients):
        self._clients = clients

    def compute_stand_location_and_yaw(self, vision_tform_target: SE3Pose, distance_margin: float) -> tuple[tuple[float, float, float], float]:
        # grab necessary clients
        robot_state_client = self._clients[RobotStateClient]

        # Compute drop-off location:
        #   Draw a line from Spot to the person
        #   Back up 2.0 meters on that line
        vision_tform_robot = frame_helpers.get_a_tform_b(
            robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
            frame_helpers.VISION_FRAME_NAME,
            frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME
        )

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
            # noinspection PyTypeChecker
            robot_rt_person_ewrt_vision_hat = robot_rt_person_ewrt_vision / np.linalg.norm(robot_rt_person_ewrt_vision)

        # Starting at the person, back up meters along the unit vector.
        drop_position_rt_vision = (
            vision_tform_target.x + robot_rt_person_ewrt_vision_hat[0] * distance_margin,
            vision_tform_target.y + robot_rt_person_ewrt_vision_hat[1] * distance_margin,
            vision_tform_target.z + robot_rt_person_ewrt_vision_hat[2] * distance_margin
        )

        # We also want to compute a rotation (yaw) so that we will face the person when dropping.
        # We'll do this by computing a rotation matrix with X along
        #   -robot_rt_person_ewrt_vision_hat (pointing from the robot to the person) and Z straight up:
        xhat = -robot_rt_person_ewrt_vision_hat
        zhat = [0.0, 0.0, 1.0]
        yhat = np.cross(zhat, xhat)
        mat = np.matrix([xhat, yhat, zhat]).transpose()
        heading_rt_vision = math_helpers.Quat.from_matrix(mat).to_yaw()

        return drop_position_rt_vision, heading_rt_vision


class Spot:
    def __init__(self, authentication_file: str | Path, ip: str = '192.168.80.3', client_name: str | None = None):
        self.sdk = create_standard_sdk(client_name or str(uuid.uuid4()))
        self.robot = self.sdk.create_robot(ip)

        authentication = json.loads(Path(authentication_file).read_bytes())
        self.robot.authenticate(authentication['username'], authentication['password'])

        self.robot.time_sync.wait_for_sync()

        self.clients = _SpotClients(self)
        self.gripper = _SpotGripper(self.clients)
        self.arm = _SpotArm(self.clients)
        self.body = _SpotBody(self.clients)
