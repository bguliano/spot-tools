import json
import uuid
from pathlib import Path
from typing import Optional

from bosdyn.client import create_standard_sdk

from robot_clients import RobotClients


def setup_spot(authentication_file: str, client_name: Optional[str] = None, robot_ip: str = '192.168.80.3') -> RobotClients:
    sdk = create_standard_sdk(client_name or str(uuid.uuid4()))
    robot = sdk.create_robot(robot_ip)

    authentication = json.loads(Path(authentication_file).read_bytes())
    robot.authenticate(authentication['username'], authentication['password'])

    robot.time_sync.wait_for_sync()
    return RobotClients(robot)
