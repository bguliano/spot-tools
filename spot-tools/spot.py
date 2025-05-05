import json
import threading
import uuid
from pathlib import Path
from typing import TypeVar, Type

from bosdyn.client import create_standard_sdk, BaseClient

T = TypeVar('T', bound=BaseClient)


class SpotClients[T]:
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


class Spot:
    def __init__(self, authentication_file: str | Path, ip: str = '192.168.80.3', client_name: str | None = None):
        self.sdk = create_standard_sdk(client_name or str(uuid.uuid4()))
        self.robot = self.sdk.create_robot(ip)

        authentication = json.loads(Path(authentication_file).read_bytes())
        self.robot.authenticate(authentication['username'], authentication['password'])

        self.robot.time_sync.wait_for_sync()

        self.clients = SpotClients(self)
