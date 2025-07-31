import logging
import random
import string
import threading
from typing import Optional, Dict, List

import bosdyn.client
from bosdyn.api.mission import remote_service_pb2_grpc, remote_pb2
from bosdyn.client.server_util import ResponseContext

from spot_tools.param_manager import ParamManager


class RMSSkeleton(remote_service_pb2_grpc.RemoteMissionServiceServicer):
    """A skeleton class for subclassing a RemoteMissionService that handles sessions, params, callbacks
    automatically
    """

    def __init__(self, bosdyn_sdk_robot: bosdyn.client.robot.Robot, logger: Optional[logging.Logger]):
        self.bosdyn_sdk_robot = bosdyn_sdk_robot
        self.logger = logger

        # used for threading, each session is assigned its own thread
        # threads are used so that responses can be sent as soon as possible (as per BD)
        self._threads_by_session_id: Dict[str, Optional[threading.Thread]] = {}
        self._used_session_ids: List[str] = []
        self._lock = threading.Lock()
        self._ticks_running: List[str] = []

        # init custom params
        self.param_manager = ParamManager(logger=logger)

    def _get_unique_random_session_id(self) -> str:
        """Create a random 16-character session ID that hasn't been used."""

        while True:
            session_id = ''.join([random.choice(string.ascii_letters) for _ in range(16)])
            if session_id not in self._used_session_ids:
                return session_id

    # --- CLASS STRUCTURE ------------------------------------------------------------------------------------
    # | * the camel-case methods are override methods from the RemoteMissionServiceServicer parent class     |
    # | * the _implementation methods are in charge of managing the servicer, as well as providing responses |
    # | * the on_ methods are meant to be overridden by child classes, and perform the main logic            |
    # --------------------------------------------------------------------------------------------------------

    def EstablishSession(self, request, context) -> remote_pb2.EstablishSessionResponse:
        response = remote_pb2.EstablishSessionResponse()
        with ResponseContext(response, request):
            self.logger.info(f'Establishing session with {len(request.leases)} leases and '
                             f'{len(request.inputs)} inputs')
            with self._lock:
                self._establish_session_implementation(request, response)
            self.logger.info(f'Session id is {response.session_id}')
        return response

    def _establish_session_implementation(self, request, response):
        # generate a new session id and corresponding Session object
        session_id = self._get_unique_random_session_id()
        self._threads_by_session_id[session_id] = None
        self._used_session_ids.append(session_id)
        response.session_id = session_id

        # run subclass code before responding with OK
        self.on_init(request)

        response.status = remote_pb2.EstablishSessionResponse.STATUS_OK

    def Tick(self, request, context) -> remote_pb2.TickResponse:
        response = remote_pb2.TickResponse()
        with ResponseContext(response, request):
            self.logger.info(f'Ticked with session ID "{request.session_id}" with {len(request.leases)} leases and '
                             f'{len(request.inputs)} inputs')
            with self._lock:
                self._tick_implementation(request, response)
        return response

    def _tick_implementation(self, request, response):
        """Set up a proper response based on the status of this session's thread."""

        # the session id is not valid, something is wrong
        if request.session_id not in self._threads_by_session_id:
            self.logger.error(f'Cannot find session id "{request.session_id}"')
            response.status = remote_pb2.TickResponse.STATUS_INVALID_SESSION_ID
            return

        # stop Spot from creating a new thread each time this method is called
        if request.session_id not in self._ticks_running:
            # params = self._get_params(request)
            new_thread = threading.Thread(target=self.on_start, args=(request,))
            new_thread.start()
            self._threads_by_session_id[request.session_id] = new_thread
            self._ticks_running.append(request.session_id)
            response.status = remote_pb2.TickResponse.STATUS_RUNNING
        else:
            # find the thread associated with this session
            session_thread = self._threads_by_session_id[request.session_id]

            # if the thread is already running, respond with STATUS_RUNNING
            if session_thread.is_alive():
                response.status = remote_pb2.TickResponse.STATUS_RUNNING

            # if the thread is no longer running, respond with STATUS_SUCCESS
            else:
                response.status = remote_pb2.TickResponse.STATUS_SUCCESS

    def Stop(self, request, context) -> remote_pb2.StopResponse:
        response = remote_pb2.StopResponse()
        with ResponseContext(response, request):
            self.logger.info(f'Stopping session {request.session_id}')
            with self._lock:
                self._stop_implementation(request, response)
        return response

    def _stop_implementation(self, request, response):
        # loses its reference to its session thread
        self._threads_by_session_id[request.session_id] = None

        # run subclass code before responding with OK
        self.on_stop(request)

        response.status = remote_pb2.StopResponse.STATUS_OK

    def TeardownSession(self, request, context) -> remote_pb2.TeardownSessionResponse:
        response = remote_pb2.TeardownSessionResponse()
        with ResponseContext(response, request):
            self.logger.info(f'Tearing down session {request.session_id}')
            with self._lock:
                self._teardown_session_implementation(request, response)
        return response

    def _teardown_session_implementation(self, request, response):
        # deletes the current session, as it is no longer needed
        if request.session_id in self._threads_by_session_id:
            # remove all knowledge of session
            del self._threads_by_session_id[request.session_id]
            if request.session_id in self._ticks_running:
                self._ticks_running.remove(request.session_id)

            # run subclass code before responding with OK
            self.on_deinit(request)

            response.status = remote_pb2.TeardownSessionResponse.STATUS_OK
        else:  # edge case where the session was not found
            response.status = remote_pb2.TeardownSessionResponse.STATUS_INVALID_SESSION_ID

    def GetRemoteMissionServiceInfo(self, request, context) -> remote_pb2.GetRemoteMissionServiceInfoResponse:
        response = remote_pb2.GetRemoteMissionServiceInfoResponse()
        with ResponseContext(response, request):
            self.logger.info('Giving RMS info')
            with self._lock:
                self._get_remote_mission_service_info_implementation(request, response)
        return response

    def _get_remote_mission_service_info_implementation(self, request, response):
        # run subclass code before continuing
        self.on_params_requested(request)

        response.custom_params.CopyFrom(self.param_manager.spec)

    # ----------------- below methods should be overridden by subclass -----------------

    def on_init(self, request):
        # run when Spot is requesting to start a session
        pass

    def on_start(self, request):
        # run when Spot is requesting to run the action's code
        pass

    def on_stop(self, request):
        # run when Spot is requesting to stop the action's code
        pass

    def on_deinit(self, request):
        # run when Spot is requesting to destroy a session
        pass

    def on_params_requested(self, request):
        # run when Spot wants info about the params offered
        pass
