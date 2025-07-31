import atexit
import platform
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Union, Type, List, Optional, Dict

import bosdyn
import cv2
import numpy as np
import torch
from bosdyn.api import header_pb2
from bosdyn.api.network_compute_bridge_pb2 import NetworkComputeResponse, WorkerComputeResponse, \
    ListAvailableModelsResponse, ModelData, NetworkComputeRequest, WorkerComputeRequest
from bosdyn.api.network_compute_bridge_service_pb2_grpc import NetworkComputeBridgeWorkerServicer, \
    add_NetworkComputeBridgeWorkerServicer_to_server
from bosdyn.client.directory import DirectoryClient
from bosdyn.client.directory_registration import DirectoryRegistrationClient
from bosdyn.client.util import GrpcServiceRunner

from spot_tools.common import DirectoryServiceRegistration, BoundingBox
from spot_tools.spot import Spot


class _NetworkComputeServerWorker:
    def __init__(self):
        pass


@dataclass
class Result:
    label: str
    confidence: float
    bounding_box: BoundingBox


@dataclass
class ResultCollection:
    results: List[Result]
    annotated_image: np.ndarray


class _Flow(ABC):
    @abstractmethod
    def process(self) -> ResultCollection:
        ...


class BasicFlow(_Flow):
    def __init__(self, model_path: Union[Path, str], confidence_threshold: float):
        pass

    def process(self) -> ResultCollection:
        pass


class PipelineFlow(_Flow):
    pass


class NoopFlow(_Flow):
    pass


class NoFlowsAdded(Exception):
    pass


class ReceivedTooManyImages(Exception):
    pass


class NetworkComputeServer(NetworkComputeBridgeWorkerServicer):
    def __init__(self, spot: Spot, registration: DirectoryServiceRegistration):
        self._spot = spot
        self._registration = registration

        self._initial_connection = Event()
        self._server: Optional[GrpcServiceRunner] = None

        self._registered_flows: Dict[str, _Flow] = {}

        # find appropriate device to run models on
        if platform.system() == 'Darwin':
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

    def register_flow(self, name: str, flow: _Flow):
        self._registered_flows[name] = flow

        # then, perform first time inference on each model so subsequent requests are fast
        # print(f'Preloading model {model_path.stem}...')
        # _ = model.predict(np.zeros((640, 640, 3)), device=self.device)

    def start(self):
        if not self._registered_flows:
            raise NoFlowsAdded('No models have been added to the server.')

        directory_client = self._spot.clients[DirectoryClient]
        directory_registration_client = self._spot.clients[DirectoryRegistrationClient]

        # Check to see if a service is already registered with our name
        services = directory_client.list()
        for s in services:
            if s.name == self._registration.name:
                print(f"Existing service with name, \"{self._registration.name}\", removing it.")
                directory_registration_client.unregister(self._registration.name)
                break

        # Register service
        server_ip = self._registration.ip or bosdyn.client.common.get_self_ip(self._spot.robot.address)
        print(
            f'Attempting to register {server_ip}:{self._registration.port} onto {self._spot.robot.address} directory...')
        directory_registration_client.register(
            self._registration.name,
            "bosdyn.api.NetworkComputeBridgeWorker",
            f'{self._registration.name}.spot.robot',
            server_ip,
            self._registration.port
        )
        print('Successfully registered service.')

        # spin up server
        self._server = GrpcServiceRunner(
            service_servicer=self,
            add_servicer_to_server_fn=add_NetworkComputeBridgeWorkerServicer_to_server,
            port=self._registration.port
        )
        print('Started NetworkComputeBridgeWorker gRPC server.')
        atexit.register(self.stop)

        # print loaded models
        print('Loaded models:')
        print(*(f'\t{i}. {model}' for i, model in enumerate(self.models, 1)), sep='\n')

        # show success message
        print('Successfully established a NetworkComputeServer.')

    def stop(self):
        self._server.stop()
        print('Stopped NetworkComputeBridgeWorker gRPC server.')

    # ---- NetworkComputeBridgeWorkerServicer methods ------------------------------------------------

    def NetworkCompute(self, request, context):
        print('Got NetworkCompute request')
        try:
            return self._process_network_compute_request(request)
        except Exception as e:
            traceback.print_exception(e)
            return self._create_error_response(NetworkComputeResponse, 'A Python error occurred.')

    def WorkerCompute(self, request, context):
        print('Got WorkerCompute request')
        try:
            return self._process_worker_compute_request(request)
        except Exception as e:
            traceback.print_exception(e)
            return self._create_error_response(WorkerComputeResponse, 'A Python error occurred.')

    def ListAvailableModels(self, request, context):
        print('Got ListAvailableModels request')
        if not self._initial_connection.is_set():
            self._initial_connection.set()
        return self._process_list_available_models_request()

    # ------------------------------------------------------------------------------------------------

    @staticmethod
    def _create_error_response(type_: Union[Type[NetworkComputeResponse], Type[WorkerComputeResponse]],
                               message: str) -> NetworkComputeResponse:
        out_proto = type_()
        out_proto.header.error.code = header_pb2.CommonError.CODE_INVALID_REQUEST
        out_proto.header.error.message = message
        return out_proto

    def _process_list_available_models_request(self) -> ListAvailableModelsResponse:
        out_proto = ListAvailableModelsResponse()
        for model_name, model_flow in self.models.items():
            out_proto.models.data.append(ModelData(
                model_name=model_name,
                available_labels=list(model_flow.model.names.values())
            ))
        return out_proto

    def _process_network_compute_request(self, request: NetworkComputeRequest) -> NetworkComputeResponse:
        # first, perform extraction of model name and image from request
        model_name = request.input_data.model_name
        image = np.frombuffer(request.input_data.image.data, dtype=np.uint8)
        image = cv2.imdecode(image, -1)

    def _process_worker_compute_request(self, request: WorkerComputeRequest) -> WorkerComputeResponse:
        # first, perform extraction of model name and image from request
        model_name = request.input_data.parameters.model_name
        if len(request.input_data.images) != 1:
            print(err_str := f'Expected 1 image, received {len(request.input_data.images)}')
            raise ReceivedTooManyImages(err_str)
        image = np.frombuffer(request.input_data.images[0].shot.image.data, dtype=np.uint8)
        image = cv2.imdecode(image, -1)

        # next, run the flow on the model
