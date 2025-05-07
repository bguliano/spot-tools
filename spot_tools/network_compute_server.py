import atexit
import traceback
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import bosdyn.client.common
import cv2
import numpy as np
from bosdyn.api import header_pb2, network_compute_bridge_pb2
from bosdyn.api import image_pb2
from bosdyn.api.network_compute_bridge_pb2 import ListAvailableModelsResponse, ModelData, NetworkComputeRequest, \
    NetworkComputeResponse
from bosdyn.api.network_compute_bridge_service_pb2_grpc import NetworkComputeBridgeWorkerServicer, \
    add_NetworkComputeBridgeWorkerServicer_to_server
from bosdyn.client.directory import DirectoryClient
from bosdyn.client.directory_registration import DirectoryRegistrationClient
from bosdyn.client.server_util import GrpcServiceRunner
from google.protobuf.wrappers_pb2 import StringValue, FloatValue
from ultralytics import YOLO
from ultralytics.engine.results import Results

from spot_tools.common import rotate_bd_image, process_network_compute_request
from spot_tools.spot import Spot


@dataclass
class DirectoryServiceRegistration:
    name: str
    ip: str | None = None
    port: int = 50051


class NetworkComputeServer(NetworkComputeBridgeWorkerServicer):
    def __init__(self, spot: Spot, registration: DirectoryServiceRegistration, models_path: str | Path):
        self._spot = spot
        self._initial_connection = Event()

        directory_client = self._spot.clients[DirectoryClient]
        directory_registration_client = self._spot.clients[DirectoryRegistrationClient]

        # Check to see if a service is already registered with our name
        services = directory_client.list()
        for s in services:
            if s.name == registration.name:
                print(f"Existing service with name, \"{registration.name}\", removing it.")
                directory_registration_client.unregister(registration.name)
                break

        # Register service
        server_ip = registration.ip or bosdyn.client.common.get_self_ip(self._spot.robot.address)
        print(f'Attempting to register {server_ip}:{registration.port} onto {self._spot.robot.address} directory...')
        directory_registration_client.register(
            registration.name,
            "bosdyn.api.NetworkComputeBridgeWorker",
            f'{registration.name}.spot.robot',
            server_ip,
            registration.port
        )
        print('Successfully registered service.')

        # spin up server
        self._server = GrpcServiceRunner(
            service_servicer=self,
            add_servicer_to_server_fn=add_NetworkComputeBridgeWorkerServicer_to_server,
            port=registration.port
        )
        print('Started NetworkComputeBridgeWorker gRPC server.')
        atexit.register(self.shutdown)

        # first, locate all pt files in models_path
        detected_models = [
            model_path for model_path in Path(models_path).iterdir()
            if model_path.suffix == '.pt'
        ]

        # then, create the actual YOLO models
        self.models = {
            model_path.stem: YOLO(model_path).to('mps')
            for model_path in detected_models
        }

        # print loaded models
        print('Loaded models:')
        print(*(f'\t{i}. {model}' for i, model in enumerate(self.models, 1)), sep='\n')

        # show success message
        print('Successfully established a NetworkComputeServer.')

    def shutdown(self):
        self._server.stop()
        print('Stopped NetworkComputeBridgeWorker gRPC server.')

    # ---- NetworkComputeBridgeWorkerServicer methods ------------------------------------------------

    def NetworkCompute(self, request, context):
        print('Got NetworkCompute request')
        try:
            return self._process_network_compute_request(request)
        except Exception as e:
            traceback.print_exception(e)
            return self._create_error_response('A Python error occurred.')

    def WorkerCompute(self, request, context):
        print('Got WorkerCompute request')
        return self._create_error_response('Not implemented yet.')

    def ListAvailableModels(self, request, context):
        print('Got ListAvailableModels request')
        if not self._initial_connection.is_set():
            self._initial_connection.set()
        return self._process_list_available_models_request()

    # ------------------------------------------------------------------------------------------------

    @staticmethod
    def _create_error_response(message: str) -> NetworkComputeResponse:
        out_proto = NetworkComputeResponse()
        out_proto.header.error.code = header_pb2.CommonError.CODE_INVALID_REQUEST
        out_proto.header.error.message = message
        return out_proto

    def _process_list_available_models_request(self) -> ListAvailableModelsResponse:
        out_proto = ListAvailableModelsResponse()
        for model_name, model in self.models.items():
            out_proto.models.data.append(ModelData(
                model_name=model_name,
                available_labels=list(model.names.values())
            ))
        return out_proto

    def _process_network_compute_request(self, request: NetworkComputeRequest) -> NetworkComputeResponse:
        # create initial response
        out_proto = NetworkComputeResponse()

        # ensure the model requested is actually loaded
        if request.input_data.model_name not in self.models:
            print(err_str := f'Cannot find model "{request.input_data.model_name}" in loaded models.')
            return self._create_error_response(err_str)

        # grab requested model
        current_model = self.models[request.input_data.model_name]

        # extract image
        image = process_network_compute_request(request)

        # run prediction
        results: Results = current_model.predict(image)[0].cpu().numpy()
        boxes = results.boxes

        # iterate through each detection
        # it is easier to use a range since iterating through results produces tensors of len 1
        for i in range(len(results)):
            # skip if confidence is lower than requested
            confidence = float(boxes.conf[i])
            if confidence < request.input_data.min_confidence:
                continue

            # get label
            label_id = int(boxes.cls[i])
            label = results.names.get(label_id)

            print('Found object with label: "' + label + '" and score: ' + str(confidence))

            # extract coordinates and class of box
            box_xyxy = boxes.xyxy[i].tolist()

            # create ObjectInImage
            out_object_in_image = out_proto.object_in_image.add()
            out_object_in_image.name = label

            # add vertex protos
            vertex1 = out_object_in_image.image_properties.coordinates.vertexes.add()  # tl
            vertex1.x = box_xyxy[0]
            vertex1.y = box_xyxy[1]
            vertex2 = out_object_in_image.image_properties.coordinates.vertexes.add()  # tr
            vertex2.x = box_xyxy[2]
            vertex2.y = box_xyxy[1]
            vertex3 = out_object_in_image.image_properties.coordinates.vertexes.add()  # br
            vertex3.x = box_xyxy[2]
            vertex3.y = box_xyxy[3]
            vertex4 = out_object_in_image.image_properties.coordinates.vertexes.add()  # bl
            vertex4.x = box_xyxy[0]
            vertex4.y = box_xyxy[3]

            # add confidence
            proto_confidence = FloatValue(value=confidence)
            out_object_in_image.additional_properties.Pack(proto_confidence)

        print(f'Found {len(results)} object(s)')

        out_proto.status = network_compute_bridge_pb2.NETWORK_COMPUTE_STATUS_SUCCESS
        return out_proto

    @staticmethod
    def debug_wait_forever():
        Event().wait()

    def wait_for_initial_connection(self):
        self._initial_connection.wait()


def main() -> None:
    spot = Spot(
        authentication_file='../authentication.json'
    )
    server = NetworkComputeServer(
        spot=spot,
        registration=DirectoryServiceRegistration(
            name='bg-spot-fetch-3'
        ),
        models_path='../models'
    )
    server.debug_wait_forever()


if __name__ == '__main__':
    main()
