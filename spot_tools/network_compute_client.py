import time
from dataclasses import dataclass

import cv2
import numpy as np
from bosdyn.api.image_pb2 import ImageRequest, Image, ImageResponse
from bosdyn.api.network_compute_bridge_pb2 import ImageSourceAndService, NetworkComputeInputData, \
    NetworkComputeServerConfiguration, NetworkComputeRequest
from bosdyn.client import frame_helpers
from bosdyn.client.directory import DirectoryClient
from bosdyn.client.math_helpers import SE3Pose, Quat
from bosdyn.client.network_compute_bridge_client import NetworkComputeBridgeClient, ExternalServerError
from bosdyn.client.robot_state import RobotStateClient
from google.protobuf.wrappers_pb2 import StringValue, FloatValue

from spot_tools.common import BoundingBox, SpotImageSource, pose_dist
from spot_tools.spot import Spot


class CannotFindServiceError(Exception):
    pass


class CannotConnectToServerError(Exception):
    pass


@dataclass
class ModelData:
    name: str
    labels: list[str]


@dataclass
class InferenceObject:
    name: str
    confidence: float
    bounding_box: BoundingBox
    image_response: ImageResponse  # for more fine-grained information

    # sometimes these cannot be determined by Spot
    vision_tform_obj: SE3Pose | None
    distance: float | None


@dataclass
class InferenceResult:
    image_source_name: str
    image: np.ndarray  # rotated
    annotated_image: np.ndarray
    objects: list[InferenceObject]

    def get_first(self, label: str) -> InferenceObject | None:
        return next((obj for obj in self.objects if obj.name == label), None)

    def get_closest(self, label: str) -> InferenceObject | None:
        matching = (obj for obj in self.objects if obj.name == label)
        return min(matching, key=lambda obj: obj.distance, default=None)


@dataclass
class InferenceResultCollection:
    results: list[InferenceResult]

    def get_first(self, label: str) -> InferenceObject | None:
        return next((result.get_first(label) for result in self.results), None)

    def get_closest(self, label: str) -> InferenceObject | None:
        initial = (result.get_closest(label) for result in self.results)
        return min(initial, key=lambda obj: obj.distance, default=None)


def _annotate_image(image: np.ndarray, label: str, confidence: float, bounding_box: BoundingBox) -> np.ndarray:
    # draw bounding box
    cv2.rectangle(image, bounding_box.tl.as_tuple(), bounding_box.br.as_tuple(), (0, 255, 0), 2)

    # draw caption
    caption = f'{label} {confidence:.2f}'
    (_, label_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    org = (bounding_box.min_x, max(bounding_box.min_y - baseline, label_h + baseline))
    cv2.putText(image, caption, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # output current stage of annotated image
    return image


class NetworkComputeClient:
    def __init__(self, spot: Spot, service_name: str):
        self.spot = spot
        self.service_name = service_name

        # make sure directory contains this service
        if self.service_name not in self.get_all_servers():
            raise CannotFindServiceError(f'Cannot find {self.service_name} in robot directory. Please try again.')

        # make sure we can actually connect to this service
        try:
            _ = self.get_models()
        except ExternalServerError:
            raise CannotConnectToServerError(f'Cannot connect to {self.service_name}. Please try again.')

        # for showing annotated images
        self._cv2_window_name: str | None = None

    def enable_showing_annotated_images(self, window_name: str) -> None:
        self._cv2_window_name = window_name
        cv2.namedWindow(self._cv2_window_name)
        cv2.waitKey(500)

    def disable_showing_annotated_images(self) -> None:
        if self._cv2_window_name is None:
            return

        cv2.destroyWindow(self._cv2_window_name)
        self._cv2_window_name = None

    def get_all_servers(self) -> list[str]:
        directory_client = self.spot.clients[DirectoryClient]
        dir_list = directory_client.list()
        return [
            service.name
            for service in dir_list
            if service.type == 'bosdyn.api.NetworkComputeBridgeWorker'
        ]

    def get_models(self) -> list[ModelData]:
        network_compute_client = self.spot.clients[NetworkComputeBridgeClient]
        response = network_compute_client.list_available_models(self.service_name)
        return [
            ModelData(
                name=model.model_name,
                labels=list(model.available_labels)
            )
            for model in response.models.data
        ]

    def perform_inspection(self, model_name: str, image_source: SpotImageSource, color_image: bool = True,
                           image_quality: int = 100, whitelist_labels: list[str] | None = None) -> InferenceResult:
        # generate image request using parameters
        pf = Image.PixelFormat.PIXEL_FORMAT_RGB_U8 if color_image else Image.PixelFormat.PIXEL_FORMAT_GREYSCALE_U8
        image_request = ImageRequest(
            image_source_name=image_source,
            image_format=Image.Format.FORMAT_JPEG,
            pixel_format=pf,
            quality_percent=image_quality
        )
        image_source_and_service = ImageSourceAndService(image_request=image_request)

        # combine image request and model name (also pack image source to allow for proper rotation in server)
        input_data = NetworkComputeInputData(
            image_source_and_service=image_source_and_service,
            model_name=model_name
        )
        input_data.other_data.Pack(StringValue(value=image_source))

        # final request components
        server_data = NetworkComputeServerConfiguration(service_name=self.service_name)
        network_compute_request = NetworkComputeRequest(
            input_data=input_data,
            server_config=server_data
        )

        # send request
        network_compute_bridge_client = self.spot.clients[NetworkComputeBridgeClient]
        response = network_compute_bridge_client.network_compute_bridge_command(network_compute_request)

        # first, extract image
        image = self.spot.image.process_image_response(response)

        # prepare robot transformation state for later calculations
        state = self.spot.clients[RobotStateClient].get_robot_state()
        transforms_snapshot = state.kinematic_state.transforms_snapshot

        # iterate over each detection
        annotated_image = image.copy()
        inference_objects: list[InferenceObject] = []
        for obj in response.object_in_image:
            # skip if the label is not whitelisted
            if whitelist_labels is not None and obj.name not in whitelist_labels:
                continue

            # extract confidence from additional_properties
            conf_proto = FloatValue()
            obj.additional_properties.Unpack(conf_proto)
            confidence = conf_proto.value

            # calculate bounding box using handy classmethod constructor
            bounding_box = BoundingBox.from_coordinates(obj.image_properties.coordinates)

            # annotate using object_in_image
            annotated_image = _annotate_image(annotated_image, obj.name, confidence, bounding_box)

            # calculate transform from vision frame to object and its distance
            vision_tform_obj = frame_helpers.get_a_tform_b(
                obj.transforms_snapshot,
                frame_helpers.VISION_FRAME_NAME,
                obj.image_properties.frame_name_image_coordinates
            )

            # calculate distance from the robot if a transform exists
            if vision_tform_obj is None:
                obj_distance = None
            else:
                body_tform_vision = frame_helpers.get_a_tform_b(
                    transforms_snapshot,
                    frame_helpers.BODY_FRAME_NAME,
                    frame_helpers.VISION_FRAME_NAME
                )
                origin_pose = SE3Pose(0, 0, 0, Quat())
                obj_distance = pose_dist(origin_pose, body_tform_vision * vision_tform_obj)

            # add inference object to list
            inference_objects.append(InferenceObject(
                name=obj.name,
                confidence=confidence,
                bounding_box=bounding_box,
                image_response=response.image_response,
                vision_tform_obj=vision_tform_obj,
                distance=obj_distance
            ))

        # write name of camera source on the image
        text_x, text_y = 10, annotated_image.shape[0] - 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated_image, image_source, (text_x, text_y), font, 1, (255, 255, 255), 1)

        # show annotated image if desired
        if self._cv2_window_name is not None:
            cv2.imshow(self._cv2_window_name, annotated_image)
            cv2.waitKey(1)

        # output final inference result
        return InferenceResult(
            image_source_name=image_source,
            image=image,
            annotated_image=annotated_image,
            objects=inference_objects
        )

    def perform_360_inspection(self, model_name: str, color_image: bool = True, image_quality: int = 100,
                               whitelist_labels: list[str] | None = None) -> InferenceResultCollection:
        all_image_sources = [
            SpotImageSource.FRONT_LEFT,
            SpotImageSource.FRONT_RIGHT,
            SpotImageSource.LEFT,
            SpotImageSource.RIGHT,
            SpotImageSource.BACK
        ]
        return InferenceResultCollection(
            results=[
                self.perform_inspection(model_name, source, color_image, image_quality, whitelist_labels)
                for source in all_image_sources
            ]
        )


def main() -> None:
    spot = Spot(
        authentication_file='../authentication.json'
    )
    client = NetworkComputeClient(
        spot=spot,
        service_name='bg-spot-fetch-3'
    )

    start_time = time.time()
    for inference_result in client.perform_360_inspection('yolov8n').results:
        print(f'Inference result arrived at {time.time() - start_time:.2f}s: {len(inference_result.objects)} objects')


if __name__ == '__main__':
    main()
