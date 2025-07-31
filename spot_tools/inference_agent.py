from pathlib import Path
from typing import Union, Optional, List

from spot_tools.common import SpotImageSource
from spot_tools.network_compute_client import NetworkComputeClient, InferenceResultCollection, InferenceResult
from spot_tools.network_compute_server import NetworkComputeServer, DirectoryServiceRegistration
from spot_tools.spot import Spot


# wrapper for combining a server and client in one application
class InferenceAgent:
    def __init__(self, spot: Spot, registration: DirectoryServiceRegistration, models_path: Union[str, Path], enable_cv2_window: bool):
        self._server = NetworkComputeServer(spot, registration, models_path)
        self._server.wait_for_initial_connection()
        self._client = NetworkComputeClient(spot, registration.name)
        if enable_cv2_window:
            self._client.enable_showing_annotated_images(registration.name)

    def perform_inspection(self, model_name: str, image_source: SpotImageSource, color_image: bool = True,
                           image_quality: int = 100, whitelist_labels: Optional[List[str]] = None) -> InferenceResult:
        return self._client.perform_inspection(model_name, image_source, color_image, image_quality, whitelist_labels)

    def perform_360_inspection(self, model_name: str, color_image: bool = True, image_quality: int = 100,
                               whitelist_labels: Optional[List[str]] = None) -> InferenceResultCollection:
        return self._client.perform_360_inspection(model_name, color_image, image_quality, whitelist_labels)