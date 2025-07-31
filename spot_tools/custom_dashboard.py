import time
from collections.abc import Callable
from typing import Iterable, Tuple

import bosdyn
import cv2
import numpy as np
from bosdyn.api import image_pb2, image_service_pb2_grpc
from bosdyn.client.directory_registration import DirectoryRegistrationClient, DirectoryRegistrationKeepAlive
from bosdyn.client.image_service_helpers import CameraInterface, CameraBaseImageServicer, VisualImageSource, \
    convert_RGB_to_grayscale
from bosdyn.client.server_util import GrpcServiceRunner

from spot_tools.common import DirectoryServiceRegistration
from spot_tools.param_manager import ParamManager, SpotParams, BoolParam
from spot_tools.spot import Spot

GetImageCallback = Callable[[SpotParams], np.ndarray]


class DashboardSource(CameraInterface):
    def __init__(self, name: str, *params):
        super().__init__()
        print(f'Initializing CV2Camera with name: {name}')

        self.name = name
        self._param_manager = ParamManager()
        if params:
            self._param_manager.add_params(*params)

        self.default_jpeg_quality = 80

        print('Collecting camera properties...')
        self.pixel_formats = []
        image = self.get_image_callback({})
        self.rows, self.cols, _ = image.shape
        if image.shape[2] == 1:
            self.pixel_formats = [
                image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8
            ]
        elif image.shape[2] == 3:
            self.pixel_formats = [
                image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8, image_pb2.Image.PIXEL_FORMAT_RGB_U8
            ]
        elif image.shape[2] == 4:
            self.pixel_formats = [
                image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8,
                image_pb2.Image.PIXEL_FORMAT_RGB_U8,
                image_pb2.Image.PIXEL_FORMAT_RGBA_U8
            ]
        print(f'Camera properties - Rows: {self.rows}, Cols: {self.cols}, Pixel formats: {self.pixel_formats}')

    def get_image_callback(self, params: SpotParams) -> np.ndarray:
        raise NotImplementedError

    def blocking_capture(self, *, custom_params=None, **kwargs) -> Tuple[np.ndarray, float]:
        if custom_params is None:
            params = {}
        else:
            params = self._param_manager.parse_request_params(custom_params)

        with self.capture_lock:
            image = self.get_image_callback(params)
            capture_time = time.time()

        return image, capture_time

    def image_decode(self, image_data, image_proto, image_req):
        pixel_format = image_req.pixel_format
        image_format = image_req.image_format

        # Determine the pixel format for the data.
        if image_data.shape[2] == 3:
            # RGB image.
            if pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                image_data = convert_RGB_to_grayscale(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
                image_proto.pixel_format = pixel_format
            else:
                image_proto.pixel_format = image_pb2.Image.PIXEL_FORMAT_RGB_U8
        elif image_data.shape[2] == 1:
            # Greyscale image.
            image_proto.pixel_format = image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8
        elif image_data.shape[2] == 4:
            # RGBA image.
            if pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                image_data = convert_RGB_to_grayscale(
                    cv2.cvtColor(image_data, cv2.COLOR_BGRA2RGB))
                image_proto.pixel_format = pixel_format
            else:
                image_proto.pixel_format = image_pb2.Image.PIXEL_FORMAT_RGBA_U8
        else:
            # The number of pixel channels did not match any of the known formats.
            image_proto.pixel_format = image_pb2.Image.PIXEL_FORMAT_UNKNOWN

        # Note, we are currently not setting any information for the transform snapshot or the frame
        # name for an image sensor since this information can't be determined with openCV.

        resize_ratio = image_req.resize_ratio
        quality_percent = image_req.quality_percent

        if resize_ratio < 0 or resize_ratio > 1:
            raise ValueError(f'Resize ratio {resize_ratio} is out of bounds.')

        if resize_ratio != 1.0 and resize_ratio != 0:
            image_proto.rows = int(image_proto.rows * resize_ratio)
            image_proto.cols = int(image_proto.cols * resize_ratio)
            image_data = cv2.resize(image_data, (image_proto.cols, image_proto.rows), interpolation=cv2.INTER_AREA)

        # Set the image data.
        if image_format == image_pb2.Image.FORMAT_RAW:
            image_proto.data = np.ndarray.tobytes(image_data)
            image_proto.format = image_pb2.Image.FORMAT_RAW
        elif image_format == image_pb2.Image.FORMAT_JPEG or image_format == image_pb2.Image.FORMAT_UNKNOWN or image_format is None:
            # If the image format is requested as JPEG or if no specific image format is requested, return
            # a JPEG. Since this service is for a webcam, we choose a sane default for the return if the
            # request format is unpopulated.
            quality = self.default_jpeg_quality
            if 0 < quality_percent <= 100:
                # A valid image quality percentage was passed with the image request,
                # so use this value instead of the service's default.
                quality = quality_percent
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            image_proto.data = cv2.imencode('.jpg', image_data, encode_param)[1].tobytes()
            image_proto.format = image_pb2.Image.FORMAT_JPEG
        else:
            # Unsupported format.
            raise Exception(f'Image format {image_pb2.Image.Format.Name(image_format)} is unsupported.')

    def to_image_source(self) -> VisualImageSource:
        return VisualImageSource(
            image_name=self.name,
            camera_interface=self,
            rows=self.rows,
            cols=self.cols,
            pixel_formats=self.pixel_formats,
            param_spec=self._param_manager.spec
        )


class CustomDashboard:
    def __init__(self, spot: Spot, registration: DirectoryServiceRegistration, sources: Iterable[DashboardSource]):
        print(f'Initializing CustomDashboard with service name: {registration.name}')
        self._spot = spot

        print('Creating image sources...')
        image_sources = [source.to_image_source() for source in sources]
        print('Image sources created successfully.')

        print('Setting up image servicer...')
        self._image_servicer = CameraBaseImageServicer(
            bosdyn_sdk_robot=spot.robot,
            service_name=registration.name,
            image_sources=image_sources,
            use_background_capture_thread=False
        )
        print('Image servicer setup complete.')

        print('Starting gRPC service runner...')
        add_servicer_to_server_fn = image_service_pb2_grpc.add_ImageServiceServicer_to_server
        self._grpc_service_runner = GrpcServiceRunner(
            self._image_servicer,
            add_servicer_to_server_fn,
            port=registration.port
        )
        print('gRPC service runner started.')

        print('Setting up directory registration...')
        directory_registration_client = self._spot.clients[DirectoryRegistrationClient]
        self._keep_alive = DirectoryRegistrationKeepAlive(directory_registration_client)
        print('Directory registration setup complete.')

        server_ip = registration.ip or bosdyn.client.common.get_self_ip(self._spot.robot.address)
        print(f'Attempting to register {server_ip}:{registration.port} onto {self._spot.robot.address} directory...')
        self._keep_alive.start(
            registration.name,
            'bosdyn.api.ImageService',
            f'{registration.name}.spot.robot',
            server_ip,
            registration.port
        )
        print('Successfully registered service.')
        print('CustomDashboard initialization complete.')

    def start(self):
        print('Starting gRPC service runner...')
        with self._keep_alive:
            self._grpc_service_runner.run_until_interrupt()


def main() -> None:
    spot = Spot(
        authentication_file='../authentication.json'
    )

    capture = cv2.VideoCapture(0)

    def get_image_callback(params: SpotParams) -> np.ndarray:
        return capture.read()[1]

    mac_camera = DashboardSource(
        'Mac Camera',
        BoolParam(
            key='one_of_test',
            display_name='Testing',
            description='',
            default_value=False
        )
    )
    mac_camera.get_image_callback = get_image_callback

    custom_dashboard = CustomDashboard(
        spot=spot,
        registration=DirectoryServiceRegistration(
            name='cv2-dashboard'
        ),
        sources=[mac_camera]
    )

    custom_dashboard.start()


if __name__ == '__main__':
    main()
