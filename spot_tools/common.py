from dataclasses import dataclass
from enum import Enum
from typing import Union, Tuple

import cv2
import math
import numpy as np
from bosdyn.api.geometry_pb2 import Polygon, SE2VelocityLimit
from bosdyn.api.image_pb2 import ImageResponse
from bosdyn.api.network_compute_bridge_pb2 import NetworkComputeRequest
from bosdyn.client.math_helpers import SE2Velocity, Vec2
from bosdyn.client.robot_command import RobotCommandBuilder
from google.protobuf.wrappers_pb2 import StringValue

# part of requirements_full.txt
try:
    from ultralytics.engine.results import Results
except ImportError:
    pass


class StrEnum(str, Enum):
    def __str__(self):
        return self.value

    def __format__(self, spec):
        # so format(member, fmt) == format(member.value, fmt)
        return format(self.value, spec)


# ---- Dataclasses -------------------------------------------------------------------------------

@dataclass
class Point:
    x: int
    y: int

    def __init__(self, x: float, y: float):
        self.x = int(x)
        self.y = int(y)

    def as_tuple(self) -> Tuple[int, int]:
        return self.x, self.y

    def as_vec2(self) -> Vec2:
        return Vec2(x=self.x, y=self.y)


@dataclass
class BoundingBox:
    tl: Point
    br: Point

    @classmethod
    def from_bosdyn_coordinates(cls, coordinates: Polygon) -> 'BoundingBox':
        return cls(
            tl=Point(coordinates.vertexes[0].x, coordinates.vertexes[0].y),
            br=Point(coordinates.vertexes[2].x, coordinates.vertexes[2].y)
        )

    @classmethod
    def from_results(cls, results: 'Results', index: int) -> 'BoundingBox':
        xyxy = results.boxes[index].xyxy.tolist()
        return cls(
            tl=Point(xyxy[0], xyxy[1]),
            br=Point(xyxy[2], xyxy[3])
        )

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.tl.as_tuple() + self.br.as_tuple()

    @property
    def center(self) -> Point:
        x = math.fabs(self.max_x - self.min_x) / 2.0 + self.min_x
        y = math.fabs(self.max_y - self.min_y) / 2.0 + self.min_y
        return Point(x, y)

    @property
    def min_x(self) -> int:
        return min(self.as_tuple()[::2])

    @property
    def min_y(self) -> int:
        return min(self.as_tuple()[1::2])

    @property
    def max_x(self) -> int:
        return max(self.as_tuple()[::2])

    @property
    def max_y(self) -> int:
        return max(self.as_tuple()[1::2])

    @property
    def tr(self) -> Point:
        return Point(self.br.x, self.tl.y)

    @property
    def bl(self) -> Point:
        return Point(self.tl.x, self.br.y)

    def annotate_image(self, image: np.ndarray, caption: str, color: Tuple[int, int, int], font_scale: int,
                       thickness: int):
        cv2.rectangle(image, self.tl.as_tuple(), self.br.as_tuple(), color, thickness)
        cv2.putText(image, caption, (self.min_x, self.max_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


@dataclass
class DirectoryServiceRegistration:
    name: str
    ip: Union[str, None] = None
    port: int = 50051


# ------------------------------------------------------------------------------------------------

# ---- Enums -------------------------------------------------------------------------------------

class SpotImageSource(StrEnum):
    FRONT_LEFT = 'frontleft_fisheye_image'
    FRONT_RIGHT = 'frontright_fisheye_image'
    LEFT = 'left_fisheye_image'
    RIGHT = 'right_fisheye_image'
    BACK = 'back_fisheye_image'


# ------------------------------------------------------------------------------------------------

# ---- Functions ---------------------------------------------------------------------------------

def rotate_bd_image(image: np.ndarray, image_source_name: str) -> np.ndarray:
    if image_source_name[:5] == "front":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif image_source_name[:5] == "right":
        return cv2.rotate(image, cv2.ROTATE_180)
    return image


def rotate_bd_image_advanced(image: np.ndarray, image_source_name: str) -> np.ndarray:
    # accurately rotates so that all image content is level with the ground
    if image_source_name[:5] == "front":
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        angle = 12 if image_source_name[:9] == 'frontleft' else -12

        # calculate center to perform affine
        height, width = image.shape[:2]
        center = (width / 2, height / 2)

        # get the 2x3 rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # apply warpAffine using matrix
        return cv2.warpAffine(
            src=image,
            M=rotation_matrix,
            dsize=(width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT
        )

    elif image_source_name[:5] == "right":
        return cv2.rotate(image, cv2.ROTATE_180)

    return image


def process_image_response(image_response: ImageResponse) -> np.ndarray:
    image = np.frombuffer(image_response.shot.image.data, dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    image = rotate_bd_image(image, image_response.source.name)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def process_network_compute_request(request: NetworkComputeRequest) -> np.ndarray:
    # create image from bytes
    image = np.frombuffer(request.input_data.image.data, dtype=np.uint8)
    image = cv2.imdecode(image, -1)

    # image source name is packed in other_data
    image_source_name = StringValue()
    request.input_data.other_data.Unpack(image_source_name)
    image = rotate_bd_image(image, image_source_name.value)

    # convert to RGB if is grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image


def pose_dist(pose1, pose2) -> float:
    diff_vec = [pose1.x - pose2.x, pose1.y - pose2.y, pose1.z - pose2.z]
    return float(np.linalg.norm(diff_vec))


def get_walking_params(max_linear_vel: float, max_rotation_vel: float):
    max_vel_se2 = SE2Velocity(x=max_linear_vel, y=max_linear_vel, angular=max_rotation_vel)
    vel_limit = SE2VelocityLimit(max_vel=max_vel_se2)
    params = RobotCommandBuilder.mobility_params()
    params.vel_limit.CopyFrom(vel_limit)
    return params


def encode_to_jpeg(image: np.ndarray, quality: int = 80) -> bytes:
    _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes()

# ------------------------------------------------------------------------------------------------
