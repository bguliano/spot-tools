import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

import cv2
import numpy as np
from bosdyn.api.geometry_pb2 import Polygon, SE2VelocityLimit
from bosdyn.client.math_helpers import SE2Velocity
from bosdyn.client.robot_command import RobotCommandBuilder


# ---- Dataclasses -------------------------------------------------------------------------------

@dataclass
class Point:
    x: int
    y: int

    def __init__(self, x: float, y: float):
        self.x = int(x)
        self.y = int(y)

    def as_tuple(self) -> tuple[int, int]:
        return self.x, self.y


@dataclass
class BoundingBox:
    tl: Point
    br: Point

    @classmethod
    def from_coordinates(cls, coordinates: Polygon) -> Self:
        return cls(
            tl=Point(coordinates.vertexes[0].x, coordinates.vertexes[0].y),
            br=Point(coordinates.vertexes[2].x, coordinates.vertexes[2].y)
        )

    def as_tuple(self) -> tuple[int, int, int, int]:
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


def pose_dist(pose1, pose2) -> float:
    diff_vec = [pose1.x - pose2.x, pose1.y - pose2.y, pose1.z - pose2.z]
    return float(np.linalg.norm(diff_vec))


def get_walking_params(max_linear_vel: float, max_rotation_vel: float):
    max_vel_se2 = SE2Velocity(x=max_linear_vel, y=max_linear_vel, angular=max_rotation_vel)
    vel_limit = SE2VelocityLimit(max_vel=max_vel_se2)
    params = RobotCommandBuilder.mobility_params()
    params.vel_limit.CopyFrom(vel_limit)
    return params

# ------------------------------------------------------------------------------------------------
