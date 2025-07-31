import time

from bosdyn.client import frame_helpers

from spot_tools.spot import Spot


class Experiments:
    def __init__(self, spot: Spot):
        self.spot = spot

    def get_hand_pose(self, refresh_rate_secs: float = 0.1):
        while True:
            body_tform_hand = self.spot.get_parent_tform_frame(frame_helpers.HAND_FRAME_NAME)
            print(f'spot.arm.pose({body_tform_hand.x}, {body_tform_hand.y}, {body_tform_hand.z}, '
                  f'{body_tform_hand.rot.w}, {body_tform_hand.rot.x}, {body_tform_hand.rot.y}, '
                  f'{body_tform_hand.rot.z})' + ' ' * 5, end='\r')
            time.sleep(refresh_rate_secs)
