from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass
class Frame:
    bgr: np.ndarray
    source_timestamp_s: float
    source_clock: str
    received_monotonic_s: float
    optical_frame: str
    serial: str
    K: np.ndarray
    distortion: np.ndarray
    sequence: int
    capture_monotonic_s: float | None = None
    timing_uncertainty_s: float | None = None

    def validate(self):
        h, w = self.bgr.shape[:2]
        if self.bgr.dtype != np.uint8 or self.bgr.shape != (h, w, 3):
            raise ValueError('Expected uint8 BGR image')
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all() or min(self.K[0, 0], self.K[1, 1]) <= 0 or not np.allclose(self.K[2], [0, 0, 1]):
            raise ValueError('Invalid camera matrix')
        if self.distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(self.distortion).all():
            raise ValueError('Invalid OpenCV pinhole distortion')
        if not np.isfinite([self.source_timestamp_s, self.received_monotonic_s]).all() or not self.source_clock or not self.optical_frame or not self.serial:
            raise ValueError('Missing capture identity/time')
        return self


class Camera(Protocol):
    def read(self, timeout_s=3.) -> Frame: ...
    def close(self): ...


def open_camera(config):
    if config['backend'] == 'network':
        from .g1_camera import G1Camera
        return G1Camera(config)
    if config['backend'] == 'depthai':
        from .oak_camera import OakCamera
        return OakCamera(config)
    raise ValueError('Unsupported camera backend')
