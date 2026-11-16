"""Receive inspected SONIC/metric-tap protocol; never open/restart RealSense."""
import base64
import time
import cv2
import numpy as np
from .base import Frame


def decode_packet(payload, config, received=None):
    import msgpack
    m = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    key = config.get('color_key', 'ego_view')
    raw = m['images'][key]
    legacy = isinstance(raw, str)
    if legacy:
        raw = base64.b64decode(raw, validate=True)
    elif not raw.startswith((b'\xff\xd8', b'\x89PNG')):
        raw, legacy = base64.b64decode(raw, validate=True), True
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Invalid encoded RGB')
    if legacy:  # legacy SONIC feeds RGB directly to imencode
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    meta = m.get('depth_metadata', {})
    intr = meta.get('color_intrinsics')
    if not intr:
        raise ValueError('Stream lacks metric color intrinsics; use existing 5556 side channel')
    if (intr['width'], intr['height']) != (image.shape[1], image.shape[0]):
        raise ValueError('Intrinsics resolution does not match image')
    dist = np.asarray(intr['distortion_coefficients'], dtype=float)
    model = intr['distortion_model'].lower().split('.')[-1]
    if model not in ('brown_conrady', 'plumb_bob', 'rational_polynomial', 'none', 'pinhole') and not (model == 'inverse_brown_conrady' and np.all(dist == 0)):
        raise ValueError(f'Unsupported distortion model: {model}')
    serial = str(meta.get('serial', ''))
    if config.get('expected_serial') and serial != config['expected_serial']:
        raise ValueError('Camera serial differs from configured identity')
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]], float)
    return Frame(image, float(m['timestamps'][key]), str(m.get('timestamp_domains', {}).get(key, '')),
                 time.monotonic() if received is None else received, config['optical_frame'], serial,
                 K, dist, int(m['frame_numbers'][key])).validate()


class G1Camera:
    def __init__(self, config):
        import zmq
        self.config, self.zmq = config, zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.MAXMSGSIZE, 16_000_000)
        self.socket.connect(config['endpoint'])
        self.previous = None

    def read(self, timeout_s=3.):
        deadline = time.monotonic()+timeout_s
        while time.monotonic() < deadline:
            if not self.socket.poll(max(1, int((deadline-time.monotonic())*1000))):
                break
            raw = self.socket.recv()
            frame = decode_packet(raw, self.config, time.monotonic())
            current = (frame.source_clock, frame.source_timestamp_s, frame.sequence)
            if self.previous:
                if current[0] != self.previous[0] or current[1] < self.previous[1] or current[2] < self.previous[2]:
                    raise RuntimeError('Source restarted or clock changed; begin a new collection session')
                if current[1] == self.previous[1] or current[2] == self.previous[2]:
                    continue
            self.previous = current
            return frame
        raise TimeoutError('Camera disconnected, frozen, or endpoint unavailable')

    def close(self):
        self.socket.close()
        self.context.term()
