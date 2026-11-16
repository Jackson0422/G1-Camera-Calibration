"""Independent robot-side OAK publisher. Never restarts a factory service."""
import argparse
import sys
import threading
import time
import cv2
from ..common import ROOT, load_yaml
from ..hardware.oak_camera import OakCamera


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', help='Optional YAML config; requires PyYAML on the robot')
    p.add_argument('--bind', required=True, help='Explicit robot IP: e.g. tcp://192.168.123.164:5567')
    p.add_argument('--duration', type=float, default=0, help='0 means until Ctrl+C')
    p.add_argument('--exit-on-stdin-eof', action='store_true',
                   help='Stop when stdin closes, e.g. when the controlling SSH session ends')
    a = p.parse_args()
    import zmq
    import msgpack
    stop = threading.Event()
    if a.exit_on_stdin_eof:
        threading.Thread(target=lambda: (sys.stdin.buffer.read(), stop.set()), daemon=True).start()
    c = load_yaml(a.config)['oak'] if a.config else dict(width=1920, height=1080, fps=15,
        color_key='oak_rgb', optical_frame='oak_color_optical_calibrated')
    camera = None
    context = zmq.Context()
    sock = context.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.bind(a.bind) # Fails if port owned; never kills its owner.
        camera = OakCamera(c)
        print(f'OAK ready: serial={camera.serial}, board={camera.board_name}, product={camera.product_name}, DepthAI={camera.dai.__version__}', flush=True)
        start = time.monotonic()
        while not stop.is_set() and (a.duration == 0 or time.monotonic()-start < a.duration):
            f = camera.read()
            ok, image = cv2.imencode('.jpg', f.bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError('JPEG encoding failed')
            key = c.get('color_key', 'oak_rgb')
            intr = dict(width=f.bgr.shape[1], height=f.bgr.shape[0], fx=float(f.K[0, 0]), fy=float(f.K[1, 1]),
                        cx=float(f.K[0, 2]), cy=float(f.K[1, 2]), distortion_model='rational_polynomial',
                        distortion_coefficients=f.distortion.tolist())
            sock.send(msgpack.packb(dict(images={key: image.tobytes()}, timestamps={key: f.source_timestamp_s},
                timestamp_domains={key: f.source_clock}, frame_numbers={key: f.sequence},
                depth_metadata=dict(serial=f.serial, color_intrinsics=intr)), use_bin_type=True))
    finally:
        if camera:
            camera.close()
        sock.close()
        context.term()


if __name__ == '__main__':
    main()
