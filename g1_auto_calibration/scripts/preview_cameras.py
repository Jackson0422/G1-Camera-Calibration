import argparse
from contextlib import ExitStack
from pathlib import Path
import time
import cv2
from ..common import ROOT, load_yaml
from ..hardware.base import open_camera
from ..hardware.robot_oak_server import start_robot_server


def main():
    p = argparse.ArgumentParser(description='Read-only live preview or headless frame capture')
    p.add_argument('--cameras', default=str(ROOT/'config/cameras.yaml'))
    p.add_argument('--camera', choices=['d435', 'oak', 'all'], default='all')
    p.add_argument('--duration', type=float, default=30)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--output', default=str(ROOT/'data/preview'))
    a = p.parse_args()
    configs = load_yaml(a.cameras)
    names = list(configs) if a.camera == 'all' else [a.camera]
    Path(a.output).mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        readers = {}
        for name in names:
            server = start_robot_server(configs[name])
            if server is not None:
                stack.callback(server.stop)
            readers[name] = open_camera(configs[name])
            stack.callback(readers[name].close)
        start = time.monotonic()
        while time.monotonic()-start < a.duration:
            for name, reader in readers.items():
                f = reader.read()
                cv2.imwrite(str(Path(a.output)/(name+'_latest.png')), f.bgr)
                if not a.headless:
                    view = f.bgr.copy()
                    cv2.putText(view, f'{name} {f.serial} seq={f.sequence}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
                    cv2.imshow(name, view)
            if not a.headless and cv2.waitKey(1) & 255 == ord('q'):
                break
    if not a.headless:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
