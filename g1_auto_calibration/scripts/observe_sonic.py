"""Observe selected wired camera streams and settled robot state without motion.

This command does not initialize SONIC, release another controller, publish a
joint command, or infer that a planned target has been reached. It records raw
candidate observations in independent camera subsets. The resulting directory
is deliberately not a hand-eye solver session.
"""

import argparse
from datetime import datetime, timezone
import ipaddress
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

from ..calibration.board_detector import BoardDetector
from ..common import ROOT, load_yaml
from ..hardware.base import open_camera
from ..hardware.unitree_state import UnitreeState
from ..observation.dual_capture import CameraWorker, ObservationRecorder, overlay_packet
from .check_hardware import model_from_config


def verify_wired_routes(camera_configs, interface, *, route_reader=None, sys_root='/sys/class/net'):
    """Require explicit camera IPv4 endpoints routed over a non-wireless interface."""
    interface_path = Path(sys_root) / interface
    if not interface_path.is_dir():
        raise ValueError(f'Configured wired interface does not exist: {interface}')
    if (interface_path / 'wireless').exists():
        raise ValueError('Configured robot interface is wireless')
    if route_reader is None:
        def route_reader(host):
            completed = subprocess.run(['ip', '-j', 'route', 'get', host],
                                       check=True, capture_output=True, text=True,
                                       timeout=3.)
            return json.loads(completed.stdout)
    evidence = {}
    for name, config in camera_configs.items():
        if config.get('backend') != 'network':
            raise ValueError('SONIC wired observation requires network camera backends')
        endpoint = urlsplit(config.get('endpoint', ''))
        if endpoint.scheme != 'tcp' or endpoint.hostname is None or endpoint.port is None:
            raise ValueError(f'Invalid TCP camera endpoint: {name}')
        address = ipaddress.ip_address(endpoint.hostname)
        if address.version != 4 or address.is_loopback or address.is_unspecified:
            raise ValueError('Camera endpoint must be a robot IPv4 address')
        routes = route_reader(str(address))
        if not isinstance(routes, list) or len(routes) != 1 or routes[0].get('dev') != interface:
            raise ValueError(f'{name} endpoint is not routed through {interface}')
        evidence[name] = dict(endpoint=config['endpoint'], route=routes[0],
                              interface=interface, interface_is_wireless=False)
    return evidence


def _positive(value):
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('Value must be positive and finite')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', default=str(ROOT / 'config/robot.yaml'))
    parser.add_argument('--cameras', default=str(ROOT / 'config/cameras.yaml'))
    parser.add_argument('--board', default=str(ROOT / 'config/board.yaml'))
    parser.add_argument('--camera', choices=['d435', 'oak', 'all'], default='all',
                        help='Use one camera for a separate calibration round, or preview both')
    parser.add_argument('--output', default=None,
                        help='New directory; existing sessions are never overwritten')
    parser.add_argument('--duration', type=_positive, default=60.)
    parser.add_argument('--startup-timeout', type=_positive, default=10.)
    parser.add_argument('--camera-stale-timeout', type=_positive, default=2.)
    parser.add_argument('--headless', action='store_true')
    arguments = parser.parse_args()
    robot_config, fk = model_from_config(arguments.robot)
    cameras = load_yaml(arguments.cameras)
    names = ('d435', 'oak') if arguments.camera == 'all' else (arguments.camera,)
    if not all(name in cameras for name in names):
        raise ValueError('The selected camera configuration is missing')
    cameras = {name: cameras[name] for name in names}
    board = load_yaml(arguments.board)
    route_evidence = verify_wired_routes(cameras, robot_config['interface'])
    output = arguments.output or str(ROOT / 'data' / (
        'sonic_observations_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')))
    recorder = ObservationRecorder(output, robot_config, cameras, board, fk=fk)
    recorder.session['wired_route_evidence'] = route_evidence
    recorder.session['requested_duration_s'] = arguments.duration
    recorder.session['selected_camera_names'] = list(names)
    recorder.flush()
    workers, state = {}, None
    last_seen, displayed, camera_health = {}, {}, {}
    failure = None
    try:
        state = UnitreeState(robot_config)
        workers = {name: CameraWorker(name, config, open_camera, BoardDetector(board))
                   for name, config in cameras.items()}
        for worker in workers.values():
            worker.start()
        deadline = time.monotonic() + arguments.startup_timeout
        while True:
            packets = {name: worker.snapshot() for name, worker in workers.items()}
            with state.lock:
                state_ready = bool(not state.error and state.states and
                                   time.monotonic() - state.states[-1].received_s <= .25)
            if (all(packet is not None and received_s is not None and
                    time.monotonic() - received_s <= arguments.camera_stale_timeout
                    for packet, _, received_s in packets.values())
                    and state_ready):
                break
            if time.monotonic() >= deadline:
                messages = {name: error or 'No valid frame received'
                            for name, (packet, error, _) in packets.items() if packet is None}
                if not state_ready:
                    messages['robot_state'] = state.error or 'No fresh PR LowState'
                raise TimeoutError(f'Selected-camera/state preflight failed: {messages}')
            time.sleep(.03)
        print(f'Selected wired cameras {list(names)} and PR robot state are available.')
        print('Read-only observation started. No controller is started and no commands are sent.')
        print('Stationary intervals are observations, not verified planned-pose arrivals.')
        print(f'Output: {Path(output).resolve()}')
        if not arguments.headless:
            for index, name in enumerate(workers):
                title = 'Read-only SONIC observation: ' + name
                cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(title, 900, 600)
                cv2.moveWindow(title, 20 + 920 * index, 80)
            print('Live video windows: Q or close a window to exit the selected streams.')
        started, last_flush = time.monotonic(), time.monotonic()
        while time.monotonic() - started < arguments.duration:
            now = time.monotonic()
            for name, worker in workers.items():
                packet, error, received_s = worker.snapshot()
                healthy = received_s is not None and now - received_s <= arguments.camera_stale_timeout
                camera_health[name] = dict(healthy=healthy, last_frame_received_s=received_s,
                                           error=error)
                if packet is None:
                    continue
                identity = (packet.frame.source_clock, packet.frame.source_timestamp_s,
                            packet.frame.sequence)
                if identity != last_seen.get(name) and healthy:
                    last_seen[name] = identity
                    record = recorder.observe(name, packet, state, now=now)
                    if record:
                        print(f'{name}: interval {record["stationary_interval_id"]}, '
                              f'candidate {record["index"]}, '
                              f'board_quality_passed={record["observation_quality_passed"]}')
                    view = overlay_packet(name, packet, recorder.last_status.get(name, ''))
                    displayed[name] = view
                    cv2.imwrite(str(Path(output) / (name + '_latest.jpg')), view)
                if not arguments.headless and name in displayed:
                    view = displayed[name]
                    if not healthy:
                        view = view.copy()
                        cv2.putText(view, 'CAMERA STALE', (20, view.shape[0] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
                    cv2.imshow('Read-only SONIC observation: ' + name, view)
            recorder.session['camera_health'] = dict(camera_health)
            if not arguments.headless and cv2.waitKey(1) & 255 == ord('q'):
                break
            if not arguments.headless and any(cv2.getWindowProperty(
                    'Read-only SONIC observation: ' + name, cv2.WND_PROP_VISIBLE) < 1
                    for name in workers):
                break
            if now - last_flush >= 1.:
                recorder.flush()
                last_flush = now
            time.sleep(.015)
    except (Exception, KeyboardInterrupt) as exc:
        failure = f'{type(exc).__name__}: {exc}'
        recorder.finish(status='failed', error=failure)
        raise
    finally:
        shutdown = {name: worker.stop() for name, worker in workers.items()}
        recorder.session['camera_workers_stopped'] = shutdown
        if state is not None:
            state.close()
        if not arguments.headless:
            cv2.destroyAllWindows()
        if failure is None:
            recorder.finish()
        else:
            recorder.flush()
    counts = {name: dict(attempts=len(value['samples']),
                         board_quality_passed=value['quality_passed_count'])
              for name, value in recorder.session['cameras'].items()}
    print(f'Observation complete: {counts}; calibration_ready=False; control_commands_sent=0')


if __name__ == '__main__':
    main()
