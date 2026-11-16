"""Read-only inventory; use --live to subscribe to actual streams for bounded time."""
import argparse
from datetime import datetime, timezone
import importlib
import platform
import time
import cv2
import numpy as np
from ..common import ROOT, load_yaml, write_yaml
from ..hardware.base import open_camera
from ..hardware.unitree_state import UnitreeState
from ..calibration.forward_kinematics import ForwardKinematics


def model_from_config(path):
    c = load_yaml(path)
    from pathlib import Path
    urdf = Path(c['urdf'])
    if not urdf.is_absolute():
        urdf = Path(path).parent/urdf
    return c, ForwardKinematics(urdf, c['base_frame'], c['mount_frame'], c['joint_indices'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', default=str(ROOT/'config/robot.yaml'))
    p.add_argument('--cameras', default=str(ROOT/'config/cameras.yaml'))
    p.add_argument('--live', action='store_true')
    p.add_argument('--duration', type=float, default=5)
    p.add_argument('--camera', choices=['d435', 'oak', 'all', 'none'], default='all')
    p.add_argument('--output', default=str(ROOT/'results/hardware.yaml'))
    a = p.parse_args()
    if not 0 < a.duration <= 120:
        p.error('Duration must be in (0, 120] seconds')
    c, fk = model_from_config(a.robot)
    report = dict(time_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  control_commands_sent=0, live_requested=a.live, modules={},
                  urdf=dict(path=str(fk.path), sha256=fk.sha256, base=fk.base, mount=fk.mount, chain=fk.chain_names),
                  opencv_handeye_available=hasattr(cv2, 'calibrateHandEye'))
    for name in ['numpy', 'scipy', 'yaml', 'depthai', 'zmq', 'msgpack', 'rclpy']:
        try:
            module = importlib.import_module(name)
            report['modules'][name] = dict(available=True, version=getattr(module, '__version__', 'unknown'))
        except Exception as exc:
            report['modules'][name] = dict(available=False, error=str(exc))
    if a.live:
        state = None
        try:
            state = UnitreeState(c)
            time.sleep(min(a.duration, 5))
            with state.lock:
                samples = list(state.states)
            report['state'] = dict(messages=len(samples), duplicate_ticks_discarded=state.duplicate_ticks,
                                   error=state.error, interface=c['interface'], domain=c['domain'])
            if samples:
                s = samples[-1]
                report['state'].update(mode_pr=s.mode_pr, mode_machine=s.mode_machine, tick=s.tick, q=s.q, dq=s.dq,
                    age_s=time.monotonic()-s.received_s, T_base_mount=fk.compute(s.q),
                    rate_hz=(len(samples)-1)/max(samples[-1].received_s-samples[0].received_s, 1e-9))
        except Exception as exc:
            report['state'] = dict(error=f'{type(exc).__name__}: {exc}')
        finally:
            if state:
                state.close()
        report['cameras'] = {}
        configs = load_yaml(a.cameras)
        for name in ([] if a.camera == 'none' else list(configs) if a.camera == 'all' else [a.camera]):
            reader, frames, last = None, 0, None
            try:
                reader = open_camera(configs[name])
                start = time.monotonic()
                while time.monotonic()-start < a.duration:
                    last = reader.read(timeout_s=min(3, a.duration))
                    frames += 1
                from pathlib import Path
                image_path = Path(a.output).with_name(name+'_latest.jpg')
                image_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_path), last.bgr)
                report['cameras'][name] = dict(frames=frames, rate_hz=frames/(time.monotonic()-start),
                    serial=last.serial, resolution=[last.bgr.shape[1], last.bgr.shape[0]], optical_frame=last.optical_frame,
                    K=last.K, distortion=last.distortion, source_timestamp_s=last.source_timestamp_s,
                    source_clock=last.source_clock, received_monotonic_s=last.received_monotonic_s,
                    sequence=last.sequence, saved_image=str(image_path), synchronized_for_calibration=False)
            except Exception as exc:
                report['cameras'][name] = dict(frames=frames, error=f'{type(exc).__name__}: {exc}')
            finally:
                if reader:
                    reader.close()
    write_yaml(a.output, report)
    print(f'Hardware report written to {a.output}. Robot control commands sent: 0.')


if __name__ == '__main__':
    main()
