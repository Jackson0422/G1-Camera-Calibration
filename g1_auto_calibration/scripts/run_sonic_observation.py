"""Prepare a SONIC waist-pose calibration trial (default: the 40 poses in config/calibration_poses_40.yaml)."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from ..common import ROOT
from ..execution.sonic_observation import DEFAULT_POSES, prepare_trial, run_trial
from ..verification.sonic_policy import DEFAULT_SONIC_ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', type=Path, help='Existing prepared trial directory; hashes are checked')
    p.add_argument('--output', type=Path, help='New trial directory (prepare only by default)')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--preview', action='store_true', help='Fresh offline onscreen dynamics preview')
    mode.add_argument('--hardware-after-preview', action='store_true', help='Fresh preview, onsite yes, then real observation trial')
    p.add_argument('--camera', choices=('d435', 'oak'), default='d435')
    p.add_argument('--standby', action='store_true', help='Prepare balanced standby; automatically play once fresh stability checks pass, U to stop')
    p.add_argument('--local-ip', help='Actual workstation wired IPv4 address')
    p.add_argument('--poses', type=Path, default=DEFAULT_POSES,
                   help='YAML list of waist [yaw, roll, pitch] poses in degrees (default: the 40 calibration poses)')
    p.add_argument('--no-solve', action='store_true',
                   help='Do not solve the camera extrinsics automatically after a hardware trial')
    p.add_argument('--no-camera-view', action='store_true',
                   help='Skip the live camera/board check window before a hardware trial')
    p.add_argument('--deployment-template', type=Path,
        default=DEFAULT_SONIC_ROOT/'robot_decoder/deployments/walking_test_wired.json')
    a = p.parse_args()
    if a.trial and a.output:
        p.error('Use either --trial or --output')
    if a.hardware_after_preview and not a.local_ip:
        p.error('--hardware-after-preview requires --local-ip')
    output = a.trial or a.output or ROOT/'results'/('sonic_observation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    if not a.trial:
        prepare_trial(output, standby=a.standby, poses=a.poses)
    print(f'Prepared observation-only trial: {output}', flush=True)
    if a.preview or a.hardware_after_preview:
        result = run_trial(output, camera=a.camera, hardware=a.hardware_after_preview,
                           local_ip=a.local_ip, deployment_template=a.deployment_template,
                           camera_view=not a.no_camera_view)
        print(f'Session: {result["session"]}', flush=True)
        if a.hardware_after_preview and not a.no_solve:
            solve_and_print(Path(result['session']), a.camera, result.get('recording'))


CAMERA_NAMES = {'oak': ('OAK', 'OAK_head'), 'd435': ('D435i', 'D435i')}


def solve_and_print(session, camera, recording):
    """Solve with the intrinsics recorded in each frame (camera defaults) and print the result."""
    if not recording or recording.get('complete') is not True:
        print('Recording is incomplete (U may have been pressed before the motion ended); not solving.', flush=True)
        return None
    from ..scripts.estimate_sonic_camera import estimate
    name, prefix = CAMERA_NAMES[camera]
    output = ROOT/'results'/f'{prefix.lower()}_calibration_{session.name}'
    print('Solving the extrinsics with the camera default intrinsics (about half a minute) ...', flush=True)
    try:
        estimate(session, output)
    except Exception as exc:
        print(f'Automatic solve failed: {type(exc).__name__}: {exc}', flush=True)
        print('Run it manually: bash run.sh -m g1_auto_calibration.scripts.estimate_sonic_camera '
              f'--session {session} --output <new directory>', flush=True)
        return None
    # The JSON/diagram export needs matplotlib, which lives in the project (run.sh) environment.
    export = subprocess.run(['bash', str(ROOT.parent/'run.sh'), '-B', '-m', 'g1_auto_calibration.scripts.export_d435_estimate',
                             '--result-directory', str(output), '--camera-name', name, '--file-prefix', prefix],
                            capture_output=True, text=True, timeout=300)
    if export.returncode:
        print(f'JSON/diagram export failed: {export.stderr.strip().splitlines()[-1:]}', flush=True)
    else:
        print(export.stdout.strip(), flush=True)
    return output


if __name__ == '__main__':
    main()
