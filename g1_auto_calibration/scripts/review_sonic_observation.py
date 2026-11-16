"""Hash-check a closed raw SONIC archive and plot measured joints offline."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from ..common import digest, load_yaml, write_yaml
from ..execution.timing_trace import load_trace
from ..observation.continuous_video import render_review_video


def review(session):
    session = Path(session).resolve()
    directory = session/'recording'
    archive = load_yaml(directory/'recording.yaml')
    if not archive['complete'] or not archive['requested_window_complete']:
        raise ValueError('Observation recording is incomplete')
    if digest(directory/archive['frame_index']) != archive['frame_index_sha256']:
        raise ValueError('Raw frame index hash changed')
    frames = [json.loads(line) for line in (directory/archive['frame_index']).read_text().splitlines()]
    if len(frames) != archive['frame_count']:
        raise ValueError('Raw frame count differs from the closed manifest')
    for i, frame in enumerate(frames):
        image = (directory/frame['image_file']).resolve()
        image.relative_to(directory)
        if frame['video_frame_index'] != i or digest(image) != frame['image_sha256']:
            raise ValueError('Raw image/index identity changed')
    video = archive['video']
    if not video:
        # Trials close without a video so the program exits promptly after U.
        print(f'Rendering review.mp4 from {len(frames)} raw frames ...', flush=True)
        video = render_review_video(directory, len(frames), archive['elapsed_recording_s'])
    elif digest(directory/video['file']) != video['sha256']:
        raise ValueError('Review video hash changed')
    count = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=nb_read_frames', '-of', 'json',
        str(directory/video['file'])], check=True, capture_output=True, text=True, timeout=120.)
    if int(json.loads(count.stdout)['streams'][0]['nb_read_frames']) != len(frames):
        raise ValueError('Video does not contain exactly one frame per raw index')
    _, state, _ = load_trace(directory/'timing_trace', archive['camera'])
    t0, duration = archive['record_start_received_s'], archive['elapsed_recording_s']
    inside = (state['received_s'] >= t0) & (state['received_s'] < t0+duration)
    t, q = state['received_s'][inside]-t0, state['q'][inside]
    if not len(t):
        raise ValueError('No measured joints in the actual recording interval')
    motion = json.loads((session/'hardware/report.json').read_text())
    visible = [f for f in frames if f['board_observation'] is not None]
    result = dict(kind='reviewed_sonic_raw_observation', session=str(session),
        motion_completed=motion['motion_completed'], controller_run_passed=motion['passed'],
        motor_owner_shutdown_confirmed=motion['motor_owner_expected_shutdown_confirmed'],
        camera=archive['camera'], duration_s=duration, frame_count=len(frames),
        board_detected_frames=len(visible), indexed_image_hashes_verified=True,
        video_frame_count_verified=True, review_video=video, full_state_trace_count=len(state['q']),
        state_count_in_recording_window=len(q),
        measured_state_maximum_gap_s=float(np.max(np.diff(t))),
        measured_waist_min_deg=np.rad2deg(q[:, 12:15].min(axis=0)).tolist(),
        measured_waist_max_deg=np.rad2deg(q[:, 12:15].max(axis=0)).tolist(),
        measured_leg_joint_range_deg=np.rad2deg(np.ptp(q[:, :12], axis=0)).tolist(),
        board_max_reprojection_rms_px=max((f['board_observation']['reprojection_rms_px'] for f in visible), default=None),
        camera_source_clock=sorted({f['source_clock'] for f in frames}),
        capture_time_joint_alignment_verified=False, world_foot_trajectory_measured=False,
        calibration_ready=False, automatic_calibration_performed=False,
        note='Plots use PC receipt/reference times. Joint motion does not measure world foot slip.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    events = [json.loads(line) for line in (session/'reference_timing.jsonl').read_text().splitlines()]
    events = [e for e in events if e['original_reference'] and e['playing']]
    source = np.loadtxt(next((session/'sources').glob('*.csv')), delimiter=',', skiprows=1)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    stride = max(1, len(t)//6000)
    for j, (name, color) in enumerate(zip(('yaw', 'roll', 'pitch'), ('tab:blue', 'tab:orange', 'tab:green'))):
        axes[0].plot(t[::stride], np.rad2deg(q[::stride, 12+j]), color=color, label=f'measured {name}')
        if events:
            axes[0].plot([e['local_monotonic_s']-t0 for e in events],
                np.rad2deg(source[[e['source_frame'] for e in events], 7+12+j]),
                color=color, linestyle='--', alpha=.45)
    axes[0].set(ylabel='Waist (deg)', title='Real SONIC observation | solid: measured joints; dashed: reference')
    axes[0].legend(loc='upper right', ncol=3)
    for j in (3, 4, 9, 10):
        axes[1].plot(t[::stride], np.rad2deg(q[::stride, j]-q[0, j]), label=f'joint {j}')
    axes[1].set(ylabel='Leg angle change (deg)', title='Measured knees/ankles relative to recording start; not world foot displacement')
    axes[1].legend(ncol=4)
    axes[2].scatter([f['received_monotonic_s']-t0 for f in visible],
        [f['board_observation']['reprojection_rms_px'] for f in visible], s=3)
    axes[2].set(xlabel='PC monotonic receipt/reference time since recording start (s)',
                ylabel='Board PnP RMS (px)', title='Board detection quality; exposure-to-joint alignment remains pending')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_xlim(0, duration)
    fig.savefig(session/'joint_board_review.png', dpi=140)
    plt.close(fig)
    result['plot'] = 'joint_board_review.png'
    result['plot_sha256'] = digest(session/result['plot'])
    write_yaml(session/'review_summary.yaml', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    args = p.parse_args()
    report = review(args.session)
    print(f'Checked {report["frame_count"]} images and video frames, {report["full_state_trace_count"]} raw states. '
          f'Board visible in {report["board_detected_frames"]} frames. Calibration not performed.')


if __name__ == '__main__':
    main()
