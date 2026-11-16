import json
import shutil

import numpy as np
import pytest

from g1_auto_calibration.hardware.base import Frame
from g1_auto_calibration.hardware.unitree_state import State, StateBuffer
from g1_auto_calibration.observation.dual_capture import CameraPacket
from g1_auto_calibration.observation.continuous_video import ContinuousVideo
from g1_auto_calibration.execution.timing_trace import load_trace


def packet(sequence, received, *, board=True):
    frame = Frame(np.full((48, 64, 3), sequence, np.uint8), 1000.+received, 'camera_clock',
        received, 'test_optical', 'serial', np.array([[60., 0., 32.], [0., 60., 24.], [0., 0., 1.]]),
        np.zeros(5), sequence)
    obs = dict(T_camera_board=np.eye(4), reprojection_rms_px=.2,
        object_points_m=[[0., 0., 0.]], image_points_px=[[12., 13.]], marker_ids=[1]) if board else None
    return CameraPacket(frame, obs, None if board else 'board not visible')


def buffer():
    b = StateBuffer()
    for i in range(101):
        b.append(State(.5+i*.01, i, np.full(29, i*.001), np.full(29, .1),
                       0, np.zeros(3), 5, 0))
    return b


def test_every_video_frame_preserves_board_identity_raw_pixels_and_actual29(tmp_path, monkeypatch):
    import socket
    monkeypatch.setattr(socket, 'socket', lambda *a, **k: pytest.fail('Capture archive opened a socket'))
    now = [1.]
    r = ContinuousVideo(tmp_path/'run', buffer(), duration_s=.3, clock=lambda: now[0]).start()
    r.record(packet(8, 1.01))  # Startup packet before explicit recording.
    r.begin()
    r.record(packet(10, 1.05))
    r.record(packet(10, 1.05))  # Duplicate callback is not a new video frame.
    r.record(packet(11, 1.15, board=False))
    r.record(packet(12, 1.3))  # Outside the requested half-open interval.
    now[0] = 1.3
    report = r.close(render_video=False)
    assert report['frame_count'] == 2 and report['board_detected_frames'] == 1
    rows = [json.loads(line) for line in (r.output/'frames.jsonl').read_text().splitlines()]
    assert [row['video_frame_index'] for row in rows] == [0, 1]
    assert [row['source_sequence'] for row in rows] == [10, 11]
    assert rows[1]['board_rejection_reason'] == 'board not visible'
    assert rows[0]['board_observation']['image_points_px'] == [[12., 13.]]
    for row in rows:
        assert len(row['receive_time_joint_brackets']) == 2
        assert all(len(s['q_rad']) == 29 for s in row['receive_time_joint_brackets'])
        assert row['capture_time_joint_alignment_verified'] is False
        assert row['calibration_accepted'] is False
    _, states, frames = load_trace(r.output/'timing_trace', 'd435')
    assert states['q'].shape[1] == 29
    assert [f['sequence'] for f in frames] == [10, 11]
    assert report['robot_control_commands_sent'] == 0
    assert report['calibration_ready'] is False


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='Optional video encoder absent')
def test_encoded_video_has_exactly_one_frame_per_raw_index(tmp_path):
    now = [1.]
    r = ContinuousVideo(tmp_path/'run', buffer(), duration_s=.3, clock=lambda: now[0]).start()
    r.begin()
    for i, t in enumerate([1.05, 1.15, 1.25]):
        r.record(packet(i, t))
    now[0] = 1.3
    report = r.close()
    assert report['complete'], report
    assert report['video']['frame_count'] == 3
    assert report['video']['presentation_times_are_capture_times'] is False
    assert report['capture_time_joint_alignment_verified'] is False


def test_early_stop_keeps_raw_frames_without_claiming_complete_window(tmp_path):
    now = [1.]
    r = ContinuousVideo(tmp_path/'run', buffer(), duration_s=120., clock=lambda: now[0]).start()
    r.begin()
    r.record(packet(1, 1.05))
    now[0] = 1.2
    report = r.close(render_video=False)
    assert report['frame_count'] == 1
    assert report['elapsed_recording_s'] == pytest.approx(.2)
    assert not report['requested_window_complete']
    assert not report['complete']
