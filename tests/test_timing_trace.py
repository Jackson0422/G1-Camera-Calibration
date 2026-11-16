"""Continuous timing logging owns files and injected state only."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from g1_auto_calibration.execution.timing_trace import TimingTrace, load_trace
from g1_auto_calibration.hardware.unitree_state import State, StateBuffer


def sample(index):
    return State(10. + index * .001, index, np.full(29, index * 1e-6),
                 np.zeros(29), 0, np.zeros(3), 5)


def packet(sequence):
    return SimpleNamespace(frame=SimpleNamespace(source_timestamp_s=100. + sequence * .033,
        source_clock='SYNTHETIC', received_monotonic_s=10. + sequence * .033,
        sequence=sequence, serial='SYNTHETIC_CAMERA', optical_frame='synthetic_optical'),
        observation=dict(T_camera_board=np.eye(4), reprojection_rms_px=.1), rejection_reason=None)


def test_trace_keeps_original_values_and_verifies_every_file(tmp_path):
    buffer = StateBuffer()
    for index in range(100):
        buffer.append(sample(index))
    trace = TimingTrace(tmp_path / 'trace', buffer).start()
    trace.record_camera('d435', packet(1))
    for index in range(100, 200):
        buffer.append(sample(index))
    trace.record_camera('d435', packet(2))
    report = trace.close()
    assert report['complete']
    assert report['state_count'] == 200
    assert report['robot_control_commands_sent'] == 0
    _, states, frames = load_trace(tmp_path / 'trace', 'd435')
    np.testing.assert_array_equal(states['q'][-1], sample(199).q)
    assert states['tick'].tolist() == list(range(200))
    assert [f['sequence'] for f in frames] == [1, 2]
    (tmp_path / 'trace' / 'd435_frames.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='hash changed'):
        load_trace(tmp_path / 'trace', 'd435')


def test_incomplete_trace_cannot_be_used_for_automatic_alignment(tmp_path):
    buffer = StateBuffer()
    trace = TimingTrace(tmp_path / 'trace', buffer)
    trace.error = 'Injected disk write failure'
    report = trace.close()
    assert not report['complete']
    with pytest.raises(ValueError, match='incomplete'):
        load_trace(tmp_path / 'trace', 'd435')
