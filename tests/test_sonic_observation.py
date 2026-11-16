import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from g1_auto_calibration.execution import sonic_observation as trial
from g1_auto_calibration.verification import sonic_stationary_motion as motion


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import socket
    monkeypatch.setattr(socket, 'socket', lambda *a, **k: pytest.fail('Preparation opened a socket'))
    sonic = tmp_path/'sonic'
    profile = sonic/'robot_decoder/profiles/g1_29_wave_control.json'
    profile.parent.mkdir(parents=True)
    values = dict(default_angles_hw=[0.]*29, lower_hw=[-2.]*29, upper_hw=[2.]*29,
        velocity_hw=[2.]*29, joint_names_hw=[f'joint_{i}' for i in range(29)],
        mujoco_to_isaaclab=np.roll(np.arange(29), 7).tolist(), control_contract='original',
        kp_hw=[30.]*29, motion_source_sha256='OLD')
    profile.write_text(json.dumps(values))
    monkeypatch.setattr(trial, 'ROOT', tmp_path/'calibration')
    monkeypatch.setattr(motion, 'ROOT', tmp_path/'calibration')
    helper = SimpleNamespace(ENCODER='encoder', DECODER='decoder', FIXTURE='fixture',
        validate_encoder_source=lambda *a, **k: 'encoder_hash')
    deploy = SimpleNamespace(build_bundle=lambda *a: SimpleNamespace(
        archive=b'test bundle', bundle_hash='bundle_hash', manifest={'files': {}}))
    monkeypatch.setattr(trial, 'runtime', lambda *a: (helper, None, deploy))
    output = tmp_path/'trial'
    report = trial.prepare_trial(output, sonic_root=sonic)
    return output, report, values, helper


def test_source_and_encoder_reference_match_exactly_and_gains_are_preserved(prepared):
    output, report, original, _ = prepared
    checked = trial.validate_trial(output)
    # Default motion: the operator's 40 calibration poses, 2 s holds, about 3.2 minutes.
    assert checked['pose_count'] == 40 and 150 < checked['duration_s'] < 240
    assert checked['amplitude_deg'] == dict(yaw=20., roll=10., pitch=10.)
    profile = json.loads(Path(report['profile']).read_text())
    for key in ('kp_hw', 'control_contract', 'default_angles_hw', 'mujoco_to_isaaclab'):
        assert profile[key] == original[key]
    assert profile['motion_source_sha256'] != 'OLD'
    assert profile['motion_source_sha256'] == trial.digest(report['source'])
    assert checked['automatic_calibration'] is False
    assert checked['hardware_started'] is False
    assert checked['robot_control_commands_sent'] == 0


def test_standby_uses_existing_startup_contract_and_preserves_output_limits(prepared):
    output, report, original, _ = prepared
    second = output.parent/'standby_trial'
    standby = trial.prepare_trial(second, sonic_root=report['sonic_root'], standby=True)
    profile = json.loads(Path(standby['profile']).read_text())
    assert profile['standby_session'] is True
    assert profile['startup_completion'] == 'reference_ramp_complete'
    assert profile.get('temperature_limit_enabled') is not False
    for key in ('kp_hw', 'control_contract', 'default_angles_hw', 'lower_hw', 'upper_hw', 'velocity_hw'):
        assert profile[key] == original[key]


@pytest.mark.parametrize('target', ['source', 'profile'])
def test_changed_preparation_is_rejected_before_any_connection(prepared, target):
    output, report, _, _ = prepared
    Path(report[target]).write_text('changed')
    with pytest.raises(ValueError, match='file changed'):
        trial.validate_trial(output)


def test_preview_only_never_opens_a_capture_or_requests_approval(prepared, monkeypatch):
    output, report, _, helper = prepared
    calls = []
    preview = SimpleNamespace(run_preview=lambda *a, **k: calls.append((a, k)) or {'completed': True})
    monkeypatch.setattr(trial, 'runtime', lambda *a: (helper, preview, None))
    monkeypatch.setattr(trial.CaptureSession, 'open', lambda *a: pytest.fail('Preview opened capture'))
    result = trial.run_trial(output)
    assert not result['hardware_started']
    assert len(calls) == 1 and calls[0][1]['headless'] is False
    assert calls[0][1]['profile'] == report['profile']
    reference, preview_output = map(Path, calls[0][0][:2])
    assert reference.parent.parent/'preview' == preview_output
    assert (reference.parent.parent/'sources'/Path(report['source']).name).is_file()
    assert trial.digest(reference/'joint_pos.csv') == trial.digest(Path(report['reference'])/'joint_pos.csv')


def test_recording_starts_on_first_original_playing_token_and_return_preserves_encoder():
    events, calls = [], []

    class Encoder:
        def encode(self, frame, quaternion, *, playing):
            calls.append((frame, quaternion, playing))
            return np.ones(64)

        def retarget_reference(self, reference):
            calls.append(('return', reference))
            return self

    recording = SimpleNamespace(started=None)

    def begin():
        assert recording.started is None
        recording.started = 10.

    recording.begin = begin
    capture = SimpleNamespace(recording=recording, check=lambda: None)
    encoder = trial.RecordingEncoder(Encoder(), capture, events, clock=lambda: 10.)
    encoder.encode(0, [1, 0, 0, 0], playing=False)
    assert recording.started is None
    result = encoder.encode(3, [1, 0, 0, 0], playing=True)
    assert recording.started == 10. and result.shape == (64,)
    encoder.encode(4, [1, 0, 0, 0], playing=True)
    returned = encoder.retarget_reference('return_ref')
    returned.encode(0, [1, 0, 0, 0], playing=True)
    assert events[-1]['original_reference'] is False
    assert all(e['measured_joint_pose'] is False for e in events)
    assert ('return', 'return_ref') in calls


def test_capture_failure_prevents_sending_the_next_encoded_token():
    capture = SimpleNamespace(check=lambda: (_ for _ in ()).throw(RuntimeError('stale state')))
    encoder = trial.RecordingEncoder(None, capture, [])
    with pytest.raises(RuntimeError, match='stale state'):
        encoder.encode(0, [1, 0, 0, 0])
