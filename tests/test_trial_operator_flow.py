"""Camera check window, preview y/n prompt, and reuse of an identical earlier preview."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from g1_auto_calibration.execution.sonic_observation import ask_preview, reuse_preview
from g1_auto_calibration.observation.camera_view import show_camera_view


class FakeReader:
    def __init__(self):
        self.closed = False
        self.sequence = 0

    def read(self):
        self.sequence += 1
        return SimpleNamespace(bgr=np.zeros((1080, 1920, 3), np.uint8), optical_frame='oak',
                               serial='X', sequence=self.sequence)

    def close(self):
        self.closed = True


class FakeDetector:
    def __init__(self, fail=False):
        self.fail = fail

    def estimate(self, frame):
        if self.fail:
            raise ValueError('Too few reliable known markers')
        return dict(image_points_px=np.array([[100., 100.], [200., 120.]]), T_camera_board=np.eye(4),
                    reprojection_rms_px=.5, geometric_gates_passed=True, quality_rejection_reason=None)


@pytest.mark.parametrize('keys, fail, expected', [
    ([255, 255, 13], False, True), ([32], True, True), ([255, ord('q')], False, False), ([27], True, False)])
def test_camera_view_continue_or_abort_and_always_closes(keys, fail, expected):
    reader, shown = FakeReader(), []
    pressed = iter(keys)
    result = show_camera_view('oak', {}, {}, opener=lambda config: reader, detector=FakeDetector(fail),
                              imshow=lambda name, image: shown.append(image.shape),
                              wait_key=lambda delay: next(pressed, 255), destroy=lambda name: None,
                              visible=lambda name: True)
    assert result is expected and reader.closed
    assert shown and shown[0] == (720, 1280, 3)


def test_closing_the_window_continues_but_not_before_it_appeared():
    # Not yet mapped for two frames, then visible, then closed by the operator.
    states = iter([False, False, True, True, False])
    reader = FakeReader()
    result = show_camera_view('oak', {}, {}, opener=lambda config: reader, detector=FakeDetector(),
                              imshow=lambda name, image: None, wait_key=lambda delay: 255,
                              destroy=lambda name: None, visible=lambda name: next(states))
    assert result is True and reader.closed and reader.sequence == 5


def test_ask_preview_accepts_only_y_or_n(capsys):
    answers = iter(['', 'yes', 'N'])
    assert ask_preview(lambda prompt: next(answers)) is False
    assert capsys.readouterr().out.count('y/Y') == 2
    assert ask_preview(lambda prompt: 'Y') is True


def make_run(root, stamp, files, *, report_fields=None):
    """Create results/sonic_observation_<stamp>/sessions/<stamp>/ with an eligible preview report."""
    trial = root/f'sonic_observation_{stamp}'
    session = trial/'sessions'/stamp
    (session/'references/ref').mkdir(parents=True)
    (trial/'profile.json').write_text('profile')
    for name, text in files.items():
        (session/'references/ref'/name).write_text(text)
    digests = {str(trial/'profile.json'): 'p', '/upstream/decoder.onnx': 'd',
               **{str(session/'references/ref'/n): t for n, t in files.items()}}
    report = {**dict(eligible=True, standby_mode=True, pipeline_digests=digests), **(report_fields or {})}
    (session/'preview').mkdir()
    (session/'preview/report.json').write_text(json.dumps(report))
    return trial, session, digests


def fakes(digests_for):
    preview = SimpleNamespace(pipeline_digests=lambda reference, profile, encoder, decoder: digests_for(reference),
                              preview_is_eligible=lambda report: report.get('eligible') is True)
    def validate_standby_preview(report):
        if report.get('free_standing_too_short'):
            raise ValueError('standby preview too short')
    return SimpleNamespace(ENCODER='e', DECODER='d', validate_standby_preview=validate_standby_preview), preview


def current_run(root):
    trial = root/'sonic_observation_new'
    session = trial/'sessions/new'
    (session/'references/ref').mkdir(parents=True)
    return trial, session


def digests_like(trial, session, files):
    return {str(trial/'profile.json'): 'p', '/upstream/decoder.onnx': 'd',
            **{str(session/'references/ref'/n): t for n, t in files.items()}}


def test_identical_earlier_preview_is_reused_and_rebound_to_current_files(tmp_path):
    files = {'joint_pos.csv': 'a', 'metadata.txt': 'b'}
    _, old_session, _ = make_run(tmp_path, '20261116T080000', files)
    trial, session = current_run(tmp_path)
    helper, preview = fakes(lambda reference: digests_like(trial, session, files))
    report = reuse_preview(session, trial, dict(reference=str(session/'references/ref'), profile='p'),
                           helper, preview, standby=True, search_root=tmp_path)
    assert report['reused_preview_report'] == str((old_session/'preview/report.json').resolve())
    assert report['pipeline_digests'] == digests_like(trial, session, files)
    assert json.loads((session/'preview/report.json').read_text()) == report


@pytest.mark.parametrize('change', ['content', 'upstream', 'ineligible', 'mode', 'standby_check'])
def test_any_difference_prevents_reuse(tmp_path, change):
    files = {'joint_pos.csv': 'a'}
    fields = {'ineligible': dict(eligible=False), 'mode': dict(standby_mode=False),
              'standby_check': dict(free_standing_too_short=True)}.get(change)
    make_run(tmp_path, '20261116T080000', files, report_fields=fields)
    trial, session = current_run(tmp_path)
    def current(reference):
        digests = digests_like(trial, session, {'joint_pos.csv': 'CHANGED' if change == 'content' else 'a'})
        if change == 'upstream':
            digests['/upstream/decoder.onnx'] = 'new model'
        return digests
    helper, preview = fakes(current)
    assert reuse_preview(session, trial, dict(reference='r', profile='p'), helper, preview,
                         standby=True, search_root=tmp_path) is None
    assert not (session/'preview').exists()


def test_console_summary_reports_position_tilt_and_validation():
    from g1_auto_calibration.scripts.estimate_sonic_camera import summary
    from g1_auto_calibration.scripts.run_sonic_observation import solve_and_print
    down = np.radians(25.)
    # Optical z (viewing direction) points forward and 25 degrees down in torso axes.
    R = np.array([[0., -np.sin(down), np.cos(down)], [-1., 0., 0.], [0., -np.cos(down), -np.sin(down)]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [.07, .003, .45]
    result = dict(camera='oak', parent_frame='torso_link',
                  T_torso_camera_optical=dict(matrix=T.tolist(), quaternion_xyzw=[0., 0., 0., 1.]),
                  held_out_validation=dict(translation_rms_mm=3.2, rotation_rms_deg=.31, passed=True,
                                           thresholds=dict(translation_rms_mm=5, rotation_rms_deg=.5)),
                  sampling_stable_count=20, sample_count=20, robust_solver_succeeded=True)
    text = summary(result, '/out')
    assert 'forward 7.0 cm, left 0.3 cm, up 45.0 cm' in text and 'Looking down: 25.0 deg' in text
    assert 'translation RMS 3.2 mm (limit 5)' in text and ': PASSED' in text and 'Result directory: /out' in text
    assert solve_and_print(Path('/nonexistent'), 'oak', dict(complete=False)) is None
