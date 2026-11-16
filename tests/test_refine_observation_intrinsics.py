import json

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from g1_auto_calibration.common import write_yaml
from g1_auto_calibration.scripts.refine_observation_intrinsics import fit


def test_multiview_refit_recovers_known_intrinsics_without_training_on_heldout_poses(tmp_path):
    trial = tmp_path/'trial'
    session = trial/'sessions'/'run'
    recording = session/'recording'
    recording.mkdir(parents=True)
    K = np.array([[700., 0., 322.], [0., 702., 238.], [0., 0., 1.]])
    guess = K.copy()
    guess[0, 0] *= .9
    guess[1, 1] *= .9
    D = np.r_[-.1, .025, .0002, -.0003, np.zeros(10)]
    obj = np.array([[x*.115, y*.115, 0.] for y in range(3) for x in range(8)], np.float32)
    rng = np.random.default_rng(4)
    frames, events, holds = [], [], []
    cv2.imwrite(str(recording/'image.png'), np.zeros((480, 640, 3), np.uint8))
    for i in range(20):
        rotation = Rotation.from_euler('xyz', rng.uniform(-.4, .4, 3)).as_rotvec()
        t = np.r_[rng.uniform(-.55, -.3), rng.uniform(-.25, -.1), rng.uniform(1.7, 2.3)]
        holds.append(dict(reference_start_frame=i*300+100, reference_end_frame=(i+1)*300))
        for delta in (60, 25):
            frame = (i+1)*300-delta
            points = cv2.projectPoints(obj, rotation, t, K, D)[0].reshape(-1, 2)
            points += rng.normal(0., .04, points.shape)
            sequence = len(frames)
            frames.append(dict(source_sequence=sequence, image_file='image.png',
                received_monotonic_s=100.+frame/50., K=guess.tolist(), distortion=D.tolist(), serial='test',
                board_observation=dict(object_points_m=obj.tolist(), image_points_px=points.tolist())))
    for frame in range(6001):
        events.append(dict(original_reference=True, playing=True, source_frame=frame,
                           local_monotonic_s=100.+frame/50.))
    (recording/'frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames))
    (session/'reference_timing.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    write_yaml(trial/'motion/prepared_reference.yaml', dict(holds=holds))
    write_yaml(recording/'recording.yaml', dict(camera='oak', complete=True))
    cv2.setNumThreads(1)
    report = fit(session, tmp_path/'refit.json')
    estimated = np.array(report['camera_matrix'])
    np.testing.assert_allclose(estimated[:2, :], K[:2, :], rtol=.02, atol=2.)
    assert report['validation']['held_out_rms_px'] < .3
    train, held = set(report['training_sequences']), set(report['held_out_sequences'])
    assert train.isdisjoint(held)
    assert len(held) == 8 and len(train) == 32
    assert not report['accepted_for_calibration']
