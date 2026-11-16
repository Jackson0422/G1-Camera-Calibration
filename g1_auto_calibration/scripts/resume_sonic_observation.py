"""Resume a deployment-only failure using its still-fresh exact preview approval."""

import argparse
import json
from pathlib import Path

from ..common import load_yaml, write_yaml, digest
from ..execution.sonic_observation import (validate_trial, runtime, CaptureSession, RecordingEncoder,
                                           clear_deployed_bytecode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--local-ip', required=True)
    a = p.parse_args()
    trial = validate_trial(a.trial)
    session = a.session.resolve()
    session.relative_to(a.trial.resolve()/'sessions')
    prior = load_yaml(session/'observation_trial.yaml')
    if prior.get('motion') is not None or (session/'hardware').exists() or 'SSH deployment failed' not in prior.get('failure', ''):
        raise ValueError('Resume is only allowed after a deployment failure before any motor session')
    helper, _, deploy = runtime(trial['sonic_root'])
    reference = session/'references'/Path(trial['reference']).name
    source = session/'sources'/Path(trial['source']).name
    approval = json.loads((session/'approval.json').read_text())
    helper.validate_approval(approval, reference, trial['profile'], helper.ENCODER, helper.DECODER)
    from sonic_robot_decoder.encoder import Mode0Reference, WaveEncoder
    ref = Mode0Reference.from_directory(reference)
    expected = helper.validate_encoder_source(trial['profile'], helper.ENCODER, hardware=True, source=source)
    preloaded = WaveEncoder(helper.ENCODER, ref, [1., 0., 0., 0.], expected_sha256=expected)
    for playing in (False, True, False, True):
        preloaded.encode(0, [1., 0., 0., 0.], playing=playing)
    for name in ('recording', 'observation_trial.yaml', 'reference_timing.jsonl'):
        original = session/name
        if original.exists():
            original.rename(session/(name+'_predeployment_failure'))
    capture = CaptureSession(session/'recording', camera=prior['camera'], duration_s=float(trial['duration_s']))
    events, motion, archive = [], None, None
    failure = None
    try:
        capture.open()
        template = json.loads((Path(trial['sonic_root'])/'robot_decoder/deployments/walking_test_wired.json').read_text())
        bundle = deploy.build_bundle(helper.DECODER, trial['profile'], helper.FIXTURE)
        if bundle.bundle_hash != trial['bundle_hash']:
            raise ValueError('Prepared robot-local bundle changed')
        clear_deployed_bytecode(template, bundle.bundle_hash)
        deployed, _ = deploy.deploy(bundle, template['robot_ip'], template['robot_user'],
            output_config=session/'deployment.json', robot_python=template['robot_python'],
            sdk_path=template['sdk_path'], ssh_host_key_alias=template.get('ssh_host_key_alias'))

        def factory(model, reference, quaternion, *, expected_sha256):
            if expected_sha256 != preloaded.model_sha256:
                raise ValueError('Encoder model mismatch')
            return RecordingEncoder(preloaded.with_initial_heading(quaternion), capture, events)

        motion = helper.run_remote_wave(deployed, reference=reference, output=session/'hardware',
            local_ip=a.local_ip, hardware=True, approval=approval, profile=trial['profile'],
            encoder_model=helper.ENCODER, decoder_model=helper.DECODER, duration=float(trial['duration_s'])+60.,
            encoder_factory=factory, motion_source=source)
        failure = motion.get('failure')
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            archive = capture.close()
        finally:
            timing = session/'reference_timing.jsonl'
            timing.write_text(''.join(json.dumps(e)+'\n' for e in events))
            write_yaml(session/'observation_trial.yaml', dict(kind='sonic_observation_trial_result',
                camera=prior['camera'], motion=motion, recording=archive, failure=failure,
                resumed_deployment_only_failure=True, automatic_calibration=False, calibration_ready=False,
                capture_time_joint_alignment_verified=False, world_foot_motion_compensation_available=False,
                reference_timing_file=timing.name, reference_timing_sha256=digest(timing)))
    print(f'Completed session: {session}', flush=True)


if __name__ == '__main__':
    main()
