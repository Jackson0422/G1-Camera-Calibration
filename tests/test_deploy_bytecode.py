"""Pre-deploy cleanup removes only robot-written bytecode from a reused bundle."""
import subprocess

import pytest

from g1_auto_calibration.execution.sonic_observation import clear_deployed_bytecode

TEMPLATE = dict(robot_user='unitree', robot_ip='192.168.123.164', ssh_host_key_alias='192.168.0.130')
HASH = '986efbe5c031' + '0'*52


def run_locally(root):
    """Execute the remote shell script locally, with the robot home mapped to root."""
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        script = command[-1].replace('/home/unitree', str(root))
        return subprocess.run(['sh', '-c', script], **kwargs)
    return calls, run


def test_only_pycache_bytecode_is_removed(tmp_path):
    bundle = tmp_path/f'sonic_decoder_{HASH[:12]}'
    package = bundle/'src/sonic_robot_decoder'
    (package/'__pycache__').mkdir(parents=True)
    (package/'control.py').write_text('x = 1\n')
    (package/'__pycache__/control.cpython-310.pyc').write_bytes(b'pyc')
    (bundle/'model_decoder.onnx').write_bytes(b'model')
    (bundle/'stray.txt').write_text('left for deploy to reject')
    calls, run = run_locally(tmp_path)
    clear_deployed_bytecode(TEMPLATE, HASH, run=run)
    assert not (package/'__pycache__').exists()
    assert (package/'control.py').exists() and (bundle/'model_decoder.onnx').exists()
    assert (bundle/'stray.txt').exists()
    command = calls[0]
    assert command[:2] == ['ssh', '-T'] and 'BatchMode=yes' in command and 'StrictHostKeyChecking=yes' in command
    assert 'HostKeyAlias=192.168.0.130' in command and command[-2] == 'unitree@192.168.123.164'


def test_non_bytecode_in_pycache_is_kept_and_missing_bundle_is_fine(tmp_path):
    cache = tmp_path/f'sonic_decoder_{HASH[:12]}/src/sonic_robot_decoder/__pycache__'
    cache.mkdir(parents=True)
    (cache/'notes.txt').write_text('keep')
    _, run = run_locally(tmp_path)
    clear_deployed_bytecode(TEMPLATE, HASH, run=run)
    assert (cache/'notes.txt').exists()
    _, run = run_locally(tmp_path/'empty')
    clear_deployed_bytecode(TEMPLATE, HASH, run=run)  # first deployment: nothing to clean


def test_ssh_failure_is_reported():
    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 255, '', 'Permission denied')
    with pytest.raises(RuntimeError, match='Permission denied'):
        clear_deployed_bytecode(TEMPLATE, HASH, run=run)
