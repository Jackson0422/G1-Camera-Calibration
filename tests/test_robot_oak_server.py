"""Robot OAK server lifecycle, with local fake processes instead of SSH."""
import socket
import sys
import time

import pytest

from g1_auto_calibration.hardware import robot_oak_server as module
from g1_auto_calibration.hardware.robot_oak_server import (
    RobotOakServer, endpoint_address, endpoint_listening, start_robot_server)


def config(port=5567):
    return dict(backend='network', endpoint=f'tcp://127.0.0.1:{port}',
                robot_server=dict(host='unitree@robot', remote_root='/tmp/oak_test', python='python'))


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def fake(monkeypatch, script):
    monkeypatch.setattr(RobotOakServer, 'sync', lambda self: None)
    monkeypatch.setattr(RobotOakServer, 'command', lambda self: [sys.executable, '-u', '-c', script])


def test_endpoint_parsing_and_listening():
    assert endpoint_address('tcp://192.168.123.164:5567') == ('192.168.123.164', 5567)
    with pytest.raises(ValueError):
        endpoint_address('ipc:///tmp/x')
    port = free_port()
    assert not endpoint_listening(f'tcp://127.0.0.1:{port}', .2)
    with socket.socket() as s:
        s.bind(('127.0.0.1', port))
        s.listen()
        assert endpoint_listening(f'tcp://127.0.0.1:{port}', .2)


def test_cameras_without_robot_server_are_untouched():
    assert start_robot_server(dict(backend='network', endpoint='tcp://127.0.0.1:1')) is None
    assert start_robot_server(dict(backend='depthai', robot_server={})) is None


def test_ready_server_runs_until_stop_closes_its_stdin(monkeypatch, tmp_path):
    marker = tmp_path/'exited'
    fake(monkeypatch, 'import sys; print("loading"); print("OAK ready: serial=X", flush=True); '
                      f'sys.stdin.read(); open({str(marker)!r}, "w").write("eof")')
    server = RobotOakServer(config(free_port()), log=lambda *a, **k: None).start()
    process = server.process
    time.sleep(.2)
    assert process.poll() is None and not server.reused
    server.stop()
    assert process.returncode == 0 and marker.read_text() == 'eof'
    assert server.process is None
    server.stop()  # idempotent


def test_existing_endpoint_is_reused_and_never_stopped(monkeypatch):
    monkeypatch.setattr(RobotOakServer, 'sync', lambda self: pytest.fail('must not copy'))
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        s.listen()
        server = RobotOakServer(config(s.getsockname()[1]), log=lambda *a, **k: None).start()
    assert server.reused and server.process is None
    server.stop()


def test_early_exit_reports_remote_output_and_busy_hint(monkeypatch):
    fake(monkeypatch, 'import sys; print("RuntimeError: No available devices"); sys.exit(1)')
    with pytest.raises(RuntimeError, match='exited before it was ready') as error:
        RobotOakServer(config(free_port()), log=lambda *a, **k: None).start()
    assert 'No available devices' in str(error.value) and 'someone else' in str(error.value)


def test_ready_timeout_stops_the_process(monkeypatch):
    fake(monkeypatch, 'import sys; print("booting"); sys.stdin.read()')
    server = RobotOakServer(config(free_port()), ready_timeout_s=.5, log=lambda *a, **k: None)
    with pytest.raises(TimeoutError, match='not ready'):
        server.start()
    assert server.process is None


def test_remote_command_copies_nothing_into_results_and_exits_on_eof():
    server = RobotOakServer(config(), log=lambda *a, **k: None)
    command = server.command()
    assert command[0] == 'ssh' and 'BatchMode=yes' in command and command[-2] == 'unitree@robot'
    assert command[-1].startswith('cd /tmp/oak_test && exec python -u -m g1_auto_calibration.scripts.serve_oak')
    assert '--exit-on-stdin-eof' in command[-1] and '--duration 0' in command[-1]
    assert module.PACKAGE.name == 'g1_auto_calibration'
