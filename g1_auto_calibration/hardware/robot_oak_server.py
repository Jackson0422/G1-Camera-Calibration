"""Run the robot-side OAK publisher over SSH only while a capture needs it.

The OAK is attached to the robot by USB and is published by `serve_oak`. A robot
reboot clears the copied package in /tmp, so every start re-copies it. If the
endpoint is already being served (for example started by hand), it is reused
and left running. A server started here is stopped on close; if this PC dies,
the SSH session drops, the remote stdin closes and `serve_oak` exits on its own,
releasing the OAK.
"""
from collections import deque
from pathlib import Path
import queue
import shlex
import socket
import subprocess
import threading
import time
from urllib.parse import urlparse

PACKAGE = Path(__file__).resolve().parents[1]
SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
               '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=3']


def endpoint_address(endpoint):
    parsed = urlparse(endpoint)
    if parsed.scheme != 'tcp' or not parsed.hostname or not parsed.port:
        raise ValueError(f'Expected a tcp://host:port endpoint, got {endpoint!r}')
    return parsed.hostname, parsed.port


def endpoint_listening(endpoint, timeout_s=1.):
    try:
        with socket.create_connection(endpoint_address(endpoint), timeout=timeout_s):
            return True
    except OSError:
        return False


class RobotOakServer:
    def __init__(self, camera_config, *, ready_timeout_s=45., log=print):
        settings = camera_config['robot_server']
        self.endpoint = camera_config['endpoint']
        endpoint_address(self.endpoint)
        self.host = settings['host']
        self.remote_root = settings['remote_root']
        self.python = settings['python']
        if not self.remote_root.startswith('/'):
            raise ValueError('robot_server.remote_root must be an absolute path')
        self.ready_timeout_s = float(ready_timeout_s)
        self.log = log
        self.process = self.reader = None
        self.reused = False
        self.output = deque(maxlen=200)
        self._lines = queue.Queue()

    def ssh(self, remote_command):
        return ['ssh', *SSH_OPTIONS, self.host, remote_command]

    def sync(self):
        """Copy the package (without results/data) into the robot's remote root."""
        root = shlex.quote(self.remote_root)
        tar = subprocess.Popen(['tar', '-C', str(PACKAGE.parent), '-cf', '-',
                                '--exclude=results', '--exclude=data', '--exclude=__pycache__',
                                PACKAGE.name], stdout=subprocess.PIPE)
        try:
            copy = subprocess.run(self.ssh(f'mkdir -p {root} && tar -C {root} -xf -'),
                                  stdin=tar.stdout, capture_output=True, text=True, timeout=60)
        finally:
            tar.stdout.close()
            tar.wait()
        if tar.returncode or copy.returncode:
            raise RuntimeError(f'Could not copy the calibration package to {self.host}: '
                               f'{copy.stderr.strip() or "tar failed"}')

    def command(self):
        serve = shlex.join([self.python, '-u', '-m', 'g1_auto_calibration.scripts.serve_oak',
                            '--bind', self.endpoint, '--duration', '0', '--exit-on-stdin-eof'])
        return self.ssh(f'cd {shlex.quote(self.remote_root)} && exec {serve}')

    def _drain(self):
        for line in self.process.stdout:
            line = line.rstrip()
            self.output.append(line)
            self._lines.put(line)
        self._lines.put(None)

    def start(self):
        if endpoint_listening(self.endpoint):
            self.reused = True
            self.log(f'OAK endpoint {self.endpoint} is already served; reusing it', flush=True)
            return self
        self.log(f'Starting the OAK server on {self.host} ...', flush=True)
        self.sync()
        self.process = subprocess.Popen(self.command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.reader = threading.Thread(target=self._drain, name='robot-oak-server', daemon=True)
        self.reader.start()
        deadline = time.monotonic()+self.ready_timeout_s
        try:
            while True:
                try:
                    line = self._lines.get(timeout=max(.01, deadline-time.monotonic()))
                except queue.Empty:
                    raise TimeoutError(self._failure(f'not ready after {self.ready_timeout_s:.0f} s'))
                if line is None:
                    raise RuntimeError(self._failure('exited before it was ready'))
                if line.startswith('OAK ready'):
                    self.log(line, flush=True)
                    return self
        except BaseException:
            self.stop()
            raise

    def _failure(self, reason):
        tail = '\n'.join(list(self.output)[-15:]) or '(no output)'
        hint = ''
        if any(word in tail for word in ('X_LINK', 'No available devices', 'already', 'in use', 'Address already')):
            hint = '\nThe OAK or port may be in use by someone else; agree on a time slot first.'
        return f'Robot OAK server {reason}. Last output from {self.host}:\n{tail}{hint}'

    def stop(self):
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None:
            try:
                process.stdin.close()  # serve_oak exits on stdin EOF and closes the OAK
            except OSError:
                pass
            try:
                # The remote side also exits by itself once the SSH session is gone.
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if self.reader is not None:
            self.reader.join(timeout=2)
        self.log('Robot OAK server stopped; the OAK is released', flush=True)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def start_robot_server(camera_config, **kwargs):
    """Start the robot server for configs that declare one; otherwise return None."""
    if camera_config.get('backend') != 'network' or not camera_config.get('robot_server'):
        return None
    return RobotOakServer(camera_config, **kwargs).start()
