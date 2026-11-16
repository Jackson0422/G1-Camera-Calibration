"""Archive every camera packet and measured state; no motor-control interface."""

from pathlib import Path
import json
import subprocess
import threading
import time

import cv2
import numpy as np

from ..common import digest, plain, write_yaml
from ..execution.timing_trace import TimingTrace


class ContinuousVideo:
    """Frame indices join raw images, corners and timing records exactly.

    Receipt-time joint brackets are explicitly provisional. The full raw state
    and camera clocks are retained for later capture-time clock association.
    Video FPS is for review only; actual per-frame times live in frames.jsonl.
    """

    def __init__(self, output, buffer, *, camera='d435', duration_s=120., clock=time.monotonic):
        if not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError('Duration must be finite and positive')
        self.output, self.buffer = Path(output), buffer
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output/'frames').mkdir()
        self.stream = (self.output/'frames.jsonl').open('x')
        self.trace = TimingTrace(self.output/'timing_trace', buffer)
        self.camera, self.duration, self.clock = camera, float(duration_s), clock
        self.lock = threading.RLock()
        self.started = self.deadline = None
        self.count = self.good = 0
        self.last_sequence = self.error = None
        self.closed = False

    def start(self):
        self.trace.start()
        return self

    def begin(self):
        with self.lock:
            if self.started is not None or self.closed:
                raise RuntimeError('Continuous capture is one-shot')
            self.started = float(self.clock())
            self.deadline = self.started+self.duration

    def record(self, packet):
        with self.lock:
            frame = packet.frame
            if (self.closed or self.started is None or
                    not self.started <= frame.received_monotonic_s < self.deadline):
                return
            identity = (frame.source_clock, frame.sequence, frame.source_timestamp_s)
            if identity == self.last_sequence:
                return
            try:
                frame.validate()
                index = self.count
                image_file = f'frames/{index:06d}.png'
                if packet.png is not None:
                    (self.output/image_file).write_bytes(packet.png)
                elif not cv2.imwrite(str(self.output/image_file), frame.bgr):
                    raise OSError('Lossless frame write failed')
                with self.buffer.lock:
                    states = list(self.buffer.states)
                times = np.asarray([s.received_s for s in states])
                right = int(np.searchsorted(times, frame.received_monotonic_s))
                brackets = []
                for j in (right-1, right):
                    if 0 <= j < len(states):
                        s = states[j]
                        brackets.append(dict(received_s=s.received_s, tick=s.tick,
                            q_rad=s.q, dq_rad_s=s.dq, mode_pr=s.mode_pr,
                            mode_machine=s.mode_machine, imu_quaternion_wxyz=getattr(s, 'imu_quaternion_wxyz', None)))
                observation = packet.observation
                record = dict(video_frame_index=index, image_file=image_file,
                    image_sha256=digest(self.output/image_file), camera=self.camera,
                    source_sequence=frame.sequence, source_timestamp_s=frame.source_timestamp_s,
                    source_clock=frame.source_clock, received_monotonic_s=frame.received_monotonic_s,
                    serial=frame.serial, optical_frame=frame.optical_frame,
                    K=frame.K, distortion=frame.distortion,
                    board_observation=observation, board_rejection_reason=packet.rejection_reason,
                    receive_time_joint_brackets=brackets,
                    capture_time_joint_alignment_verified=False,
                    alignment_status='pending validated camera/state clock association',
                    calibration_accepted=False)
                self.stream.write(json.dumps(plain(record), allow_nan=False)+'\n')
                self.trace.record_camera(self.camera, packet)
                self.count += 1
                self.good += observation is not None
                self.last_sequence = identity
            except Exception as exc:
                self.error = f'{type(exc).__name__}: {exc}'
                raise

    def close(self, *, render_video=True):
        with self.lock:
            stopped = float(self.clock())
            self.closed = True
            self.stream.close()
        elapsed = 0. if self.started is None else max(0., min(stopped, self.deadline)-self.started)
        window_complete = self.deadline is not None and stopped >= self.deadline
        trace = self.trace.close()
        video = None
        video_error = None
        if render_video and self.count:
            print(f'Encoding and checking review video ({self.count} frames, may take a minute); '
                  'do not press Ctrl+C or recording.yaml will not be written', flush=True)
            try:
                video = render_review_video(self.output, self.count, elapsed)
            except Exception as exc:
                video_error = f'{type(exc).__name__}: {exc}'
        result = dict(kind='continuous_camera_joint_archive', camera=self.camera,
            frame_count=self.count, board_detected_frames=self.good, raw_images='frames/*.png',
            frame_index='frames.jsonl', frame_index_sha256=digest(self.output/'frames.jsonl'),
            requested_duration_s=self.duration, record_start_received_s=self.started,
            requested_record_end_received_s=self.deadline, timing_trace=trace,
            stopped_received_s=stopped, elapsed_recording_s=elapsed,
            requested_window_complete=window_complete,
            video=video, video_error=video_error, error=self.error,
            complete=window_complete and self.count > 0 and self.error is None
                and trace['complete'] and (not render_video or video is not None),
            capture_time_joint_alignment_verified=False, calibration_ready=False,
            read_only=True, robot_control_commands_sent=0,
            note='Raw lossless image/frame ID and corners match exactly; receipt brackets are not exposure-time joint poses.')
        write_yaml(self.output/'recording.yaml', result)
        return result


def render_review_video(directory, count, elapsed_s):
    """Encode frames/*.png into review.mp4 with exactly one video frame per raw frame.

    Playback is retimed to the recorded interval; frames.jsonl keeps the original clocks.
    """
    directory = Path(directory)
    destination = directory/'review.mp4'
    if elapsed_s <= 0:
        raise ValueError('Recording has no positive elapsed interval')
    subprocess.run(['ffmpeg', '-n', '-loglevel', 'error', '-framerate', str(count/elapsed_s),
        '-i', str(directory/'frames/%06d.png'), '-frames:v', str(count),
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(destination)], check=True, timeout=600.)
    check = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=nb_read_frames', '-of', 'json',
        str(destination)], check=True, capture_output=True, text=True, timeout=120.)
    if int(json.loads(check.stdout)['streams'][0]['nb_read_frames']) != count:
        raise ValueError('Review video frame count differs from the raw frame index')
    return dict(file=destination.name, sha256=digest(destination), frame_count=count,
                display_duration_s=elapsed_s, presentation_times_are_capture_times=False)
