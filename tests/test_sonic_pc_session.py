from collections import Counter
import threading
import time
from types import SimpleNamespace

import pytest

from g1_auto_calibration.hardware.sonic_pc_session import AdvancingFeedbackClient, QualifiedAutoStartKeys


class Client:
    def __init__(self, packets, now):
        self.packets, self.now, self.stats = iter(packets), now, Counter()

    def receive_feedback(self, timeout):
        delay, packet = next(self.packets)
        if delay > timeout:
            self.now[0] += timeout
            raise TimeoutError('Original receive timeout')
        self.now[0] += delay
        return packet


def test_duplicate_authenticated_snapshot_is_discarded_without_retimestamping():
    now = [0.]
    a = dict(ready=True, tick=10, received_at=.001)
    duplicate = dict(ready=True, tick=10, received_at=.002)
    b = dict(ready=True, tick=11, received_at=.003)
    base = Client([(.001, a), (.001, duplicate), (.001, b)], now)
    client = AdvancingFeedbackClient(base, clock=lambda: now[0])
    assert client.receive_feedback() is a
    assert client.receive_feedback() is b
    assert b['received_at'] == .003
    assert base.stats['duplicate_tick_feedback_discarded'] == 1


def test_receipt_packets_cannot_extend_a_true_tick_freeze_beyond20ms():
    now = [0.]
    packets = [(.001, dict(ready=True, tick=10, received_at=i*.001)) for i in range(100)]
    base = Client(packets, now)
    client = AdvancingFeedbackClient(base, clock=lambda: now[0])
    client.receive_feedback()
    with pytest.raises(TimeoutError):
        client.receive_feedback(timeout=.08)
    assert now[0] <= .022+1e-9


def test_out_of_order_feedback_is_bounded_and_uint32_wrap_advances():
    now = [0.]
    packets = [(.001, dict(ready=True, tick=tick)) for tick in (0xffffffff, 0xfffffffe, 0)]
    base = Client(packets, now)
    client = AdvancingFeedbackClient(base, clock=lambda: now[0])
    assert client.receive_feedback()['tick'] == 0xffffffff
    assert client.receive_feedback()['tick'] == 0
    assert base.stats['reordered_tick_feedback_discarded'] == 1


def test_terminal_unready_feedback_is_never_hidden_by_tick_filter():
    now = [0.]
    fault = dict(ready=False, tick=10, motion_phase='FAULT')
    base = Client([(.001, dict(ready=True, tick=10)), (.001, fault)], now)
    client = AdvancingFeedbackClient(base, clock=lambda: now[0])
    client.receive_feedback()
    assert client.receive_feedback() is fault


def test_auto_start_waits_for_original_qualified_readiness_and_starts_once():
    entered, exited, started = threading.Event(), threading.Event(), threading.Event()
    controls = SimpleNamespace(stop=threading.Event(), motion_used=False, start_ready=False, calls=0)

    def request():
        assert controls.start_ready
        controls.calls += 1
        controls.motion_used = True
        started.set()
        return True

    controls.request_start = request
    controls.request_stop = lambda reason: controls.stop.set()

    class Keys:
        def __init__(self, controls):
            pass

        def __enter__(self):
            entered.set()

        def __exit__(self, *args):
            exited.set()

    events = []
    with QualifiedAutoStartKeys(controls, Keys, events):
        assert entered.is_set()
        assert not started.wait(.03)
        controls.start_ready = True
        assert started.wait(1.)
        time.sleep(.02)
        assert controls.calls == 1
    assert exited.is_set() and len(events) == 1
    assert not events[0]['robot_readiness_gates_bypassed']
