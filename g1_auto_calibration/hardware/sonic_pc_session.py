"""PC session adapters: qualified auto-start and bounded feedback de-duplication."""

import threading
import time


class AdvancingFeedbackClient:
    """Consume original authenticated packets, without refreshing their timestamps.

    Different feedback challenges can contain the same robot state snapshot.
    Discard those packets and require advancing ticks within the unchanged
    20 ms state freshness interval. A true freeze/reset still stops the session.
    """

    def __init__(self, client, *, clock=time.monotonic):
        self.client, self.clock = client, clock
        self.last_tick = None

    def __getattr__(self, name):
        return getattr(self.client, name)

    def receive_feedback(self, timeout=1.):
        deadline = self.clock()+timeout
        discarded = False
        while True:
            remaining = deadline-self.clock()
            if remaining <= 0:
                raise TimeoutError('No advancing authenticated robot state within the feedback deadline')
            feedback = self.client.receive_feedback(timeout=remaining)
            if feedback.get('ready') is not True:
                return feedback
            tick = feedback.get('tick')
            if type(tick) is not int or not 0 <= tick <= 0xffffffff:
                raise RuntimeError('Invalid authenticated robot state tick')
            delta = None if self.last_tick is None else (tick-self.last_tick) & 0xffffffff
            if delta is None or 0 < delta < 1 << 31:
                self.last_tick = tick
                return feedback
            kind = 'duplicate' if delta == 0 else 'reordered'
            self.client.stats[kind+'_tick_feedback_discarded'] += 1
            if not discarded:
                deadline = min(deadline, self.clock()+.020)
                discarded = True


class QualifiedAutoStartKeys:
    """Keep ordinary U/Y controls and start only on fresh qualified readiness."""

    def __init__(self, controls, operator_keys, events, *, clock=time.monotonic):
        self.controls, self.keys, self.events = controls, operator_keys(controls), events
        self.clock = clock
        self.closed = threading.Event()
        self.thread = None

    def __enter__(self):
        self.keys.__enter__()

        def run():
            try:
                while not self.closed.is_set() and not self.controls.stop.is_set():
                    if self.controls.motion_used:
                        return
                    if self.controls.start_ready and self.controls.request_start():
                        self.events.append(dict(local_monotonic_s=self.clock(),
                            event='automatic_start_after_fresh_qualified_standby',
                            robot_readiness_gates_bypassed=False))
                    self.closed.wait(.005)
            except Exception:
                self.controls.request_stop('Automatic start adapter failed')

        self.thread = threading.Thread(target=run, name='qualified-sonic-auto-start', daemon=True)
        self.thread.start()
        print('Automatic start enabled: qualified stable standby starts this motion once; U/Y controls remain active.', flush=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed.set()
        if self.thread is not None:
            self.thread.join(timeout=.1)
        return self.keys.__exit__(exc_type, exc_value, traceback)
