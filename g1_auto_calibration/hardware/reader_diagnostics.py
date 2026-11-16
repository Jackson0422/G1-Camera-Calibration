"""Bounded diagnostics on one existing SDK reader, with no DDS side effects.

The SDK hides exceptions from ``DataReader.take`` behind a generic message. This
observer retains exception details before re-raising the identical exception.
It never creates a reader, changes a class, or replaces another observer.
"""

from collections import deque
from copy import deepcopy
import math
import threading
import time
import traceback
import weakref


class ReaderDiagnostics:
    """Instrument only the existing LowState reader after subscriber startup."""

    COVERAGE = (
        'Only take() calls on the existing rt/lowstate reader after installation. '
        'Startup errors before installation and other readers, including RPC reply '
        'readers, are outside coverage. Listener arguments that never reach this '
        'reader.take() are also outside coverage.'
    )

    def __init__(self, *, capacity=8, clock=time.monotonic):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= 64:
            raise ValueError('Diagnostic capacity must be an integer from 1 to 64')
        if not callable(clock):
            raise TypeError('Diagnostic clock must be callable')
        self._clock = clock
        self._capacity = capacity
        self._lock = threading.Lock()
        self._reader_ref = None
        self._wrapper = None
        self._attempted = False
        self._installed = False
        self._active = False
        self._installation_error = None
        self._restore_status = 'not_requested'
        self._phase = 'initializing'
        self._take_count = 0
        self._success_count = 0
        self._exception_count = 0
        self._first_exception = None
        self._recent = deque(maxlen=capacity)

    def install(self, subscriber):
        """Guard private SDK traversal; unavailable diagnostics do not alter data."""
        with self._lock:
            if self._attempted:
                return False
            self._attempted = True
        try:
            channel = getattr(subscriber, '_ChannelSubscriber__channel')
            owner = getattr(channel, '_Channel__reader')
            reader = getattr(owner, '_Reader__reader')
            reader_ref = weakref.ref(reader)
            instance_fields = getattr(reader, '__dict__', None)
            if not isinstance(instance_fields, dict):
                raise TypeError('Existing reader has no writable instance dictionary')
            if 'take' in instance_fields:
                raise ValueError('Existing reader already owns a take override; left unchanged')
            original_take = getattr(type(reader), 'take', None)
            if not callable(original_take):
                raise TypeError('Existing reader class has no callable take method')
            observer_ref = weakref.ref(self)

            def observed_take(*args, **kwargs):
                # Both references are weak. Capturing a bound take method would
                # keep the reader alive through its own instance dictionary.
                current_reader = reader_ref()
                if current_reader is None:
                    raise ReferenceError('The observed reader has been released')
                observer = observer_ref()
                if observer is not None:
                    with observer._lock:
                        observer._take_count += 1
                try:
                    result = original_take(current_reader, *args, **kwargs)
                except Exception as error:
                    if observer is not None:
                        # Diagnostics must never replace the original failure.
                        try:
                            observer._record_exception(error)
                        except Exception:
                            pass
                    raise
                if observer is not None:
                    with observer._lock:
                        observer._success_count += 1
                return result

            setattr(reader, 'take', observed_take)
            with self._lock:
                self._reader_ref, self._wrapper = reader_ref, observed_take
                self._installed = self._active = True
                self._phase = 'active'
            return True
        except Exception as error:
            with self._lock:
                self._installation_error = self._safe_message(error)
                self._phase = 'active'
            return False

    @staticmethod
    def _safe_message(error):
        try:
            message = str(error)
        except Exception:
            message = '<exception message unavailable>'
        return message[:1024]

    def _record_exception(self, error):
        try:
            stamp = float(self._clock())
            stamp = stamp if math.isfinite(stamp) else None
        except Exception:
            stamp = None
        try:
            trace = ''.join(traceback.format_exception(type(error), error,
                error.__traceback__, limit=6))[-4096:]
        except Exception:
            trace = '<exception traceback unavailable>'
        record = dict(type=f'{type(error).__module__}.{type(error).__qualname__}',
                      message=self._safe_message(error), traceback=trace,
                      monotonic_s=stamp)
        with self._lock:
            record['phase'] = self._phase
            self._exception_count += 1
            if self._first_exception is None:
                self._first_exception = record
            self._recent.append(record)

    def mark_closing(self):
        with self._lock:
            self._phase = 'closing'

    def restore(self):
        """Remove only our own override, including when SDK close raises.

        The SDK may already have released the reader. An unrelated replacement
        is preserved. No strong reader reference is retained after this method.
        """
        with self._lock:
            reader_ref, wrapper = self._reader_ref, self._wrapper
        reader = reader_ref() if reader_ref is not None else None
        status = 'not_installed'
        active = False
        try:
            if reader is None and reader_ref is not None:
                status = 'reader_released'
            elif reader is not None:
                if reader.__dict__.get('take') is wrapper:
                    delattr(reader, 'take')
                    status = 'restored'
                else:
                    status = 'ownership_changed_left_unchanged'
        except Exception as error:
            status = 'restore_failed: ' + self._safe_message(error)
            active = bool(reader is not None and reader.__dict__.get('take') is wrapper)
        finally:
            with self._lock:
                self._active = active
                self._restore_status = status
                self._phase = 'closed'
                self._reader_ref = self._wrapper = None

    def snapshot(self):
        """Return serializable bounded details, without touching a DDS entity."""
        with self._lock:
            return deepcopy(dict(installed=self._installed, active=self._active,
                installation_error=self._installation_error,
                installation_attempted=self._attempted, phase=self._phase,
                coverage=self.COVERAGE, take_count=self._take_count,
                successful_take_count=self._success_count,
                exception_count=self._exception_count,
                first_exception=self._first_exception,
                recent_exceptions=list(self._recent),
                maximum_retained_exceptions=self._capacity,
                restore_status=self._restore_status))
