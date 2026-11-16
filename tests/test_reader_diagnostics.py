"""Reader diagnostics use local fake instances and never initialize DDS."""

from collections import deque
import gc
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from g1_auto_calibration.hardware.reader_diagnostics import ReaderDiagnostics
from g1_auto_calibration.hardware.unitree_state import UnitreeState


class FakeReader:
    def __init__(self, outcomes=()):
        self.outcomes = deque(outcomes)

    def take(self, count=1, **kwargs):
        result = self.outcomes.popleft() if self.outcomes else []
        if isinstance(result, Exception):
            raise result
        return result


def fake_subscriber(reader):
    return SimpleNamespace(_ChannelSubscriber__channel=SimpleNamespace(
        _Channel__reader=SimpleNamespace(_Reader__reader=reader)))


class ReaderDiagnosticTests(unittest.TestCase):
    def test_observes_only_one_instance_and_keeps_class_unchanged(self):
        original = FakeReader.take
        answer = [object()]
        observed = FakeReader([answer])
        other = FakeReader([RuntimeError('RPC reader remains outside this observer')])
        diagnostic = ReaderDiagnostics()
        self.assertTrue(diagnostic.install(fake_subscriber(observed)))
        self.assertIs(observed.take(1), answer)
        with self.assertRaisesRegex(RuntimeError, 'RPC reader'):
            other.take(1)
        snapshot = diagnostic.snapshot()
        self.assertTrue(snapshot['installed'])
        self.assertEqual(snapshot['take_count'], 1)
        self.assertEqual(snapshot['successful_take_count'], 1)
        self.assertEqual(snapshot['exception_count'], 0)
        self.assertIs(FakeReader.take, original)
        self.assertNotIn('take', other.__dict__)
        self.assertIn('Startup errors before installation', snapshot['coverage'])
        self.assertIn('RPC reply', snapshot['coverage'])

    def test_preserves_exception_identity_traceback_and_bounded_ring(self):
        errors = [ValueError(f'offline test failure {index}') for index in range(5)]
        reader = FakeReader(errors)
        diagnostic = ReaderDiagnostics(capacity=3, clock=lambda: 123.)
        diagnostic.install(fake_subscriber(reader))
        for error in errors:
            try:
                reader.take()
            except ValueError as actual:
                self.assertIs(actual, error)
            else:
                self.fail('The original failure was suppressed')
        snapshot = diagnostic.snapshot()
        self.assertEqual(snapshot['exception_count'], 5)
        self.assertEqual(len(snapshot['recent_exceptions']), 3)
        self.assertEqual(snapshot['first_exception']['message'], 'offline test failure 0')
        self.assertEqual(snapshot['recent_exceptions'][0]['message'], 'offline test failure 2')
        first = snapshot['first_exception']
        self.assertEqual(first['type'], 'builtins.ValueError')
        self.assertEqual(first['phase'], 'active')
        self.assertEqual(first['monotonic_s'], 123.)
        self.assertIn('raise result', first['traceback'])
        snapshot['first_exception']['message'] = 'changed copy'
        self.assertNotEqual(diagnostic.snapshot()['first_exception']['message'], 'changed copy')

    def test_lifecycle_restores_only_the_owned_instance_override(self):
        reader = FakeReader([RuntimeError('while closing')])
        diagnostic = ReaderDiagnostics()
        diagnostic.install(fake_subscriber(reader))
        diagnostic.mark_closing()
        with self.assertRaises(RuntimeError):
            reader.take()
        self.assertEqual(diagnostic.snapshot()['first_exception']['phase'], 'closing')
        diagnostic.restore()
        self.assertNotIn('take', reader.__dict__)
        snapshot = diagnostic.snapshot()
        self.assertTrue(snapshot['installed'])
        self.assertFalse(snapshot['active'])
        self.assertEqual(snapshot['phase'], 'closed')
        self.assertEqual(snapshot['restore_status'], 'restored')
        self.assertEqual(reader.take(), [])
        self.assertEqual(diagnostic.snapshot()['take_count'], 1)

    def test_unrelated_override_is_preserved_during_restore(self):
        reader = FakeReader()
        diagnostic = ReaderDiagnostics()
        diagnostic.install(fake_subscriber(reader))
        replacement = lambda *args, **kwargs: ['other owner']
        reader.take = replacement
        diagnostic.restore()
        self.assertIs(reader.take, replacement)
        self.assertEqual(diagnostic.snapshot()['restore_status'], 'ownership_changed_left_unchanged')

    def test_reader_can_be_collected_while_observer_still_exists(self):
        reader = FakeReader()
        subscriber = fake_subscriber(reader)
        reference = weakref.ref(reader)
        diagnostic = ReaderDiagnostics()
        diagnostic.install(subscriber)
        del reader, subscriber
        gc.collect()
        self.assertIsNone(reference())
        diagnostic.mark_closing()
        diagnostic.restore()
        self.assertEqual(diagnostic.snapshot()['restore_status'], 'reader_released')

    def test_reader_keeps_working_after_observer_is_collected(self):
        reader = FakeReader([[1]])
        diagnostic = ReaderDiagnostics()
        reference = weakref.ref(diagnostic)
        diagnostic.install(fake_subscriber(reader))
        del diagnostic
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(reader.take(), [1])

    def test_unavailable_private_layout_and_existing_override_are_guarded(self):
        for subscriber in (object(), fake_subscriber(None)):
            diagnostic = ReaderDiagnostics()
            self.assertFalse(diagnostic.install(subscriber))
            snapshot = diagnostic.snapshot()
            self.assertFalse(snapshot['installed'])
            self.assertTrue(snapshot['installation_error'])
            self.assertEqual(snapshot['exception_count'], 0)
            diagnostic.restore()
        reader = FakeReader()
        replacement = lambda *args, **kwargs: ['prior override']
        reader.take = replacement
        diagnostic = ReaderDiagnostics()
        self.assertFalse(diagnostic.install(fake_subscriber(reader)))
        self.assertIs(reader.take, replacement)
        self.assertIn('already owns', diagnostic.snapshot()['installation_error'])

    def test_installation_is_single_use_and_large_error_details_are_bounded(self):
        reader = FakeReader([RuntimeError('x' * 10000)])
        diagnostic = ReaderDiagnostics()
        self.assertTrue(diagnostic.install(fake_subscriber(reader)))
        wrapper = reader.take
        self.assertFalse(diagnostic.install(fake_subscriber(FakeReader())))
        self.assertIs(reader.take, wrapper)
        with self.assertRaises(RuntimeError):
            reader.take()
        error = diagnostic.snapshot()['first_exception']
        self.assertLessEqual(len(error['message']), 1024)
        self.assertLessEqual(len(error['traceback']), 4096)

    def test_unitree_state_installs_after_existing_subscriber_and_restores_on_close_error(self):
        calls = []
        reader = FakeReader([ValueError('original lowstate read error')])

        class Subscriber:
            def __init__(self, topic, data_type):
                self._ChannelSubscriber__channel = fake_subscriber(reader)._ChannelSubscriber__channel
                calls.append(('subscriber', topic))

            def Init(self, handler, queue):
                calls.append(('init', queue, 'take' in reader.__dict__))

            def Close(self):
                calls.append(('close', state.reader_diagnostics_snapshot()['phase']))
                raise RuntimeError('offline close failure')

        channel_module = SimpleNamespace(ChannelFactoryInitialize=lambda *args: calls.append(('factory', args)),
                                         ChannelSubscriber=Subscriber)
        idl_module = SimpleNamespace(LowState_=object())
        with patch.dict(sys.modules, {'unitree_sdk2py.core.channel': channel_module,
                                      'unitree_sdk2py.idl.unitree_hg.msg.dds_': idl_module}), \
                patch.object(sys, 'path', list(sys.path)):
            state = UnitreeState(dict(sdk_source=str(Path('/unused/offline-test-sdk')),
                topic='rt/lowstate', domain=0, interface='offline_fake_interface'))
        self.assertEqual(calls[2], ('init', 10, False))
        self.assertTrue(state.reader_diagnostics_snapshot()['installed'])
        with self.assertRaises(ValueError):
            reader.take()
        self.assertIsNone(state.error)
        with self.assertRaisesRegex(RuntimeError, 'offline close failure'):
            state.close()
        self.assertIn(('close', 'closing'), calls)
        self.assertNotIn('take', reader.__dict__)
        self.assertEqual(state.reader_diagnostics_snapshot()['exception_count'], 1)
        self.assertEqual(state.reader_diagnostics_snapshot()['phase'], 'closed')


if __name__ == '__main__':
    unittest.main()
