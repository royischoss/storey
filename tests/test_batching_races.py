# Copyright 2026 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Tests verifying fixes for race conditions in storey _Batching.

Races happen at `await` points where the event loop switches between the
timer task (_sleep_and_emit) and the run loop processing events.  We use
an asyncio.Event gate inside _emit to deterministically yield control at
the exact moment needed to verify each fix.

Tests use build_flow with AsyncEmitSource to exercise the full graph,
matching production configuration.
"""

import asyncio
import time as _time
from datetime import datetime as _dt

import pytest

from storey import AsyncEmitSource, BatchWriteCancelledError, Event, build_flow
from storey.dtypes import _termination_obj
from storey.flow import Flow, _Batching

# ---------------------------------------------------------------------------
# Helpers — real _Batching subclasses matching production config
# ---------------------------------------------------------------------------


class GatedTarget(_Batching):
    """_Batching subclass whose _emit blocks on an asyncio.Event gate.

    Matches production: _do_downstream_per_event=True (same as ParquetTarget
    and TimescaleDBTarget which inherit from _Batching without overriding it).
    """

    # Same as production — ParquetTarget/TimescaleDBTarget don't override this
    _do_downstream_per_event = True

    def __init__(self, gate: asyncio.Event = None, **kwargs):
        super().__init__(**kwargs)
        self.gate = gate
        self.emitted_batches: list[list] = []
        self.emitted_batch_events: list[list] = []
        self.emit_count = 0

    async def _emit(self, batch, batch_key, batch_time, batch_events, last_event_time=None):
        self.emit_count += 1
        if self.gate is not None:
            # Simulates a slow S3 write (ParquetTarget) or TSDB write (TimescaleDBTarget)
            # Both are async operations that yield to the event loop
            await self.gate.wait()
        self.emitted_batches.append(list(batch))
        self.emitted_batch_events.append(list(batch_events))


class RecordingTarget(_Batching):
    """_Batching subclass that captures state during _terminate."""

    _do_downstream_per_event = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.emitted_batches: list[list] = []
        self.terminate_called = False
        self.timeout_task_alive_during_terminate = None

    async def _emit(self, batch, batch_key, batch_time, batch_events, last_event_time=None):
        self.emitted_batches.append(list(batch))

    async def _terminate(self):
        # Capture _timeout_task state at the moment _terminate is called.
        # In production, this is where ParquetTarget closes file handles
        # and TimescaleDBTarget closes the connection pool.
        task = self._timeout_task
        if task is not None:
            self.timeout_task_alive_during_terminate = not task.done()
        else:
            self.timeout_task_alive_during_terminate = False
        self.terminate_called = True


class TerminationProbe(Flow):
    """Downstream step recording whether it received the termination signal."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.terminated = False

    async def _do(self, event):
        if event is _termination_obj:
            self.terminated = True
        return await self._do_downstream(event)


class NonCopyableError(Exception):
    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__(detail)


class FailingTerminationTarget(_Batching):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.terminate_called = False

    async def _emit(self, batch, batch_key, batch_time, batch_events, last_event_time=None):
        raise RuntimeError("final emit failed")

    async def _terminate(self):
        self.terminate_called = True


class KeyedGatedTarget(_Batching):
    """Keyed batching target with deterministic per-physical-batch emit gates."""

    _do_downstream_per_event = True

    def __init__(
        self,
        blocked_batch_keys=None,
        failed_batch_keys=None,
        errors_by_batch_key=None,
        cancelled_batch_keys=None,
        terminate_error=None,
        **kwargs,
    ):
        kwargs.setdefault("_flush_key_field", "$key")
        super().__init__(**kwargs)
        self.blocked_batch_keys = set(blocked_batch_keys or ())
        self.failed_batch_keys = set(failed_batch_keys or ())
        self.errors_by_batch_key = dict(errors_by_batch_key or {})
        self.cancelled_batch_keys = set(cancelled_batch_keys or ())
        self.terminate_error = terminate_error
        self.terminate_called = False
        self.emitted_batches = []
        self.emitted_batch_events = []
        self.emit_count_by_key = {}
        self._accepted_events = {}
        self._emit_started_events = {}
        self._emit_count_events = {}
        self._emit_finished_events = {}
        self._release_events = {}

    @staticmethod
    def _event_for(events, key):
        event = events.get(key)
        if event is None:
            event = asyncio.Event()
            events[key] = event
        return event

    def accepted_event(self, value):
        return self._event_for(self._accepted_events, value)

    def emit_started_event(self, batch_key):
        return self._event_for(self._emit_started_events, batch_key)

    def emit_count_event(self, batch_key, count):
        return self._event_for(self._emit_count_events, (batch_key, count))

    def emit_finished_event(self, batch_key):
        return self._event_for(self._emit_finished_events, batch_key)

    def release(self, batch_key):
        self._event_for(self._release_events, batch_key).set()

    def _event_to_batch_entry(self, event):
        entry = super()._event_to_batch_entry(event)
        self.accepted_event(event.body["v"]).set()
        return entry

    async def flush(self, flush_key):
        await self._flush_by_key(flush_key)

    async def _emit(self, batch, batch_key, batch_time, batch_events, last_event_time=None):
        self.emit_count_by_key[batch_key] = self.emit_count_by_key.get(batch_key, 0) + 1
        self.emit_count_event(batch_key, self.emit_count_by_key[batch_key]).set()
        self.emit_started_event(batch_key).set()
        try:
            if batch_key in self.blocked_batch_keys:
                await self._event_for(self._release_events, batch_key).wait()
            if batch_key in self.cancelled_batch_keys:
                raise asyncio.CancelledError()
            if batch_key in self.errors_by_batch_key:
                raise self.errors_by_batch_key[batch_key]
            if batch_key in self.failed_batch_keys:
                raise RuntimeError(f"emit failed for {batch_key}")
            self.emitted_batches.append((batch_key, list(batch)))
            self.emitted_batch_events.append((batch_key, list(batch_events)))
        finally:
            self.emit_finished_event(batch_key).set()

    async def _terminate(self):
        self.terminate_called = True
        if self.terminate_error is not None:
            raise self.terminate_error


def _ev(value, key=None):
    """Create an Event with a dict body."""
    return Event({"v": value}, key=key)


def _partitioned_ev(value, logical_key, physical_key):
    return Event({"v": value, "partition": physical_key}, key=logical_key)


async def _emit_and_wait_until_accepted(controller, target, event):
    accepted = target.accepted_event(event.body["v"])
    await controller.emit(event)
    await accepted.wait()


async def _assert_pending(task, loop_turns=5):
    # Give the task several chances to run, so this proves it stays blocked rather
    # than merely that it had not finished within a single loop turn.
    for _ in range(loop_turns):
        await asyncio.sleep(0)
    done, _ = await asyncio.wait({task}, timeout=0)
    assert not done


# ---------------------------------------------------------------------------
# Tests — each verifies a race condition fix
# ---------------------------------------------------------------------------


class TestBatchingConcurrentEmitRace:
    """Verify concurrent timer and max-events calls to _Batching._emit_batch.

    Production trigger: event source rebalance/drain →
    drain_callback calls controller.terminate(wait=True) →
    AsyncEmitSource emits _termination_obj → propagates to _Batching._do →
    _emit_all → _emit_batch → _emit (the slow target write).

    Meanwhile, _sleep_and_emit (timer task) runs independently.
    """

    def test_race1_concurrent_emit_batch_keyerror(self):
        """Timer and max_events both call _emit_batch for the same key.

        Time  | Run loop (_do)                     | Timer (_sleep_and_emit)
        ------+------------------------------------+------------------------------------
          t0  | emit(ev1) -> append, start timer   | sleeping...
          t1  | idle at _q.get()                   | wakes -> _emit_batch(None)
          t2  | idle                               |   _batch.pop(None) -> batch1
          t3  | idle                               |   await _emit(batch1) YIELDS
        ------+------------------------------------+-- -- -- -- -- -- -- -- -- -- -- --
          t4  | emit(ev2) -> append                |   (suspended in _emit)
          t5  | emit(ev3) -> max_events hit        |   (suspended in _emit)
          t6  |   _emit_batch(None)                |   (suspended in _emit)
          t7  |     _batch.pop(None) -> batch2     |   (suspended in _emit)
          t8  |     await _emit(batch2) YIELDS     |   (suspended in _emit)
        ------+-- -- -- -- -- -- -- -- -- -- -- -- +-- -- -- -- -- -- -- -- -- -- -- --
          t9  |   (suspended)                      |   _emit returns
          t10 |   (suspended)                      |   pop _batch_events[None] (ok)
        ------+-- -- -- -- -- -- -- -- -- -- -- -- +------------------------------------
          t11 |   _emit returns                    |
          t12 |   pop _batch_events[None]          |   FIX: separate list, no KeyError
        """

        async def _test():
            gate = asyncio.Event()
            target = GatedTarget(gate=gate, max_events=2, flush_after_seconds=0.01)
            controller = build_flow([AsyncEmitSource(), target]).run()

            # t0: Event 1 starts the timer
            await controller.emit(_ev(1))

            # t1-t3: Wait for timer to fire (flush_after_seconds=0.01) and block on gate
            await asyncio.sleep(0.05)
            assert target.emit_count == 1, "Timer should have started _emit"

            # t4-t8: Events 2,3 arrive while _emit is blocked.
            # Event 3 hits max_events=2 -> _do calls _emit_batch(None).
            await controller.emit(_ev(2))
            await controller.emit(_ev(3))
            await asyncio.sleep(0)  # yield to let max_events _emit_batch reach gate

            # t9-t12: Release gate — both _emit_batch calls complete without KeyError
            gate.set()
            await asyncio.sleep(0)  # yield to let both _emit calls finish

            await controller.terminate()
            await controller.await_termination()

        asyncio.run(_test())


class TestKeyedBatchFlush:
    @staticmethod
    def _target(**kwargs):
        return KeyedGatedTarget(key_field=lambda event: event.body["partition"], **kwargs)

    def test_flush_emits_buffered_matching_key(self):
        async def _test():
            target = self._target(blocked_batch_keys={"partition-A"})
            controller = build_flow([AsyncEmitSource(), target]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "partition-A"))

            flush_task = asyncio.create_task(target.flush("endpoint-A"))
            await target.emit_started_event("partition-A").wait()
            await _assert_pending(flush_task)

            target.release("partition-A")
            await flush_task
            assert target.emit_count_by_key == {"partition-A": 1}
            assert [event.body["v"] for _, events in target.emitted_batch_events for event in events] == [1]

            await controller.terminate(wait=True)

        asyncio.run(_test())

    def test_flush_waits_for_matching_batch_already_in_flight(self):
        async def _test():
            target = self._target(max_events=1, blocked_batch_keys={"partition-A"})
            controller = build_flow([AsyncEmitSource(), target]).run()
            await controller.emit(_partitioned_ev(1, "endpoint-A", "partition-A"))
            await target.emit_started_event("partition-A").wait()

            flush_task = asyncio.create_task(target.flush("endpoint-A"))
            await _assert_pending(flush_task)

            target.release("partition-A")
            await flush_task
            await controller.terminate(wait=True)

        asyncio.run(_test())

    def test_flush_isolates_unrelated_keys(self):
        async def _test():
            target = self._target(blocked_batch_keys={"partition-A"})
            controller = build_flow([AsyncEmitSource(), target]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "partition-A"))
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(2, "endpoint-B", "partition-B"))

            flush_a = asyncio.create_task(target.flush("endpoint-A"))
            await target.emit_started_event("partition-A").wait()
            assert not target.emit_started_event("partition-B").is_set()

            await target.flush("endpoint-B")
            assert target.emit_finished_event("partition-B").is_set()
            await _assert_pending(flush_a)

            target.release("partition-A")
            await flush_a
            await controller.terminate(wait=True)

        asyncio.run(_test())

    def test_concurrent_fences_for_same_key_share_one_write(self):
        async def _test():
            target = self._target(blocked_batch_keys={"partition-A"})
            controller = build_flow([AsyncEmitSource(), target]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "partition-A"))

            first_flush = asyncio.create_task(target.flush("endpoint-A"))
            await target.emit_started_event("partition-A").wait()
            second_flush = asyncio.create_task(target.flush("endpoint-A"))
            await _assert_pending(first_flush)
            await _assert_pending(second_flush)

            target.release("partition-A")
            await asyncio.gather(first_flush, second_flush)
            assert target.emit_count_by_key == {"partition-A": 1}
            assert [event.body["v"] for _, events in target.emitted_batch_events for event in events] == [1]
            assert "endpoint-A" not in target._flush_states

            await controller.terminate(wait=True)

        asyncio.run(_test())

    def test_unknown_key_flush_does_not_retain_state(self):
        async def _test():
            target = self._target()
            controller = build_flow([AsyncEmitSource(), target]).run()

            await target.flush("endpoint-never-seen")

            assert not target._flush_states
            await controller.terminate(wait=True)

        asyncio.run(_test())

    def test_flush_awaits_all_physical_batches_for_logical_key(self):
        async def _test():
            target = self._target(blocked_batch_keys={"hour-10", "hour-11"})
            controller = build_flow([AsyncEmitSource(), target]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "hour-10"))
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(2, "endpoint-A", "hour-11"))

            flush_task = asyncio.create_task(target.flush("endpoint-A"))
            await asyncio.gather(
                target.emit_started_event("hour-10").wait(),
                target.emit_started_event("hour-11").wait(),
            )

            target.release("hour-10")
            await target.emit_finished_event("hour-10").wait()
            await _assert_pending(flush_task)

            target.release("hour-11")
            await flush_task
            assert target.emit_count_by_key == {"hour-10": 1, "hour-11": 1}

            await controller.terminate(wait=True)

        asyncio.run(_test())

    def test_mixed_logical_keys_share_physical_batch_completion(self):
        async def _test():
            target = self._target(blocked_batch_keys={"shared"})
            controller = build_flow([AsyncEmitSource(), target]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "shared"))
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(2, "endpoint-B", "shared"))

            flush_a = asyncio.create_task(target.flush("endpoint-A"))
            await target.emit_started_event("shared").wait()
            flush_b = asyncio.create_task(target.flush("endpoint-B"))
            await _assert_pending(flush_a)
            await _assert_pending(flush_b)

            target.release("shared")
            await asyncio.gather(flush_a, flush_b)
            assert target.emit_count_by_key == {"shared": 1}
            assert sorted(event.body["v"] for _, events in target.emitted_batch_events for event in events) == [1, 2]

            await controller.terminate(wait=True)

        asyncio.run(_test())

    def test_flush_propagates_and_retains_emit_failure(self):
        async def _test():
            target = self._target(failed_batch_keys={"partition-A"})
            controller = build_flow([AsyncEmitSource(), target]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "partition-A"))

            with pytest.raises(RuntimeError, match="emit failed for partition-A"):
                await target.flush("endpoint-A")
            target.failed_batch_keys.clear()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(2, "endpoint-A", "partition-A"))
            with pytest.raises(RuntimeError, match="emit failed for partition-A"):
                await target.flush("endpoint-A")
            assert target.emit_count_by_key == {"partition-A": 2}
            assert [event.body["v"] for _, events in target.emitted_batch_events for event in events] == [2]

            with pytest.raises(RuntimeError, match="emit failed for partition-A"):
                await controller.terminate(wait=True)

        asyncio.run(_test())

    def test_flush_retains_non_copyable_write_error(self):
        async def _test():
            write_error = NonCopyableError(503, "storage unavailable")
            target = self._target(errors_by_batch_key={"partition-A": write_error})
            controller = build_flow([AsyncEmitSource(), target]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "partition-A"))

            for _ in range(2):
                with pytest.raises(NonCopyableError) as exc_info:
                    await target.flush("endpoint-A")
                assert exc_info.value.code == 503
                assert exc_info.value.detail == "storage unavailable"
                assert not target._flush_states["endpoint-A"].in_flight

            with pytest.raises(NonCopyableError):
                await controller.terminate(wait=True)
            assert target.terminate_called

        asyncio.run(_test())

    def test_composite_event_key_is_normalized_for_flush(self):
        async def _test():
            target = self._target()
            controller = build_flow([AsyncEmitSource(), target]).run()
            await _emit_and_wait_until_accepted(
                controller,
                target,
                _partitioned_ev(1, ["tenant-1", "endpoint-A"], "partition-A"),
            )

            await target.flush(["tenant-1", "endpoint-A"])

            assert target.emit_count_by_key == {"partition-A": 1}
            assert not target._flush_states
            await controller.terminate(wait=True)

        asyncio.run(_test())

    def test_timer_and_max_events_race_with_keyed_tracking(self):
        async def _test():
            target = self._target(
                max_events=2,
                flush_after_seconds=0.01,
                blocked_batch_keys={"partition-A"},
            )
            controller = build_flow([AsyncEmitSource(), target]).run()

            await controller.emit(_partitioned_ev(1, "endpoint-A", "partition-A"))
            await target.emit_count_event("partition-A", 1).wait()

            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(2, "endpoint-A", "partition-A"))
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(3, "endpoint-A", "partition-A"))
            await target.emit_count_event("partition-A", 2).wait()

            target.release("partition-A")
            await controller.terminate(wait=True)

            emitted_values = [event.body["v"] for _, events in target.emitted_batch_events for event in events]
            assert sorted(emitted_values) == [1, 2, 3]
            assert target.emit_count_by_key == {"partition-A": 2}

        asyncio.run(_test())

    def test_events_arriving_after_fence_remain_for_next_flush(self):
        async def _test():
            target = self._target(blocked_batch_keys={"partition-A"})
            controller = build_flow([AsyncEmitSource(), target]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "partition-A"))

            first_flush = asyncio.create_task(target.flush("endpoint-A"))
            await target.emit_started_event("partition-A").wait()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(2, "endpoint-A", "partition-A"))

            target.release("partition-A")
            await first_flush
            assert target.emit_count_by_key == {"partition-A": 1}

            await target.flush("endpoint-A")
            assert target.emit_count_by_key == {"partition-A": 2}
            emitted_values = [event.body["v"] for _, events in target.emitted_batch_events for event in events]
            assert emitted_values == [1, 2]

            await controller.terminate(wait=True)

        asyncio.run(_test())

    def test_timed_out_waiter_does_not_cancel_or_forget_write(self):
        async def _test():
            target = self._target(blocked_batch_keys={"partition-A"})
            controller = build_flow([AsyncEmitSource(), target]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "partition-A"))

            flush_task = asyncio.create_task(target.flush("endpoint-A"))
            await target.emit_started_event("partition-A").wait()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(flush_task, timeout=0)

            state = target._flush_states.get("endpoint-A")
            assert state is not None
            assert state.in_flight
            assert all(not task.cancelled() for _, task in state.in_flight.values())
            target.release("partition-A")
            await target.emit_finished_event("partition-A").wait()
            await target.flush("endpoint-A")
            assert target.emit_count_by_key == {"partition-A": 1}

            await controller.terminate(wait=True)

        asyncio.run(_test())

    def test_failed_write_survives_waiter_timeout_and_reaches_termination(self):
        async def _test():
            target = self._target(
                blocked_batch_keys={"partition-A"},
                failed_batch_keys={"partition-A"},
            )
            controller = build_flow([AsyncEmitSource(), target]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "partition-A"))

            flush_task = asyncio.create_task(target.flush("endpoint-A"))
            await target.emit_started_event("partition-A").wait()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(flush_task, timeout=0)

            target.release("partition-A")
            await target.emit_finished_event("partition-A").wait()
            with pytest.raises(RuntimeError, match="emit failed for partition-A"):
                await target.flush("endpoint-A")
            with pytest.raises(RuntimeError, match="emit failed for partition-A"):
                await controller.terminate(wait=True)

        asyncio.run(_test())

    def test_underlying_write_cancellation_is_not_caller_cancellation(self):
        async def _test():
            target = self._target(cancelled_batch_keys={"partition-A"})
            controller = build_flow([AsyncEmitSource(), target]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "partition-A"))

            with pytest.raises(BatchWriteCancelledError, match=r"Batch write 0 for physical key 'partition-A'"):
                await target.flush("endpoint-A")
            with pytest.raises(BatchWriteCancelledError):
                await controller.terminate(wait=True)
            assert target.terminate_called

        asyncio.run(_test())

    def test_termination_waits_for_active_keyed_flush(self):
        async def _test():
            target = self._target(blocked_batch_keys={"partition-A"})
            controller = build_flow([AsyncEmitSource(), target]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "partition-A"))

            flush_task = asyncio.create_task(target.flush("endpoint-A"))
            await target.emit_started_event("partition-A").wait()
            termination_task = asyncio.create_task(controller.terminate(wait=True))
            await _assert_pending(termination_task)

            target.release("partition-A")
            await asyncio.gather(flush_task, termination_task)
            assert target.emit_count_by_key == {"partition-A": 1}
            assert [event.body["v"] for _, events in target.emitted_batch_events for event in events] == [1]

        asyncio.run(_test())

    def test_termination_waits_for_active_write_after_another_batch_fails(self):
        async def _test():
            target = self._target(
                blocked_batch_keys={"partition-A"},
                failed_batch_keys={"partition-B"},
            )
            controller = build_flow([AsyncEmitSource(), target]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "partition-A"))
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(2, "endpoint-B", "partition-B"))

            flush_a = asyncio.create_task(target.flush("endpoint-A"))
            await target.emit_started_event("partition-A").wait()
            termination_task = asyncio.create_task(controller.terminate(wait=True))
            await target.emit_finished_event("partition-B").wait()
            await _assert_pending(termination_task)

            target.release("partition-A")
            await flush_a
            with pytest.raises(RuntimeError, match="emit failed for partition-B"):
                await termination_task
            assert target.emit_count_by_key == {"partition-A": 1, "partition-B": 1}

        asyncio.run(_test())

    def test_termination_drains_all_batches_and_aggregates_failures(self):
        async def _test():
            target = self._target(
                errors_by_batch_key={
                    "partition-A": RuntimeError("write A failed"),
                    "partition-B": ValueError("write B failed"),
                },
                terminate_error=OSError("cleanup failed"),
            )
            controller = build_flow([AsyncEmitSource(), target]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "partition-A"))
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(2, "endpoint-B", "partition-B"))

            with pytest.raises(ExceptionGroup) as exc_info:
                await controller.terminate(wait=True)

            assert target.emit_count_by_key == {"partition-A": 1, "partition-B": 1}
            assert target.terminate_called
            errors = exc_info.value.exceptions
            assert [type(error) for error in errors] == [RuntimeError, ValueError, OSError]
            assert [str(error) for error in errors] == [
                "write A failed",
                "write B failed",
                "cleanup failed",
            ]

        asyncio.run(_test())

    def test_termination_forwards_downstream_even_when_writes_failed(self):
        async def _test():
            target = self._target(errors_by_batch_key={"partition-A": RuntimeError("write A failed")})
            probe = TerminationProbe()
            controller = build_flow([AsyncEmitSource(), target, probe]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "partition-A"))

            with pytest.raises(RuntimeError, match="write A failed"):
                await target.flush("endpoint-A")

            with pytest.raises(RuntimeError, match="write A failed"):
                await controller.terminate(wait=True)
            assert target.terminate_called
            assert probe.terminated

        asyncio.run(_test())

    def test_flush_retains_all_failures_for_same_logical_key(self):
        async def _test():
            target = self._target(
                errors_by_batch_key={
                    "partition-A": RuntimeError("write A failed"),
                    "partition-B": ValueError("write B failed"),
                }
            )
            controller = build_flow([AsyncEmitSource(), target]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "partition-A"))
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(2, "endpoint-A", "partition-B"))

            # A flush starts the key's physical batches in arbitrary order, so only the
            # set of reported failures is defined, not their relative order.
            for operation in (lambda: target.flush("endpoint-A"), lambda: controller.terminate(wait=True)):
                with pytest.raises(ExceptionGroup) as exc_info:
                    await operation()
                assert {type(error) for error in exc_info.value.exceptions} == {RuntimeError, ValueError}
                assert sorted(str(error) for error in exc_info.value.exceptions) == [
                    "write A failed",
                    "write B failed",
                ]

        asyncio.run(_test())

    def test_flush_is_rejected_during_and_after_termination(self):
        async def _test():
            target = self._target(blocked_batch_keys={"partition-A"})
            controller = build_flow([AsyncEmitSource(), target]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "partition-A"))

            termination_task = asyncio.create_task(controller.terminate(wait=True))
            await target.emit_started_event("partition-A").wait()
            with pytest.raises(RuntimeError, match="termination has started"):
                await target.flush("endpoint-A")

            target.release("partition-A")
            await termination_task
            with pytest.raises(RuntimeError, match="termination has started"):
                await target.flush("endpoint-A")

        asyncio.run(_test())

    def test_termination_cancellation_waits_for_write_cleanup_then_propagates(self):
        async def _test():
            target = self._target(blocked_batch_keys={"partition-A"})
            probe = TerminationProbe()
            controller = build_flow([AsyncEmitSource(), target, probe]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "partition-A"))

            termination_task = asyncio.create_task(controller.terminate(wait=True))
            await target.emit_started_event("partition-A").wait()
            termination_task.cancel()
            await _assert_pending(termination_task)

            target.release("partition-A")
            with pytest.raises(asyncio.CancelledError):
                await termination_task
            assert target.terminate_called
            assert target.emit_count_by_key == {"partition-A": 1}
            # Downstream steps own resources too, so cancellation must not skip them.
            assert probe.terminated

        asyncio.run(_test())

    def test_termination_emits_batches_buffered_during_a_write(self):
        async def _test():
            target = self._target(blocked_batch_keys={"partition-A"})
            controller = build_flow([AsyncEmitSource(), target]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "partition-A"))

            termination_task = asyncio.create_task(controller.terminate(wait=True))
            await target.emit_started_event("partition-A").wait()

            # Mimic an event accepted while the first write was yielding, which is what
            # the loop in _emit_all exists to catch.
            target._batch["partition-B"].append({"v": 2, "partition": "partition-B"})
            target._batch_events["partition-B"].append(_partitioned_ev(2, "endpoint-B", "partition-B"))
            target._batch_first_event_time["partition-B"] = _dt.now()
            target._batch_last_event_time["partition-B"] = _dt.now()
            target._batch_start_time["partition-B"] = _time.monotonic()
            target._batch_flush_keys["partition-B"].add("endpoint-B")
            target._get_or_create_flush_state("endpoint-B").batch_keys.add("partition-B")

            target.release("partition-A")
            await termination_task

            assert target.emit_count_by_key == {"partition-A": 1, "partition-B": 1}

        asyncio.run(_test())

    def test_repeated_termination_cancellation_still_completes_cleanup(self):
        async def _test():
            target = self._target(blocked_batch_keys={"partition-A"})
            probe = TerminationProbe()
            controller = build_flow([AsyncEmitSource(), target, probe]).run()
            await _emit_and_wait_until_accepted(controller, target, _partitioned_ev(1, "endpoint-A", "partition-A"))

            termination_task = asyncio.create_task(controller.terminate(wait=True))
            await target.emit_started_event("partition-A").wait()
            for _ in range(3):
                termination_task.cancel()
                await _assert_pending(termination_task)

            target.release("partition-A")
            with pytest.raises(asyncio.CancelledError):
                await termination_task
            assert target.terminate_called
            assert target.emit_count_by_key == {"partition-A": 1}
            assert probe.terminated

        asyncio.run(_test())

    def test_flush_key_extraction_errors_name_the_configured_field(self):
        target = self._target()
        with pytest.raises(ValueError, match=r"flush_key_field '\$key' resolved to None"):
            target._event_flush_key(_partitioned_ev(1, None, "partition-A"))
        with pytest.raises(TypeError, match=r"flush_key_field '\$key' must be a string or list of strings, got int"):
            target._event_flush_key(_partitioned_ev(1, 7, "partition-A"))

        body_keyed = KeyedGatedTarget(key_field=lambda event: event.body["partition"], _flush_key_field="endpoint")
        with pytest.raises(ValueError, match="flush_key_field 'endpoint' is missing from the body"):
            body_keyed._event_flush_key(_partitioned_ev(1, "endpoint-A", "partition-A"))

    def test_flush_rejects_malformed_keys(self):
        async def _test():
            target = self._target()
            controller = build_flow([AsyncEmitSource(), target]).run()

            with pytest.raises(ValueError, match="flush key cannot be an empty list"):
                await target.flush([])
            with pytest.raises(TypeError, match="flush key must be a string or list of strings, got int"):
                await target.flush(7)

            await controller.terminate(wait=True)

        asyncio.run(_test())

    def test_flush_before_target_start_is_rejected(self):
        async def _test():
            target = self._target()
            with pytest.raises(RuntimeError, match="must be running"):
                await target.flush("endpoint-A")

        asyncio.run(_test())

    def test_flush_is_not_public_on_other_batching_targets(self):
        async def _test():
            target = GatedTarget()
            controller = build_flow([AsyncEmitSource(), target]).run()
            assert not hasattr(target, "flush")
            await controller.terminate(wait=True)

        asyncio.run(_test())

    def test_other_batching_targets_reject_flush_key_field(self):
        with pytest.raises(TypeError, match="only by ParquetTarget"):
            GatedTarget(flush_key_field="$key")


class TestBatchingRaceConditions:
    """Verify timer, event-ingestion, and termination batching races."""

    def test_failed_final_emit_preserves_legacy_termination_behavior(self):
        async def _test():
            target = FailingTerminationTarget()
            controller = build_flow([AsyncEmitSource(), target]).run()
            await controller.emit(_ev(1))

            with pytest.raises(RuntimeError, match="final emit failed"):
                await controller.terminate(wait=True)
            assert not target.terminate_called

        asyncio.run(_test())

    def test_race2_events_deleted_by_concurrent_finally(self):
        """Events arriving during a slow _emit go into a separate list.

        Time  | Run loop (_do)                     | Timer (_sleep_and_emit)
        ------+------------------------------------+------------------------------------
          t0  | emit(ev1) -> append, start timer   | sleeping...
          t1  | idle at _q.get()                   | sleeping...
          t2  | idle                               | wakes -> _emit_batch(None)
          t3  | idle                               |   _batch.pop, _batch_events.pop
          t4  | idle                               |   await _emit(batch1) YIELDS
        ------+------------------------------------+-- -- -- -- -- -- -- -- -- -- -- --
          t5  | emit(ev2) -> _Batching._do         |   (suspended in _emit)
          t6  |   _batch[None].append(data2)       |   (suspended in _emit)
          t7  |   _batch_events[None].append(ev2)  |   FIX: fresh list, not timer's
          t8  | idle at _q.get()                   |   (suspended in _emit)
        ------+------------------------------------+-- -- -- -- -- -- -- -- -- -- -- --
          t9  | idle                               |   _emit returns
          t10 | idle                               |   (timer's batch_events is local)
        ------+------------------------------------+------------------------------------
              |                                    |   ev2 ref SURVIVES in new list
        """

        async def _test():
            gate = asyncio.Event()
            target = GatedTarget(gate=gate, flush_after_seconds=0.05)
            controller = build_flow([AsyncEmitSource(), target]).run()

            # t0: emit first event
            await controller.emit(_ev(1))

            # t2-t4: Wait for timer to fire (flush_after_seconds=0.05) and block on gate
            await asyncio.sleep(0.1)
            assert target.emit_count == 1

            # t5-t7: Event 2 arrives while _emit is blocked.  The run loop
            # picks it up and appends to a fresh _batch_events[None].
            await controller.emit(_ev(2))
            await asyncio.sleep(0)  # yield so run loop processes the queued event

            # t9-t10: Release gate -> timer's _emit completes
            gate.set()
            await asyncio.sleep(0)  # yield to let timer finish

            # Event 2's reference should survive — it's in a separate list
            has_pending_data = None in target._batch and len(target._batch[None]) > 0
            batch_events_count = len(target._batch_events.get(None, []))

            assert has_pending_data, "Event 2's data should be in _batch"
            assert batch_events_count == 1, (
                f"Event 2's reference should be preserved in _batch_events[None], " f"got {batch_events_count}"
            )

            await controller.terminate()
            await controller.await_termination()

        asyncio.run(_test())

    def test_race3_timeout_task_not_cancelled_during_terminate(self):
        """_timeout_task must be stopped before _terminate runs.

        Time  | _do(_termination_obj)              | Timer (_sleep_and_emit)
        ------+------------------------------------+------------------------------------
          t0  | emit(ev1) -> append, start timer   | sleeping for 60s...
          t1  | terminate()                        | sleeping...
          t2  |   _terminating = True              | sleeping...
          t3  |   await _timeout_task              | checks _terminating -> exits loop
          t4  |   _emit_all() -> _emit_batch(None) |
          t5  |     _emit(batch) -> completes      |
          t6  |   _terminate()                     |   FIX: task is done, not alive
              |     closes DB pool / file handles  |
          t7  |   _do_downstream(_termination_obj) |
        """

        async def _test():
            # flush_after_seconds=60 -> timer sleeps a long time
            target = RecordingTarget(flush_after_seconds=60.0)
            controller = build_flow([AsyncEmitSource(), target]).run()

            # t0: emit event, starts timer
            await controller.emit(_ev(1))

            # t1-t7: Terminate — matches drain_callback path
            await controller.terminate()
            await controller.await_termination()

            assert target.terminate_called
            assert not target.timeout_task_alive_during_terminate, (
                "_timeout_task was alive when _terminate ran — "
                "TimescaleDBTarget._terminate closes the connection pool, "
                "but the timer could wake up and try to use it"
            )

        asyncio.run(_test())

    def test_race4_emit_all_misses_new_keys(self):
        """_emit_all must flush keys added during a yielding _emit.

        Time  | _do(_termination_obj)              | _batch keys
        ------+------------------------------------+-----------------------------
          t0  | emit(ev1, key="A") -> append       | {"A": [ev1]}
          t1  | terminate()                        | {"A": [ev1]}
          t2  |   _emit_all()                      |
          t3  |     while _batch: key="A"          | {"A": [ev1]}
          t4  |     _emit_batch("A")               |
          t5  |       _batch.pop("A") -> batch_A   | {}
          t6  |       await _emit(batch_A) YIELDS  | {}
        ------+-- -- -- -- -- -- -- -- -- -- -- -- +-----------------------------
          t7  |   (suspended in _emit)             | {"B": [ev2]}  (injected)
        ------+-- -- -- -- -- -- -- -- -- -- -- -- +-----------------------------
          t8  |       _emit returns                | {"B": [ev2]}
          t9  |     while _batch: key="B"          |   FIX: loop continues
          t10 |     _emit_batch("B") -> completes  | {}
          t11 |   _terminate()                     | {}
        """

        async def _test():
            gate = asyncio.Event()
            # key_field="$key" simulates ParquetTarget's partition-based keying
            target = GatedTarget(gate=gate, key_field="$key")
            controller = build_flow([AsyncEmitSource(), target]).run()

            # t0: event with key "A"
            await controller.emit(_ev(1, key="endpoint_A"))

            # t1-t6: Start termination — _emit_all begins, blocks on gate
            term_task = asyncio.ensure_future(controller.terminate())
            await asyncio.sleep(0.05)  # wait for termination to propagate to _emit
            assert target.emit_count == 1

            # t7: While _emit("endpoint_A") is blocked, inject a new partition key.
            # In production, this happens when the event loop processes a queued
            # event for a different endpoint during the S3 write yield.
            target._batch["endpoint_B"].append({"v": 2})
            target._batch_events["endpoint_B"].append(_ev(2, key="endpoint_B"))
            target._batch_first_event_time["endpoint_B"] = _dt.now()
            target._batch_last_event_time["endpoint_B"] = _dt.now()
            target._batch_start_time["endpoint_B"] = _time.monotonic()

            # t8-t11: Release gate
            gate.set()
            await term_task
            await controller.await_termination()

            emitted_values = [item["v"] for batch in target.emitted_batches for item in batch if isinstance(item, dict)]

            assert 2 in emitted_values, (
                f"endpoint_B was never flushed — _emit_all's snapshot missed it. " f"Emitted: {emitted_values}"
            )

        asyncio.run(_test())
