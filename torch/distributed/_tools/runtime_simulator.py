import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from enum import auto, Enum
from functools import partial, wraps
from typing import Any, Callable, cast, NamedTuple
from weakref import WeakKeyDictionary

import torch
from torch._C._distributed_c10d import FakeWork, ProcessGroup
from torch.distributed._tools.common_utils import get_untyped_storages
from torch.distributed._tools.fake_collectives import (
    collective_ops,
    CollectiveOp,
    functional_collectives,
    non_functional_collectives,
)
from torch.utils._pytree import tree_map_only


# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level to INFO
logger.setLevel(logging.DEBUG)


class Resource(Enum):
    INTRA_COMM = auto()
    INTER_COMM = auto()
    COMP = auto()
    HOST_TO_MEM = auto()
    MEM_TO_HOST = auto()


class _State(Enum):
    WAITING = auto()
    RUNNING = auto()
    READY = auto()
    COMPLETE = auto()


class _SyncOp(str, Enum):
    event_record = "event_record"
    event_wait = "event_wait"
    event_synchronize = "event_sync"
    wait_event = "wait_event"
    wait_stream = "wait_stream"
    stream_sync = "stream_sync"
    work_wait = "work_wait"
    sync = "sync"


class _SyncFunctions(NamedTuple):
    wait_stream: Callable
    wait_event: Callable
    event_record: Callable
    event_wait: Callable
    event_synchronize: Callable
    synchronize: Callable
    stream_synchronize: Callable
    work_wait: Callable


class _OpType(Enum):
    compute = auto()
    collective = auto()
    sync = auto()


class _SyncAction(Enum):
    STREAM_WAIT = auto()
    EVENT_WAIT = auto()
    SYNCHRONIZE_WAIT = auto()
    STREAM_RELEASE = auto()
    EVENT_RELEASE = auto()
    WORK_WAIT = auto()
    WORK_RELEASE = auto()


@dataclass
class _SyncInfo:
    sync_action: _SyncAction
    release_seq_id: int = -1
    release_event_id: int = -1


@dataclass
class _OpInfo:
    seq_id: int
    func: str
    type: _OpType
    stream_id: int
    resources: tuple[Resource, ...]
    run_time: float
    rem_time: float


class _Queue:
    def __init__(
        self,
        stream_id: int,
        priority: int,
        state: _State,
        global_sync_infos: set[_SyncInfo],
        ops: list[_OpInfo],
    ) -> None:
        self.stream_id = stream_id
        self.priority = priority
        self.state = state
        self.ops = ops
        self.sync_infos: dict[int, set[_SyncInfo]] = defaultdict(set)
        self.wait_sync_infos: dict[_SyncAction, set[_SyncInfo]] = defaultdict(set)
        for sync_info in global_sync_infos:
            self.sync_infos[-1].add(deepcopy(sync_info))

    @property
    def last_op_seq_id(self) -> int:
        return self.ops[-1].seq_id if self.ops else -1


class Simulator:
    def __init__(self, pg_to_resource: dict[ProcessGroup, set[Resource]]):
        self.streamid_to_queue: dict[int, _Queue] = {}
        self._pg_to_resource = pg_to_resource
        self._seq_id = 0
        self._work_registry: dict[int, _OpInfo] = {}
        self._wait_tensor_registry: WeakKeyDictionary[
            torch.UntypedStorage, list[_OpInfo]
        ] = WeakKeyDictionary()
        self._global_sync_infos: set[_SyncInfo] = set()

    def _get_or_create_queue(self, stream: torch.cuda.Stream) -> _Queue:
        queue = self.streamid_to_queue.setdefault(
            stream.stream_id,
            _Queue(
                stream_id=stream.stream_id,
                priority=stream.priority,
                state=_State.READY,
                global_sync_infos=self._global_sync_infos,
                ops=list(),
            ),
        )
        return queue

    def _create_and_queue_op(
        self,
        func: Any,
        op_type: _OpType,
        resources: tuple[Resource],
        op_time: float,
        stream=torch.cuda.current_stream(),
    ) -> _OpInfo:
        queue = self._get_or_create_queue(stream)
        op_info = _OpInfo(
            self._seq_id,
            str(func),
            op_type,
            stream.stream_id,
            resources,
            op_time,
            op_time,
        )
        queue.ops.append(op_info)
        self._seq_id += 1
        return op_info

    def _register_wait_tensor(self, op_info: _OpInfo, t: torch.Tensor) -> None:
        sts = get_untyped_storages(t)
        assert len(sts) == 1
        st = sts.pop()
        self._wait_tensor_registry.setdefault(st, list()).append(op_info)

    def _work_wait(self, src_op_info: _OpInfo) -> None:
        sync_op = _SyncFunctions.work_wait
        dst_queue = self.streamid_to_queue[src_op_info.stream_id]
        logger.debug(f"{sync_op} for {src_op_info.seq_id}")
        assert src_op_info.seq_id != -1
        for queue in self.streamid_to_queue.values():
            sync_info = _SyncInfo(
                sync_action=_SyncAction.WORK_WAIT, release_seq_id=src_op_info.seq_id
            )
            queue.sync_infos[queue.last_op_seq_id].add(sync_info)
        g_sync_info = _SyncInfo(
            _SyncAction.WORK_WAIT, release_seq_id=src_op_info.seq_id
        )
        self._global_sync_infos.add(g_sync_info)
        dst_sync_info = _SyncInfo(
            _SyncAction.WORK_RELEASE, release_seq_id=src_op_info.seq_id
        )
        dst_queue.sync_infos[src_op_info.seq_id].add(dst_sync_info)

    def record_op(self, func, args, kwargs, res, op_time: float) -> None:
        op_type = _OpType.collective if func in collective_ops else _OpType.compute
        if func not in [
            torch.ops.c10d.monitored_barrier_.default,
            torch.ops._c10d_functional.wait_tensor.default,
        ]:
            if op_type == _OpType.collective:
                pg = CollectiveOp.get_process_group(func, args)
                resources = tuple(self._pg_to_resource[pg])
            else:
                resources = (Resource.COMP,)

            op_info = self._create_and_queue_op(
                func=func,
                op_type=op_type,
                resources=resources,
                op_time=op_time,
            )
            if func in non_functional_collectives:
                work = cast(FakeWork, CollectiveOp.get_work(func, res))
                assert hasattr(work, "seq_id")
                self._work_registry[work.seq_id] = op_info
            if func in functional_collectives:
                tree_map_only(
                    torch.Tensor, partial(self._register_wait_tensor, op_info), res
                )

        if func == torch.ops._c10d_functional.wait_tensor.default:
            input_tensor = args[0]
            assert isinstance(input_tensor, torch.Tensor)
            sts = get_untyped_storages(input_tensor)
            assert len(sts) == 1
            st = sts.pop()
            src_op_infos = self._wait_tensor_registry.pop(st, None)
            assert src_op_infos is not None, "wait_tensor was not registered"
            for src_op_info in src_op_infos:
                self._work_wait(src_op_info)

    def capture_sync_ops(self) -> None:
        self._sync_funcs = _SyncFunctions(
            wait_stream=torch.cuda.Stream.wait_stream,
            wait_event=torch.cuda.Stream.wait_event,
            event_record=torch.cuda.Event.record,
            event_wait=torch.cuda.Event.wait,
            event_synchronize=torch.cuda.Event.synchronize,
            synchronize=torch.cuda.synchronize,
            stream_synchronize=torch.cuda.Stream.synchronize,
            work_wait=FakeWork.wait,
        )

        @wraps(torch.cuda.Stream.wait_stream)
        def wait_stream(
            self_stream: torch.cuda.Stream, dst_stream: torch.cuda.Stream
        ) -> None:
            sync_op = _SyncOp.wait_stream
            logger.debug(
                f"{sync_op} Self Stream {self_stream.stream_id} Dst Stream {dst_stream.stream_id}"
            )
            dst_queue = self._get_or_create_queue(dst_stream)
            src_queue = self._get_or_create_queue(self_stream)

            if dst_queue.last_op_seq_id != -1:
                src_sync_info = _SyncInfo(
                    sync_action=_SyncAction.STREAM_WAIT,
                    release_seq_id=dst_queue.last_op_seq_id,
                )
                dst_sync_info = _SyncInfo(
                    sync_action=_SyncAction.STREAM_RELEASE,
                    release_seq_id=dst_queue.last_op_seq_id,
                )
                src_queue.sync_infos[src_queue.last_op_seq_id].add(src_sync_info)
                dst_queue.sync_infos[dst_queue.last_op_seq_id].add(dst_sync_info)

            return self._sync_funcs.wait_stream(self_stream, dst_stream)

        @wraps(torch.cuda.Stream.wait_event)
        def wait_event(self_stream: torch.cuda.Stream, event: torch.cuda.Event) -> None:
            sync_op = _SyncOp.wait_event
            event_id = id(event)
            logger.debug(
                f"{sync_op} Self Stream {self_stream.stream_id} Event {event_id}"
            )
            src_queue = self._get_or_create_queue(self_stream)

            src_sync_info = _SyncInfo(
                sync_action=_SyncAction.EVENT_WAIT,
                release_event_id=event_id,
            )
            src_queue.sync_infos[src_queue.last_op_seq_id].add(src_sync_info)
            return self._sync_funcs.wait_event(self_stream, event)

        @wraps(torch.cuda.Event.synchronize)
        def event_synchronize(self_event: torch.cuda.Event) -> None:
            sync_op = _SyncOp.event_synchronize
            event_id = id(self_event)
            logger.debug(f"{sync_op} Event {event_id}")

            for queue in self.streamid_to_queue.values():
                sync_info = _SyncInfo(
                    sync_action=_SyncAction.EVENT_WAIT,
                    release_event_id=event_id,
                )
                queue.sync_infos[queue.last_op_seq_id].add(sync_info)
            g_sync_info = _SyncInfo(
                sync_action=_SyncAction.EVENT_WAIT, release_event_id=event_id
            )
            self._global_sync_infos.add(g_sync_info)
            return self._sync_funcs.event_synchronize(self_event)

        @wraps(torch.cuda.Event.record)
        def event_record(
            self_event: torch.cuda.Event, stream: torch.cuda.Stream = None
        ) -> None:
            sync_op = _SyncOp.event_record
            event_id = id(self_event)
            if stream is None:
                stream = torch.cuda.current_stream()
            logger.debug(f"{sync_op} Event {event_id} Stream {stream.stream_id}")
            event_op = self._create_and_queue_op(
                func=sync_op.value,
                op_type=_OpType.sync,
                resources=(Resource.COMP,),
                op_time=0.0,
                stream=stream,
            )
            src_queue = self.streamid_to_queue[stream.stream_id]

            event_sync_info = _SyncInfo(
                sync_action=_SyncAction.EVENT_RELEASE,
                release_event_id=event_id,
            )
            src_queue.sync_infos[event_op.seq_id].add(event_sync_info)
            return self._sync_funcs.event_record(self_event, stream)

        @wraps(torch.cuda.Event.wait)
        def event_wait(
            self_event: torch.cuda.Event, stream: torch.cuda.Stream = None
        ) -> None:
            sync_op = _SyncOp.event_wait
            event_id = id(self_event)
            if stream is None:
                stream = torch.cuda.current_stream()
            logger.debug(f"{sync_op} Event {event_id} Stream {stream.stream_id}")
            src_queue = self._get_or_create_queue(stream)

            src_sync_info = _SyncInfo(
                sync_action=_SyncAction.EVENT_WAIT,
                release_event_id=event_id,
            )
            src_queue.sync_infos[src_queue.last_op_seq_id].add(src_sync_info)
            return self._sync_funcs.event_wait(self_event, stream)

        @wraps(torch.cuda.synchronize)
        def synchronize(device: torch.cuda.device = None) -> None:
            sync_op = _SyncOp.sync
            logger.debug(f"Global {sync_op}")
            for queue in self.streamid_to_queue.values():
                sync_info = _SyncInfo(
                    sync_action=_SyncAction.SYNCHRONIZE_WAIT,
                )
                queue.sync_infos[queue.last_op_seq_id].add(sync_info)
            g_sync_info = _SyncInfo(
                _SyncAction.SYNCHRONIZE_WAIT,
            )
            self._global_sync_infos.add(g_sync_info)
            return self._sync_funcs.synchronize(device)

        @wraps(torch.cuda.Stream.synchronize)
        def stream_synchronize(self_stream: torch.cuda.Stream) -> None:
            sync_op = _SyncOp.stream_sync
            logger.debug(f"{sync_op} Stream {self_stream.stream_id}")
            dst_queue = self._get_or_create_queue(self_stream)
            if dst_queue.last_op_seq_id != -1:
                for queue in self.streamid_to_queue.values():
                    sync_info = _SyncInfo(
                        sync_action=_SyncAction.STREAM_WAIT,
                        release_seq_id=dst_queue.last_op_seq_id,
                    )
                    queue.sync_infos[queue.last_op_seq_id].add(sync_info)
                g_sync_info = _SyncInfo(
                    _SyncAction.STREAM_WAIT, release_seq_id=dst_queue.last_op_seq_id
                )
                self._global_sync_infos.add(g_sync_info)
                dst_sync_info = _SyncInfo(
                    sync_action=_SyncAction.STREAM_RELEASE,
                    release_seq_id=dst_queue.last_op_seq_id,
                )
                dst_queue.sync_infos[dst_queue.last_op_seq_id].add(dst_sync_info)
            return self._sync_funcs.stream_synchronize(self_stream)

        @wraps(FakeWork.wait)
        def work_wait(self_work: FakeWork, timeout: timedelta = ...) -> bool:
            assert isinstance(self_work, FakeWork)
            assert hasattr(self_work, "seq_id")
            src_op_info = self._work_registry.pop(self_work.seq_id, None)
            assert src_op_info is not None, f"Work is not registered {self_work.seq_id}"
            self._work_wait(src_op_info)
            return self._sync_funcs.work_wait(self_work, timeout)

        torch.cuda.Stream.wait_stream = wait_stream  # type: ignore[method-assign]
        torch.cuda.Stream.wait_event = wait_event  # type: ignore[method-assign]
        torch.cuda.Event.record = event_record  # type: ignore[method-assign]
        torch.cuda.Event.wait = event_wait  # type: ignore[method-assign]
        torch.cuda.Event.synchronize = event_synchronize  # type: ignore[method-assign]
        torch.cuda.synchronize = synchronize  # type: ignore[method-assign]
        torch.cuda.Stream.synchronize = stream_synchronize  # type: ignore[method-assign]
        FakeWork.wait = work_wait  # type: ignore[method-assign]

    def restore_sync_ops(self) -> None:
        torch.cuda.Stream.wait_stream = self._sync_funcs.wait_stream  # type: ignore[method-assign]
        torch.cuda.Stream.wait_event = self._sync_funcs.wait_event  # type: ignore[method-assign]
        torch.cuda.Event.record = self._sync_funcs.event_record  # type: ignore[method-assign]
        torch.cuda.Event.wait = self._sync_funcs.event_wait  # type: ignore[method-assign]
        torch.cuda.Event.synchronize = self._sync_funcs.event_synchronize  # type: ignore[method-assign]
        torch.cuda.synchronize = self._sync_funcs.synchronize  # type: ignore[method-assign]
        torch.cuda.Stream.synchronize = self._sync_funcs.stream_synchronize  # type: ignore[method-assign]
        FakeWork.wait = self._sync_funcs.work_wait  # type: ignore[method-assign]

    def _init_queue_states(self) -> None:
        for queue in self.streamid_to_queue.values():
            if sync_infos := queue.sync_infos.pop(-1, None):
                for sync_info in sync_infos:
                    queue.wait_sync_infos[sync_info.sync_action].add(sync_info)
                queue.state = _State.WAITING

    def _clean_up_wait_sync_infos(self) -> None:
        for queue in self.streamid_to_queue.values():
            for sync_action in _SyncAction:
                w_sync_infos = queue.wait_sync_infos.get(sync_action)
                if w_sync_infos is not None and not w_sync_infos:
                    queue.wait_sync_infos.pop(sync_action)

    def _maybe_update_queue_states(self) -> None:
        while all(
            queue.state == _State.WAITING for queue in self.streamid_to_queue.values()
        ):
            for queue in self.streamid_to_queue.values():
                local_sync_infos = queue.wait_sync_infos.get(
                    _SyncAction.SYNCHRONIZE_WAIT
                )
                assert local_sync_infos, (
                    "At least 1 torch.cuda.synchronize() call should exist "
                    "since all queues are in WAITING state."
                    " This could possibly be a deadlock."
                )

                # Remove one SYNCHRONIZE_WAIT op per queue
                local_sync_infos.pop()
                if not local_sync_infos:
                    queue.wait_sync_infos.pop(_SyncAction.SYNCHRONIZE_WAIT)

                # If wait list is now empty, mark queue READY
                if not queue.wait_sync_infos:
                    queue.state = _State.READY

        for queue in self.streamid_to_queue.values():
            if queue.state == _State.WAITING and not queue.wait_sync_infos:
                queue.state = _State.READY

            if (
                queue.state == _State.READY
                and not queue.ops
                and not queue.sync_infos
                and not queue.wait_sync_infos
            ):
                queue.state = _State.COMPLETE

    def _all_queues_completed(self) -> bool:
        self._clean_up_wait_sync_infos()
        self._maybe_update_queue_states()
        return all(
            queue.state == _State.COMPLETE for queue in self.streamid_to_queue.values()
        )

    def _occupy_resources(self, resource_occupancy: dict[Resource, _Queue]) -> None:
        ready_queues = [
            queue
            for queue in self.streamid_to_queue.values()
            if queue.state == _State.READY
        ]
        assert all(queue.ops for queue in ready_queues), (
            "Assertion failed: Some queues in ready_queues have no remaining operations (ops=[])"
        )
        ready_queues.sort(key=lambda q: (q.priority, q.ops[0].seq_id))
        for queue in ready_queues:
            head_op = queue.ops[0]
            if any(res in resource_occupancy for res in head_op.resources):
                continue  # Skip if any resource is already occupied
            # Occupy all required resources
            resource_occupancy.update({res: queue for res in head_op.resources})
            queue.state = _State.RUNNING

    def simulate(self) -> float:
        simulated_time = 0.0
        resource_occupancy: dict[Resource, _Queue] = {}
        completed_ops: set[int] = set()
        recorded_events: set[int] = set()
        self._init_queue_states()
        while not self._all_queues_completed():
            self._occupy_resources(resource_occupancy)
            resource_independent_queues = set(resource_occupancy.values())
            head_ops = {queue.ops[0] for queue in resource_independent_queues}
            min_rem_time_op = min(head_ops, key=lambda op: op.rem_time)
            time_quantum = min_rem_time_op.rem_time
            for head_op in head_ops:
                head_op.rem_time -= time_quantum
                if head_op.rem_time == 0.0:
                    queue = self.streamid_to_queue[head_op.stream_id]
                    queue.ops.pop(0)
                    queue.state = _State.READY

                    for res in head_op.resources:
                        resource_occupancy.pop(res)

                    completed_ops.add(head_op.seq_id)

                    for sync_info in queue.sync_infos.pop(head_op.seq_id, []):
                        match sync_info.sync_action:
                            case _SyncAction.EVENT_RELEASE:
                                assert sync_info.release_event_id != -1, (
                                    "Found event record with negative event_id"
                                )
                                recorded_events.add(sync_info.release_event_id)
                                event_waiting_queues = {
                                    ew_queue
                                    for ew_queue in self.streamid_to_queue.values()
                                    if ew_queue.state == _State.WAITING
                                    and ew_queue.wait_sync_infos.get(
                                        _SyncAction.EVENT_WAIT
                                    )
                                }
                                for ew_queue in event_waiting_queues:
                                    event_wait_sync_infos = ew_queue.wait_sync_infos[
                                        _SyncAction.EVENT_WAIT
                                    ]
                                    to_be_removed = set()
                                    for e_sync_info in event_wait_sync_infos:
                                        if (
                                            e_sync_info.release_event_id
                                            == sync_info.release_event_id
                                        ):
                                            to_be_removed.add(e_sync_info)
                                    event_wait_sync_infos.difference_update(
                                        to_be_removed
                                    )

                            case _SyncAction.EVENT_WAIT:
                                if sync_info.release_event_id not in recorded_events:
                                    queue.wait_sync_infos[_SyncAction.EVENT_WAIT].add(
                                        sync_info
                                    )
                                    queue.state = _State.WAITING
                            case _SyncAction.STREAM_RELEASE:
                                stream_waiting_queues = {
                                    sw_queue
                                    for sw_queue in self.streamid_to_queue.values()
                                    if sw_queue.state == _State.WAITING
                                    and sw_queue.wait_sync_infos.get(
                                        _SyncAction.STREAM_WAIT
                                    )
                                }
                                for sw_queue in stream_waiting_queues:
                                    stream_wait_sync_infos = sw_queue.wait_sync_infos[
                                        _SyncAction.STREAM_WAIT
                                    ]
                                    to_be_removed = set()
                                    for s_sync_info in stream_wait_sync_infos:
                                        if (
                                            s_sync_info.release_seq_id
                                            == sync_info.release_seq_id
                                        ):
                                            to_be_removed.add(s_sync_info)
                                    stream_wait_sync_infos.difference_update(
                                        to_be_removed
                                    )
                            case _SyncAction.STREAM_WAIT:
                                if sync_info.release_seq_id not in completed_ops:
                                    queue.wait_sync_infos[_SyncAction.STREAM_WAIT].add(
                                        sync_info
                                    )
                                    queue.state = _State.WAITING

                            case _SyncAction.WORK_RELEASE:
                                work_waiting_queues = {
                                    ww_queue
                                    for ww_queue in self.streamid_to_queue.values()
                                    if ww_queue.state == _State.WAITING
                                    and ww_queue.wait_sync_infos.get(
                                        _SyncAction.WORK_WAIT
                                    )
                                }
                                for ww_queue in work_waiting_queues:
                                    work_wait_sync_infos = ww_queue.wait_sync_infos[
                                        _SyncAction.WORK_WAIT
                                    ]
                                    to_be_removed = set()
                                    for w_sync_info in work_wait_sync_infos:
                                        if (
                                            w_sync_info.release_seq_id
                                            == sync_info.release_seq_id
                                        ):
                                            to_be_removed.add(w_sync_info)
                                    work_wait_sync_infos.difference_update(
                                        to_be_removed
                                    )
                            case _SyncAction.WORK_WAIT:
                                if sync_info.release_seq_id not in completed_ops:
                                    queue.wait_sync_infos[_SyncAction.WORK_WAIT].add(
                                        sync_info
                                    )
                                    queue.state = _State.WAITING
                            case _SyncAction.SYNCHRONIZE_WAIT:
                                queue.wait_sync_infos[_SyncAction.SYNCHRONIZE_WAIT].add(
                                    sync_info
                                )
                                queue.state = _State.WAITING
            simulated_time += time_quantum

        return simulated_time
