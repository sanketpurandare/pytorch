import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from enum import auto, Enum
from functools import partial, wraps
from typing import Any, Callable, cast, NamedTuple, Optional
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
    """Enumeration of resources used by the simulator."""

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
    sync_id: int = -1


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
        # A new `_Queue` is created per unique `torch.cuda.Stream` and identified by the stream-id
        self.stream_id = stream_id
        # Smaller number indicates higher priority
        self.priority = priority
        self.state = state
        # List of operations dispatched to this stream/queue
        self.ops = ops
        # Each queue maintains a dict from an op's seq_id to a set of synchronization
        # actions that must be taken after the op completes simulated execution.
        self.sync_infos: dict[int, set[_SyncInfo]] = defaultdict(set)
        # Each queue maintains an entry (synchronization action, release_event/seq_id or sync_id)
        # whenever it enters WAITING state. The entry determines the reason for the queue being
        # blocked and provides metadata for the synchronization action such as event record or op
        # completion to notify the queue of their occurence and clear the corresponding entry.
        # Only if this dict is empty, the queue can transition from WAITING to READY state.
        self.wait_sync_infos: dict[tuple[_SyncAction, int], _SyncInfo] = {}
        # Some global synchronization operations such as event/stream/global/work may occur
        # before the stream corresponding to this queue is encountered by the simulator or before
        # any op is dispatched to this queue. We add all of the global synchronization ops to its
        # local directory ahead of all the operations at a special -1 position.
        # The `_init_queue_states` method processes these ops during queue state initialization.
        for sync_info in global_sync_infos:
            self.sync_infos[-1].add(deepcopy(sync_info))

    @property
    def last_op_seq_id(self) -> int:
        return self.ops[-1].seq_id if self.ops else -1


class Simulator:
    """
    Simulator for CUDA stream and synchronization operations.

    This class intercepts CUDA operations (such as stream waits, event recordings,
    and synchronizations) by monkey-patching methods on torch.cuda.Stream, torch.cuda.Event,
    and FakeWork. It simulates asynchronous execution and synchronization behavior.

    Args:
        pg_to_resource (WeakKeyDictionary[ProcessGroup, set[Resource]]): Mapping from process groups
            to the set of resources they use.
    """

    def __init__(self, pg_to_resource:  WeakKeyDictionary[ProcessGroup, set[Resource]]):
        self.streamid_to_queue: dict[int, _Queue] = {}
        self._pg_to_resource = pg_to_resource
        # Simulator pre-processing/capture telemetry
        #   a. seq_id: A unique sequence number assigned to each recorded op following 
        #       the CPU dispatch order
        #   b. work_registry: Mapping from the seq_id of the `FakeWork` object to the 
        #       'op_info' of the non-functional collective that produced it.
        #   c. wait_tensor_registry: Mapping from the `torch.UntypedStorage` object of 
        #       the `AsyncCollectiveTensor` to the 'op_info's of the functional collective 
        #       that produced it.
        #   d. global_sync_infos: Used to store the global sync operations that affect/block 
        #       all the streams. They will used by the streams that will be discovered
        #       later in the execution order by the simulator's record/capture mechanism to
        #       populate their local sync ops.
        #   e. sync_count: A global sync counter is maintained specifically for each `torch.cuda.synchronize`
        #       call. This helps to identify whether all the queues are waiting for the same
        #       sync call.
        #   f. event_wait_ids: 
        #   g. event_record_ids:
        self._seq_id: int = 0
        self._work_registry: dict[int, _OpInfo] = {}
        self._wait_tensor_registry: WeakKeyDictionary[
            torch.UntypedStorage, list[_OpInfo]
        ] = WeakKeyDictionary()
        self._global_sync_infos: set[_SyncInfo] = set()
        self._sync_count: int = 0
        self._event_wait_ids: set[int] = set()
        self._event_record_ids: set[int] = set()
        # Simulation metrics/telemetry explained in `simulate()`
        self._simulate_sync_count: int
        self._simulated_time: float
        self._resource_occupancy: dict[Resource, _Queue]
        self._completed_ops: set[int]
        self._recorded_events: set[int]

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
        resources: tuple[Resource, ...],
        op_time: float,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> _OpInfo:
        stream = torch.cuda.current_stream() if stream is None else stream
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
        # Registers the 'op_info' of the collective operation that produced the tensor 't',
        # with its `torch.UntypedStorage` object.
        sts = get_untyped_storages(t)
        assert len(sts) == 1
        st = sts.pop()
        self._wait_tensor_registry.setdefault(st, list()).append(op_info)

    def _work_wait(self, src_op_info: _OpInfo) -> None:
        # 1. We obtain the seq_id of the last operation enqueued for all the streams,
        #   discovered so far. They act as the `blocking_seq_id` after which the streams
        #   will be blocked.
        # 2. The seq_id of the 'src_op_info' corresponding to the collective op acts as the
        #   `releasing_seq_id`, the completion of which unblocks all the streams.
        # 3. Since, this operation affects/blocks all the streams, it is also recorded as
        #   a global sync event that will be waited on by the streams that will be discovered
        #   in future by the simulator's record/capture mechanism.
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
        """
        Record an operation to be simulated.

        Depending on whether the operation is a collective or compute op, this function records
        the operation and registers any associated wait tensors.
        """
        op_type = _OpType.collective if func in collective_ops else _OpType.compute
        if func not in [
            torch.ops.c10d.monitored_barrier_.default,
            torch.ops._c10d_functional.wait_tensor.default,
        ]:
            if op_type == _OpType.collective:
                # For the collective ops, the resources occupied are determined by the
                # resources of the `ProcessGroup` against which it is issued.
                pg = CollectiveOp.get_process_group(func, args)
                resources = tuple(self._pg_to_resource[pg])
            else:
                resources = (Resource.COMP,)
            # The op is enqueued on the queue corresponding to `torch.cuda.current_stream()`
            op_info = self._create_and_queue_op(
                func=func,
                op_type=op_type,
                resources=resources,
                op_time=op_time,
            )
            if func in non_functional_collectives:
                # All the non-functional collectives return a `FakeWork` object with a unique seq_id
                # under `FakeTensorMode`. The `wait()` method of the `FakeWork` object is used to
                # block the CPU till the completion of the collective operation. We register the 'op_info'
                # of the collective op to the seq_id of 'work' object. It is used by the `work_wait()`
                # to determine the collective operation to wait for.
                work = cast(FakeWork, CollectiveOp.get_work(func, res))
                assert hasattr(work, "seq_id")
                self._work_registry[work.seq_id] = op_info
            if func in functional_collectives:
                # All the functional collectives return one or more `AsyncCollectiveTensor`s.
                # An operation on them invokes a `wait_tensor` op that issues a `wait()` call,
                # to block the CPU till the completion of the collective operation. We register the
                # 'op_info' of the collective op to the `torch.UntypedStorage` of the resultant
                # `AsyncCollectiveTensor`s. It is used by the `wait_tensor()` to determine the
                # collective operation to wait for.
                tree_map_only(
                    torch.Tensor, partial(self._register_wait_tensor, op_info), res
                )

        if func == torch.ops._c10d_functional.wait_tensor.default:
            # We obtain the `torch.UntypedStorage` of the arg tensor for which the `wait_tensor`
            # call is issued. We extract the 'op_infos' of the collective operations that produced
            # the arg tensor from the '_wait_tensor_registry'. The 'work_wait' blocks the CPU
            # till associated collective ops complete.
            input_tensor = args[0]
            assert isinstance(input_tensor, torch.Tensor), (
                "wait_tensor expects a single tensor argument"
            )
            sts = get_untyped_storages(input_tensor)
            assert len(sts) == 1, (
                "Found more than 1 torch.UntypedStorage associated with wait_tensor's arg tensor"
            )
            st = sts.pop()
            src_op_infos = self._wait_tensor_registry.pop(st, None)
            assert src_op_infos is not None, "wait_tensor was not registered"
            for src_op_info in src_op_infos:
                self._work_wait(src_op_info)

    def capture_sync_ops(self) -> None:
        """
        Add wrappers to CUDA synchronization functions to record simulation metadata.
        """
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
            # 1. We obtain the seq_id of the last operation enqueued in 'self_stream', this acts
            #   as the `blocking_seq_id` after which the stream will be blocked till all the
            #   operations enqueued so far'dst_stream' are completed.
            # 2. The seq_id of the last operation enqueued in 'dst_stream' acts as the
            #   `releasing_seq_id`, the completion of which unblocks the 'self_stream'.
            # 3. If no operations are enqueued in 'dst_stream' i.e. when `releasing_seq_id`,
            #   no blocking/unblocking is required and the call can be ignored.
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
            # 1. We obtain the seq_id of the last operation enqueued in 'self_stream', this acts
            #   as the `blocking_seq_id` after which the stream will be blocked till the 'event'
            #   for which wait is issued is recorded.
            # 2. The unique event_id of the 'event' is the `releasing_event_id`, that will unblock
            #   the 'self_stream' after it is recorded.
            # 3. The event_id is also added to the set of `event_wait_ids` for consistency checks.
            sync_op = _SyncOp.wait_event
            event_id = id(event)
            self._event_wait_ids.add(event_id)
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
            # 1. We obtain the seq_id of the last operation enqueued for all the streams,
            #   discovered so far. They act as the `blocking_seq_id` after which the streams
            #   will be blocked till 'self_event' is recorded.
            # 2. The unique event_id of 'self_event' is the `releasing_event_id`, that will
            #   unblock all the streams after it is recorded.
            # 3. Since, this operation affects/blocks all the streams, it is also recorded as
            #   a global sync event that will be waited on by the streams that will be discovered
            #   in future by the simulator's record/capture mechanism.
            # 4. The event_id is also added to the set of `event_wait_ids` for consistency checks.
            sync_op = _SyncOp.event_synchronize
            event_id = id(self_event)
            self._event_wait_ids.add(event_id)
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
            # 1. The unique event_id of 'self_event' is to be recorded on the given 'stream'.
            #   Following the original implementation, `torch.cuda.current_stream()` is used
            #   if no 'stream' is specified.
            # 2. The 'event_record' is enqueued as an op to the given stream since it is to be
            #   executed asynchronously.
            # 3. A synchronization operation is added to the seq_id of the record_event op that
            #   will release any streams waiting for it after the event record op completes.
            # 4. The event_id is also added to the set of `event_record_ids` for consistency checks.
            sync_op = _SyncOp.event_record
            event_id = id(self_event)
            self._event_record_ids.add(event_id)
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
            # 1. We obtain the seq_id of the last operation enqueued in 'stream', this acts
            #   as the `blocking_seq_id` after which the stream will be blocked till 'self_event'
            #   for which wait is issued is recorded. Following the original implementation,
            #   `torch.cuda.current_stream()` is used if no 'stream' is specified.
            # 2. The unique event_id of the 'event' is the `releasing_event_id`, that will unblock
            #   the 'stream' after it is recorded.
            # 3. The event_id is also added to the set of `event_wait_ids` for consistency checks.
            sync_op = _SyncOp.event_wait
            event_id = id(self_event)
            self._event_wait_ids.add(event_id)
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
            # 1. We obtain the seq_id of the last operation enqueued for all the streams,
            #   discovered so far. They act as the `blocking_seq_id` after which the streams
            #   will be blocked till all of them synchronize.
            # 2. A global sync counter is recorded as the sync_id to identify whether all the 
            #   queues are waiting for the same sync call.
            # 3. Since, this operation affects/blocks all the streams, it is also recorded as
            #   a global sync event that will be waited on by the streams that will be discovered
            #   in future by the simulator's record/capture mechanism.
            sync_op = _SyncOp.sync
            self._sync_count += 1
            logger.debug(f"Global {sync_op}")
            for queue in self.streamid_to_queue.values():
                sync_info = _SyncInfo(
                    sync_action=_SyncAction.SYNCHRONIZE_WAIT,
                    sync_id=self._sync_count,
                )
                queue.sync_infos[queue.last_op_seq_id].add(sync_info)
            g_sync_info = _SyncInfo(
                _SyncAction.SYNCHRONIZE_WAIT,
                sync_id=self._sync_count,
            )
            self._global_sync_infos.add(g_sync_info)
            return self._sync_funcs.synchronize(device)

        @wraps(torch.cuda.Stream.synchronize)
        def stream_synchronize(self_stream: torch.cuda.Stream) -> None:
            # 1. We obtain the seq_id of the last operation enqueued for all the streams,
            #   discovered so far. They act as the `blocking_seq_id` after which the streams
            #   will be blocked.
            # 2. The seq_id of the last operation enqueued in 'self_stream' acts as the
            #   `releasing_seq_id`, the completion of which unblocks all the streams.
            # 3. Since, this operation affects/blocks all the streams, it is also recorded as
            #   a global sync event that will be waited on by the streams that will be discovered
            #   in future by the simulator's record/capture mechanism.
            # 4. If no operations are enqueued in 'self_stream' i.e. when `releasing_seq_id = -1`,
            #   no blocking/unblocking is required and the call can be ignored.
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
            # We extract the unique seq_id of the 'self_work' object that calls the wait method.
            # We extract the 'op_info' of the collective operation that produced the 'self_work'
            # object from the '_work_registry'. The 'work_wait' blocks the CPU
            # till associated collective op completes.
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
        """
        Restore the original CUDA synchronization functions.
        """
        torch.cuda.Stream.wait_stream = self._sync_funcs.wait_stream  # type: ignore[method-assign]
        torch.cuda.Stream.wait_event = self._sync_funcs.wait_event  # type: ignore[method-assign]
        torch.cuda.Event.record = self._sync_funcs.event_record  # type: ignore[method-assign]
        torch.cuda.Event.wait = self._sync_funcs.event_wait  # type: ignore[method-assign]
        torch.cuda.Event.synchronize = self._sync_funcs.event_synchronize  # type: ignore[method-assign]
        torch.cuda.synchronize = self._sync_funcs.synchronize  # type: ignore[method-assign]
        torch.cuda.Stream.synchronize = self._sync_funcs.stream_synchronize  # type: ignore[method-assign]
        FakeWork.wait = self._sync_funcs.work_wait  # type: ignore[method-assign]

    def _init_queue_states(self) -> None:
        # Before we begin simulation, the queues are initialized to READY state.
        # However, there might be synchronization operations that the queue must
        # wait for even before it begins execution.
        for queue in self.streamid_to_queue.values():
            if sync_infos := queue.sync_infos.pop(-1, None):
                for sync_info in sync_infos:
                    if sync_info.sync_action == _SyncAction.STREAM_WAIT:
                        queue.wait_sync_infos[
                            (sync_info.sync_action, sync_info.release_seq_id)
                        ] = sync_info
                    elif sync_info.sync_action == _SyncAction.EVENT_WAIT:
                        queue.wait_sync_infos[
                            (sync_info.sync_action, sync_info.release_event_id)
                        ] = sync_info
                    elif sync_info.sync_action == _SyncAction.WORK_WAIT:
                        queue.wait_sync_infos[
                            (sync_info.sync_action, sync_info.release_seq_id)
                        ] = sync_info
                    elif sync_info.sync_action == _SyncAction.SYNCHRONIZE_WAIT:
                        queue.wait_sync_infos[
                            (sync_info.sync_action, sync_info.sync_id)
                        ] = sync_info
                    else:
                        raise ValueError(
                            f"_SyncAction {sync_info.sync_action} should not be present with key -1."
                        )
                queue.state = _State.WAITING

    def _maybe_update_queue_states(self) -> None:
        # 1. First we test if all queues are in WAITING state. This is only true when
        #   a global synchrnozation like torch.cuda.synchronize() is called.
        #   If that is not the case, then we error out, since it is a deadlock
        # 2. Only if we find that all queues are waiting on `_SyncAction.SYNCHRONIZE_WAIT`,
        #   we consider the state legit and clear the wait metadata.
        # 3. We loop over this constraint till at least one queue enters READY state
        while all(
            queue.state == _State.WAITING for queue in self.streamid_to_queue.values()
        ):
            self._simulate_sync_count += 1
            assert self._simulate_sync_count <= self._sync_count, (
                "All the torch.cuda.synchronize() calls have already been processed,"
                " yet all the queues are in WAITING state."
                " This is certainly a deadlock."
            )
            for queue in self.streamid_to_queue.values():
                sync_info = queue.wait_sync_infos.get(
                    (_SyncAction.SYNCHRONIZE_WAIT, self._simulate_sync_count)
                )
                assert sync_info, (
                    "A torch.cuda.synchronize() call should exist "
                    "since all queues are in WAITING state."
                    " This is certainly a deadlock."
                )

                # Remove one SYNCHRONIZE_WAIT op per queue
                queue.wait_sync_infos.pop(
                    (_SyncAction.SYNCHRONIZE_WAIT, self._simulate_sync_count)
                )

                # If wait list is now empty, mark queue READY
                if not queue.wait_sync_infos:
                    queue.state = _State.READY
        # 4. We mark a queue as READY if it is in WAITING state but all its
        #   synchronization operations have been completed.
        # 5. We mark a queue as COMPLETED if it satisfies all of the four conditions:
        #   a. The queue is in READY state
        #   b. The queue has exhausted all the operations
        #   c. The queue has no synchronization operations pending
        #   d. The queue is not incorrectly waiting on a synchronization operation

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
        # 1. We first update the state of all queues and try to qualify the queues
        #   to READY or COMPLETE states whenever possible.
        # 2. Only when all the queues have completed, we return True.
        self._maybe_update_queue_states()
        return all(
            queue.state == _State.COMPLETE for queue in self.streamid_to_queue.values()
        )

    def _occupy_resources(self) -> None:
        # 1. We first obtain the ready queues. We make sure that they have atleast one op
        #   left to execute.
        # 2. We sort the ready queues, first by priority then by the the sequence number of the
        #   head op. Lower number indicates higher priority in both the cases.
        # 3. resource_occupancy tracks the resourced occupied by already RUNNING queues.
        # 4. Each ready queue (in sorted order), checks if the resources required by its head op
        #   are available. If yes, the resources are occupied and the queue's state is updated
        #   to RUNNNING. Else, we move to the next ready queue.

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
            if any(res in self._resource_occupancy for res in head_op.resources):
                continue  # Skip if any resource is already occupied
            # Occupy all required resources
            self._resource_occupancy.update({res: queue for res in head_op.resources})
            queue.state = _State.RUNNING

    def simulate(self) -> float:
        """
        Run the simulation until all operations are completed.

        Returns:
            float: The total simulated time.
        """
        # 1. We first check if all the events that are to be waited on are actually recorded
        #   during capture phase. Error is thrown if missing event ids are found, since it will
        #   result in a deadlock/queue being indefinitely blocked.
        unrecorded_event_ids = set.difference(
            self._event_wait_ids, self._event_record_ids
        )
        assert not unrecorded_event_ids, (
            "Found event ids that are waited on, but not recorded."
            " This will definitely lead to deadlock/queue being blocked indefinitely."
        )
        # 2. Simulation metrics/telemetry initialization
        #   a. resource_occupancy tracks the resources occupied by the RUNNING queues
        #   b. simulation_sync_count tracks the number of global sync calls
        #   c. resource_occupancy tracks the resourced occupied by already RUNNING queues
        #   d. completed_ops records the seq_ids as the operations complete
        #   e. recorded_events tracks the event_ids as they are recorded
        self._simulated_time = 0.0
        self._simulate_sync_count = 0
        self._resource_occupancy = {}
        self._completed_ops = set()
        self._recorded_events = set()
        # 3. Initialize the queue states, some queues may need to WAITING state on synchronization ops
        #   even before they begin execution.
        self._init_queue_states()
        # 4. Simulate until all queues enter COMPLETE state.
        while not self._all_queues_completed():
            # a. READY queues occupy resources that are not occupied by RUNNING queues
            self._occupy_resources()
            # b. Obtain the head-ops from the resource-independent RUNNING queues. The minimum remaining
            #   time of an op represents the time-step of the simulator. This ensures that at-least one op
            #   completes execution in every iteration. Simulation time is incremented by time-step.
            resource_independent_queues = set(self._resource_occupancy.values())
            head_ops = {queue.ops[0] for queue in resource_independent_queues}
            min_rem_time_op = min(head_ops, key=lambda op: op.rem_time)
            time_step = min_rem_time_op.rem_time
            self._simulated_time += time_step
            for head_op in head_ops:
                # c. Subtract the time-step from each op's remaining time
                head_op.rem_time -= time_step
                if head_op.rem_time == 0.0:
                    # d. For every completed op, we remove it from its correspnding queue.
                    #   Change the queue's state to READY. Free-up any resources used by the op.
                    #   Add the op's seq_id to completed ops set.
                    queue = self.streamid_to_queue[head_op.stream_id]
                    queue.ops.pop(0)
                    queue.state = _State.READY

                    for res in head_op.resources:
                        self._resource_occupancy.pop(res)

                    self._completed_ops.add(head_op.seq_id)
                    # e. Obtain any synchronization operations that must be executed at this op's completion.
                    #   Every WAITING queue maintains a waiting metadata dictionary that has information
                    #   about the waiting reason (EVENT/STREAM/WORK) and the associated release_event/seq_id
                    #   that will release them from their WAITING STATE.
                    #   Based on the synchronization action associated with the ops, we do the following:
                    sync_infos = queue.sync_infos.pop(head_op.seq_id, set())
                    for sync_info in sync_infos:
                        # i) EVENT_RELEASE: Find queues are waiting (EVENT_WAIT) on this event_id and,
                        #  clear the corresponding wait metadata. It may then be cleared to proceed by
                        #   the `_maybe_update_queue_states` check.
                        if sync_info.sync_action == _SyncAction.EVENT_RELEASE:
                            self._recorded_events.add(sync_info.release_event_id)
                            for ew_queue in self.streamid_to_queue.values():
                                if (
                                    ew_queue.state == _State.WAITING
                                    and ew_queue.wait_sync_infos.get(
                                        (
                                            _SyncAction.EVENT_WAIT,
                                            sync_info.release_event_id,
                                        )
                                    )
                                ):
                                    ew_queue.wait_sync_infos.pop(
                                        (
                                            _SyncAction.EVENT_WAIT,
                                            sync_info.release_event_id,
                                        )
                                    )
                        # ii) EVENT_WAIT: If the queue needs to wait on a event, check if that event_id
                        #   has been recorded else enter the WAITING state on the event_id.
                        elif sync_info.sync_action == _SyncAction.EVENT_WAIT:
                            if sync_info.release_event_id not in self._recorded_events:
                                queue.wait_sync_infos[
                                    (_SyncAction.EVENT_WAIT, sync_info.release_event_id)
                                ] = sync_info
                                queue.state = _State.WAITING
                        # iii) STREAM_RELEASE: Find queues are waiting (STREAM_WAIT) on this op's seq_id and,
                        #  clear the corresponding wait metadata. It may then be cleared to proceed by
                        #   the `_maybe_update_queue_states` check.
                        elif sync_info.sync_action == _SyncAction.STREAM_RELEASE:
                            for sw_queue in self.streamid_to_queue.values():
                                if (
                                    sw_queue.state == _State.WAITING
                                    and sw_queue.wait_sync_infos.get(
                                        (
                                            _SyncAction.STREAM_WAIT,
                                            sync_info.release_seq_id,
                                        )
                                    )
                                ):
                                    sw_queue.wait_sync_infos.pop(
                                        (
                                            _SyncAction.STREAM_WAIT,
                                            sync_info.release_seq_id,
                                        )
                                    )
                        # iv) STREAM_WAIT: If the queue needs to wait on an op's completion, check if that
                        #   op's seq_id has been completed else enter the WAITING state on the seq_id.
                        elif sync_info.sync_action == _SyncAction.STREAM_WAIT:
                            if sync_info.release_seq_id not in self._completed_ops:
                                queue.wait_sync_infos[
                                    (_SyncAction.STREAM_WAIT, sync_info.release_seq_id)
                                ] = sync_info
                                queue.state = _State.WAITING
                        # v) WORK_RELEASE: Find queues are waiting (WORK_WAIT) on this op's seq_id and,
                        #  clear the corresponding wait metadata. It may then be cleared to proceed by
                        #   the `_maybe_update_queue_states` check.
                        elif sync_info.sync_action == _SyncAction.WORK_RELEASE:
                            for ww_queue in self.streamid_to_queue.values():
                                if (
                                    ww_queue.state == _State.WAITING
                                    and ww_queue.wait_sync_infos.get(
                                        (
                                            _SyncAction.WORK_WAIT,
                                            sync_info.release_seq_id,
                                        )
                                    )
                                ):
                                    ww_queue.wait_sync_infos.pop(
                                        (
                                            _SyncAction.WORK_WAIT,
                                            sync_info.release_seq_id,
                                        )
                                    )
                        # vi) WORK_WAIT: If the queue needs to wait on an op's completion, check if that
                        #   op's seq_id has been completed else enter the WAITING state on the seq_id.
                        elif sync_info.sync_action == _SyncAction.WORK_WAIT:
                            if sync_info.release_seq_id not in self._completed_ops:
                                queue.wait_sync_infos[
                                    (_SyncAction.WORK_WAIT, sync_info.release_seq_id)
                                ] = sync_info
                                queue.state = _State.WAITING
                        # vii) SYNCHRONIZE_WAIT: The queue must enter WAITING state for global synchronization.
                        #   The sync_id corresponds the global sync count recorded during capture phase.
                        #   Once all the queues are waiting on the SYNCHRONIZE_WAIT with identical sync_id,
                        #   they may be cleared to proceed by the `_maybe_update_queue_states` check.
                        elif sync_info.sync_action == _SyncAction.SYNCHRONIZE_WAIT:
                            queue.wait_sync_infos[
                                (_SyncAction.SYNCHRONIZE_WAIT, sync_info.sync_id)
                            ] = sync_info
                            queue.state = _State.WAITING

                        else:
                            raise ValueError(
                                f"Unknown _SyncAction: {sync_info.sync_action}"
                            )

        return self._simulated_time
