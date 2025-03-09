import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import auto, Enum
from functools import wraps, partial
from typing import Any, cast, NamedTuple
from weakref import WeakKeyDictionary

import torch
from torch._C._distributed_c10d import FakeWork, ProcessGroup
from torch.distributed._tools.common_utils import get_untyped_storages
from torch.distributed._tools.fake_collectives import collective_ops, CollectiveOp, functional_collectives, non_functional_collectives
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
    DMA_MEM = auto()

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
    wait_stream: callable
    wait_event: callable
    event_record: callable
    event_wait: callable
    event_synchronize: callable
    synchronize: callable
    stream_synchronize: callable
    work_wait: callable


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
    sync_op: _SyncOp
    blocking_seq_id: int = -1
    release_seq_id: int = -1
    blocking_event_id: int = -1
    release_event_id: int = -1


@dataclass
class _OpInfo:
    seq_id: int
    func: str
    type: _OpType
    stream_id: int
    resource: tuple[Resource]
    run_time: float
    rem_time: float


class _Queue:
    def __init__(
        self,
        stream_id: int,
        priority: int,
        state: _State,
        global_sync_infos: list[_SyncInfo],
        ops: list[_OpInfo] = [],
    ) -> None:
        self.stream_id = stream_id
        self.priority = priority
        self.state = state
        self.ops = ops
        self.sync_infos: dict[int, list[_SyncInfo]] = defaultdict(list)
        self.wait_sync_infos: dict[int, list[_SyncInfo]] = {}
        for sync_info in global_sync_infos:
            self.sync_infos[-1].append(deepcopy(sync_info))


class Simulator:
    def __init__(self, pg_to_resource: dict[ProcessGroup, set[Resource]]):
        self.streamid_to_queue: dict[int, _Queue] = {}
        self.pg_to_resource = pg_to_resource
        self.seq_id = 0
        self.work_registry: dict[int, _OpInfo] = {}
        self.wait_tensor_registry: WeakKeyDictionary[torch.UntypedStorage, list[_OpInfo]] = WeakKeyDictionary()
        self.global_sync_infos: list[_SyncInfo] = []

    def create_and_queue_op(
        self,
        func: Any,
        op_type: _OpType,
        resource: tuple[Resource],
        op_time: float,
        stream_id: int = None,
        stream_priority: int = None,
    ) -> _OpInfo:
        if stream_id is None:
            current_stream = torch.cuda.current_stream()
            stream_id = current_stream.stream_id
            stream_priority = current_stream.priority
        queue = self.streamid_to_queue.setdefault(
            stream_id,
            _Queue(
                stream_id=stream_id,
                priority=stream_priority,
                state=_State.READY,
                global_sync_infos=self.global_sync_infos,
                ops=list(),
            ),
        )
        op_info = _OpInfo(
            self.seq_id, str(func), op_type, stream_id, resource, op_time, op_time
        )
        queue.ops.append(op_info)
        self.seq_id += 1
        return op_info
    
    def register_wait_tensor(self, op_info: _OpInfo, t: torch.Tensor) -> None:
        sts = get_untyped_storages(t)
        assert len(sts) == 1
        st = sts.pop()
        self.wait_tensor_registry.setdefault(st, list()).append(op_info)

    def _work_wait(self, src_op_info: _OpInfo) -> None:
        sync_op = _SyncFunctions.work_wait
        dst_queue = self.streamid_to_queue[src_op_info.stream_id]
        release_seq_id = src_op_info.seq_id
        logger.debug(f"{sync_op} for {release_seq_id}")
        assert release_seq_id != -1
        for queue in self.streamid_to_queue.values():
            blocking_seq_id = queue.ops[-1].seq_id if queue.ops else -1
            sync_info = _SyncInfo(
                sync_action=_SyncAction.WORK_WAIT,
                sync_op=sync_op,
                blocking_seq_id=blocking_seq_id,
                release_seq_id=release_seq_id
            )
            queue.sync_infos[blocking_seq_id].append(sync_info)
        sync_info = _SyncInfo(
            _SyncAction.WORK_WAIT,
            release_seq_id=release_seq_id
        )
        self.global_sync_infos.append(sync_info)
        dst_sync_info = _SyncInfo(
            _SyncAction.WORK_RELEASE,
            release_seq_id=release_seq_id
        )
        dst_queue.sync_infos[release_seq_id].append(dst_sync_info)
    
    def record_op(self, func, args, kwargs, res, op_time: float) -> None:
        
        op_type = _OpType.collective if func in collective_ops else _OpType.compute
        if func not in [
            torch.ops.c10d.monitored_barrier_.default,
            torch.ops._c10d_functional.wait_tensor.default,
        ]:
            if op_type == _OpType.collective:
                pg = CollectiveOp.get_process_group(func, args)
                resource = tuple(self.pg_to_resource[pg])
            else:
                resource = (Resource.COMP,)

            op_info = self.create_and_queue_op(
                func=func,
                op_type=op_type,
                resource=resource,
                op_time=op_time,
            )
            if func in non_functional_collectives:
                work = cast(FakeWork, CollectiveOp.get_work(func, res))
                assert hasattr(work, "seq_id")
                self.work_registry[work.seq_id] = op_info
            if func in functional_collectives:
                tree_map_only(torch.Tensor, partial(self.register_wait_tensor, op_info), res)

        if func == torch.ops._c10d_functional.wait_tensor.default:
            input_tensor = args[0]
            assert isinstance(input_tensor, torch.Tensor)
            sts = get_untyped_storages(input_tensor)
            assert len(sts) == 1
            st = sts.pop()
            src_op_infos = self.wait_tensor_registry.pop(st, None)
            assert src_op_infos is not None, "wait_tensor was not registered"
            for src_op_info in src_op_infos:
                self._work_wait(src_op_info)

    def capture_sync_ops(self):
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

        @wraps(self._sync_funcs.wait_stream)
        def wait_stream(self_stream: torch.cuda.Stream, dst_stream: torch.cuda.Stream):
            sync_op = _SyncOp.wait_stream
            logger.debug(
                f"{sync_op} Self Stream {self_stream.stream_id} Dst Stream {dst_stream.stream_id}"
            )
            dst_queue = self.streamid_to_queue.setdefault(
                dst_stream.stream_id,
                _Queue(
                    stream_id=dst_stream.stream_id,
                    priority=dst_stream.priority,
                    state=_State.READY,
                    global_sync_infos=self.global_sync_infos,
                    ops=list(),
                ),
            )
            src_queue = self.streamid_to_queue.setdefault(
                self_stream.stream_id,
                _Queue(
                    stream_id=self_stream.stream_id,
                    priority=self_stream.priority,
                    state=_State.READY,
                    global_sync_infos=self.global_sync_infos,
                    ops=list(),
                ),
            )
            blocking_seq_id = src_queue.ops[-1].seq_id if src_queue.ops else -1
            release_seq_id = dst_queue.ops[-1].seq_id if dst_queue.ops else -1

            if release_seq_id != -1:
                src_sync_info = _SyncInfo(
                    sync_action=_SyncAction.STREAM_WAIT,
                    sync_op=sync_op,
                    blocking_seq_id=blocking_seq_id,
                    release_seq_id=release_seq_id,
                )
                dst_sync_info = _SyncInfo(
                    sync_action=_SyncAction.STREAM_RELEASE,
                    sync_op=sync_op,
                    blocking_seq_id=blocking_seq_id,
                    release_seq_id=release_seq_id,
                )
                src_queue.sync_infos[blocking_seq_id].append(src_sync_info)
                dst_queue.sync_infos[release_seq_id].append(dst_sync_info)

            return self._sync_funcs.wait_stream(self_stream, dst_stream)

        @wraps(self._sync_funcs.wait_event)
        def wait_event(self_stream: torch.cuda.Stream, event: torch.cuda.Event):
            sync_op = _SyncOp.wait_event
            event_id = id(event)
            logger.debug(
                f"{sync_op} Self Stream {self_stream.stream_id} Event {event_id}"
            )
            src_queue = self.streamid_to_queue.setdefault(
                self_stream.stream_id,
                _Queue(
                    stream_id=self_stream.stream_id,
                    priority=self_stream.priority,
                    state=_State.READY,
                    global_sync_infos=self.global_sync_infos,
                    ops=list(),
                ),
            )
            blocking_seq_id = src_queue.ops[-1].seq_id if src_queue.ops else -1
            
            src_sync_info = _SyncInfo(
                sync_action=_SyncAction.EVENT_WAIT,
                sync_op=sync_op,
                blocking_seq_id=blocking_seq_id,
                release_event_id=event_id,
            )
            src_queue.sync_infos[blocking_seq_id].append(src_sync_info)
            return self._sync_funcs.wait_event(self_stream, event)

        @wraps(self._sync_funcs.event_synchronize)
        def event_synchronize(self_event: torch.cuda.Event):
            sync_op = _SyncOp.event_synchronize
            event_id = id(self_event)
            logger.debug(f"{sync_op} Event {event_id}")
            
            for queue in self.streamid_to_queue.values():
                blocking_seq_id = queue.ops[-1].seq_id if queue.ops else -1
                sync_info = _SyncInfo(
                    sync_action=_SyncAction.EVENT_WAIT,
                    sync_op=sync_op,
                    blocking_seq_id=blocking_seq_id,
                    release_event_id=event_id,
                )
                queue.sync_infos[blocking_seq_id].append(sync_info)
            sync_info = _SyncInfo(
                sync_action=_SyncAction.EVENT_WAIT,
                sync_op=sync_op,
                release_event_id=event_id
            )
            self.global_sync_infos.append(sync_info)
            return self._sync_funcs.event_synchronize(self_event)

        @wraps(self._sync_funcs.event_record)
        def event_record(
            self_event: torch.cuda.Event, stream: torch.cuda.Stream = None
        ):
            sync_op = _SyncOp.event_record
            event_id = id(self_event)
            if stream is None:
                stream = torch.cuda.current_stream()
            logger.debug(
                f"{sync_op} Event {event_id} Stream {stream.stream_id}"
            )
            event_op = self.create_and_queue_op(
                func=sync_op,
                op_type=_OpType.sync,
                resource=Resource.COMP,
                op_time=0.0,
                stream_id=stream.stream_id,
                stream_priority=stream.priority,
            )
            src_queue = self.streamid_to_queue[stream.stream_id]
            
            event_sync_info = _SyncInfo(
                sync_action=_SyncAction.EVENT_RELEASE,
                sync_op=sync_op,
                release_event_id=event_id,
            )
            src_queue.sync_infos[event_op.seq_id].append(event_sync_info)
            return self._sync_funcs.event_record(self_event, stream)

        @wraps(self._sync_funcs.event_wait)
        def event_wait(self_event: torch.cuda.Event, stream: torch.cuda.Stream = None):
            sync_op = _SyncOp.event_wait
            event_id = id(self_event)
            if stream is None:
                stream = torch.cuda.current_stream()
            logger.debug(
                f"{sync_op} Event {event_id} Stream {stream.stream_id}"
            )
            src_queue = self.streamid_to_queue.setdefault(
                stream.stream_id,
                _Queue(
                    stream_id=stream.stream_id,
                    priority=stream.priority,
                    state=_State.READY,
                    global_sync_infos=self.global_sync_infos,
                    ops=list(),
                ),
            )
            blocking_seq_id = src_queue.ops[-1].seq_id if src_queue.ops else -1
            
            src_sync_info = _SyncInfo(
                sync_action=_SyncAction.EVENT_WAIT,
                sync_op=sync_op,
                blocking_seq_id=blocking_seq_id,
                release_event_id=event_id,
            )
            src_queue.sync_infos[blocking_seq_id].append(src_sync_info)
            return self._sync_funcs.event_wait(self_event, stream)

        @wraps(self._sync_funcs.synchronize)
        def synchronize(device: torch.cuda.device = None):
            sync_op = _SyncOp.sync
            logger.debug(f"Global {sync_op}")
            for queue in self.streamid_to_queue.values():
                blocking_seq_id = queue.ops[-1].seq_id if queue.ops else -1
                sync_info = _SyncInfo(
                    sync_action=_SyncAction.SYNCHRONIZE_WAIT,
                    sync_op=sync_op,
                    blocking_seq_id=blocking_seq_id,
                )
                queue.sync_infos[blocking_seq_id].append(sync_info)
            sync_info = _SyncInfo(
                _SyncAction.SYNCHRONIZE_WAIT,
            )
            self.global_sync_infos.append(sync_info)
            return self._sync_funcs.synchronize(device)

        @wraps(self._sync_funcs.stream_synchronize)
        def stream_synchronize(self_stream: torch.cuda.Stream):
            sync_op = _SyncOp.stream_sync
            logger.debug(f"{sync_op} Stream {self_stream.stream_id}")
            dst_queue = self.streamid_to_queue.setdefault(
                self_stream.stream_id,
                _Queue(
                    stream_id=self_stream.stream_id,
                    priority=self_stream.priority,
                    state=_State.READY,
                    global_sync_infos=self.global_sync_infos,
                    ops=list(),
                ),
            )
            release_seq_id = dst_queue.ops[-1].seq_id if dst_queue.ops else -1
            if release_seq_id != -1:
                for queue in self.streamid_to_queue.values():
                    blocking_seq_id = queue.ops[-1].seq_id if queue.ops else -1
                    sync_info = _SyncInfo(
                        sync_action=_SyncAction.STREAM_WAIT,
                        sync_op=sync_op,
                        blocking_seq_id=blocking_seq_id,
                        release_seq_id=release_seq_id,
                    )
                    queue.sync_infos[blocking_seq_id].append(sync_info)
                sync_info = _SyncInfo(
                    _SyncAction.STREAM_WAIT, release_seq_id=release_seq_id
                )
                self.global_sync_infos.append(sync_info)
                dst_sync_info = _SyncInfo(
                    sync_action=_SyncAction.STREAM_RELEASE,
                    sync_op=sync_op,
                    release_seq_id=release_seq_id
                )
                dst_queue.sync_infos[release_seq_id].append(dst_sync_info)
            return self._sync_funcs.stream_synchronize(self_stream)

        @wraps(self._sync_funcs.work_wait)
        def work_wait(*args, **kwargs) -> bool:
            self_work = args[0]
            assert isinstance(self_work, FakeWork)
            assert hasattr(self_work, "seq_id")
            src_op_info = self.work_registry.pop(self_work.seq_id, None)
            assert src_op_info is not None, f"Work is not registered {self_work.seq_id}"
            self._work_wait(src_op_info)
            return self._sync_funcs.work_wait(*args, **kwargs)

        torch.cuda.Stream.wait_stream = wait_stream
        torch.cuda.Stream.wait_event = wait_event
        torch.cuda.Event.record = event_record
        torch.cuda.Event.wait = event_wait
        torch.cuda.Event.synchronize = event_synchronize
        torch.cuda.synchronize = synchronize
        torch.cuda.Stream.synchronize = stream_synchronize
        FakeWork.wait = work_wait

    def restore_sync_ops(self):
        torch.cuda.Stream.wait_stream = self._sync_funcs.wait_stream
        torch.cuda.Stream.wait_event = self._sync_funcs.wait_event
        torch.cuda.Event.record = self._sync_funcs.event_record
        torch.cuda.Event.wait = self._sync_funcs.event_wait
        torch.cuda.Event.synchronize = self._sync_funcs.event_synchronize
        torch.cuda.synchronize = self._sync_funcs.synchronize
        torch.cuda.Stream.synchronize = self._sync_funcs.stream_synchronize
        FakeWork.wait = self._sync_funcs.work_wait

    def simulate(self) -> float:
        simulated_time = 0.0
        queues = list(self.streamid_to_queue.values())
        for queue in queues:
            logger.debug("------------------------", queue.stream_id)
            logger.debug(queue.sync_infos)
            for op_info in queue.ops:
                logger.debug(op_info)

        # First make the queues wait for synchronization events

        def get_ready_or_running_queues(queues: list[_Queue]) -> list[_Queue]:
            return [
                queue
                for queue in queues
                if queue.state in [_State.READY, _State.RUNNING]
            ]

        def get_resource_independent_queues(
            ready_or_running_queues: list[_Queue],
        ) -> list[_Queue]:
            res_to_queue: dict[Resource, _Queue] = {}

            for queue in ready_or_running_queues:
                if queue.state == _State.RUNNING:
                    head_op_resource = queue.ops[0].resource
                    assert head_op_resource not in res_to_queue
                    res_to_queue[head_op_resource] = queue

        # Function for getting the head ops for resource independent queues

        return simulated_time
