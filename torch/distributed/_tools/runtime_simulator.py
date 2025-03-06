import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import auto, Enum
from functools import wraps
from typing import Any, NamedTuple

import torch
from torch._C._distributed_c10d import FakeWork, ProcessGroup


# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level to INFO
logger.setLevel(logging.DEBUG)


class Resource(Enum):
    INTRA_COMM = auto()
    INTER_COMM = auto()
    INTER_INTRA_COMM = auto()
    COMP = auto()
    HOST_MEM = auto()
    DMA_MEM = auto()


class _State(Enum):
    WAITING = auto()
    RUNNING = auto()
    READY = auto()
    COMPLETE = auto()


class _SyncOps(str, Enum):
    event_record = "event_record"
    event_wait = "event_wait"
    event_synchronize = "event_synchronize"
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


class OpType(Enum):
    compute = auto()
    collective = auto()
    sync = auto()


class _SyncAction(Enum):
    STREAM_WAIT = auto()
    EVENT_WAIT = auto()
    SYNCHRONIZE_WAIT = auto()
    STREAM_RELEASE = auto()
    EVENT_RELEASE = auto()


@dataclass
class _SyncInfo:
    sync_action: _SyncAction
    blocking_seq_id: int = -1
    release_seq_id: int = -1
    blocking_event_id: int = -1
    release_event_id: int = -1


@dataclass
class _OpInfo:
    seq_id: int
    func: str
    type: OpType
    stream_id: int
    resource: Resource
    run_time: float
    rem_time: float


class _Queue:
    def __init__(
        self,
        stream_id: int,
        priority: int,
        state: _State,
        global_sync_events: list[_SyncInfo],
        ops: list[_OpInfo] = [],
    ) -> None:
        self.stream_id = stream_id
        self.priority = priority
        self.state = state
        self.ops = ops
        self.sync_infos: dict[int, list[_SyncInfo]] = defaultdict(list)
        self.wait_sync_infos: dict[int, list[_SyncInfo]] = {}
        for sync_info in global_sync_events:
            self.sync_infos[-1].append(deepcopy(sync_info))


class Simulator:
    def __init__(self, pg_to_resource: dict[ProcessGroup, Resource]):
        self.streamid_to_queue: dict[int, _Queue] = {}
        self.pg_to_resource = pg_to_resource
        self.seq_id = 0
        self.work_registry: dict[int, _OpInfo] = {}
        # self.functional_work_registry: WeakKeyDictionary[torch.UntypedStorage, _OpInfo] = WeakKeyDictionary()
        self.global_sync_events: list[_SyncInfo] = []

    def create_and_queue_op(
        self,
        func: Any,
        op_type: OpType,
        resource: Resource,
        op_time: float,
        stream_id: int = None,
        stream_priority: int = None,
    ):
        if stream_id is None:
            current_stream = torch.cuda.current_stream()
            stream_id = current_stream.stream_id
            stream_priority = current_stream.priority
        queue = self.streamid_to_queue.setdefault(
            stream_id,
            _Queue(
                stream_id,
                stream_priority,
                _State.READY,
                self.global_sync_events,
                list(),
            ),
        )
        op_info = _OpInfo(
            self.seq_id, str(func), op_type, stream_id, resource, op_time, op_time
        )
        queue.ops.append(op_info)
        self.seq_id += 1
        return op_info

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
            sync_func = _SyncOps.wait_stream
            logger.debug(
                f"{sync_func} Self Stream {self_stream.stream_id} Dst Stream {dst_stream.stream_id}"
            )
            dst_queue = self.streamid_to_queue.setdefault(
                dst_stream.stream_id,
                _Queue(
                    dst_stream.stream_id,
                    dst_stream.priority,
                    _State.READY,
                    self.global_sync_events,
                    list(),
                ),
            )
            src_queue = self.streamid_to_queue.setdefault(
                self_stream.stream_id,
                _Queue(
                    self_stream.stream_id,
                    self_stream.priority,
                    _State.READY,
                    self.global_sync_events,
                    list(),
                ),
            )
            blocking_seq_id = src_queue.ops[-1].seq_id if len(src_queue.ops) > 0 else -1
            release_seq_id = dst_queue.ops[-1].seq_id if len(dst_queue.ops) > 0 else -1

            if release_seq_id != -1:
                src_sync_info = _SyncInfo(
                    _SyncAction.STREAM_WAIT,
                    blocking_seq_id=blocking_seq_id,
                    release_seq_id=release_seq_id,
                )
                dst_sync_info = _SyncInfo(
                    _SyncAction.STREAM_RELEASE,
                    blocking_seq_id=blocking_seq_id,
                    release_seq_id=release_seq_id,
                )
                src_queue.sync_infos[blocking_seq_id].append(src_sync_info)
                dst_queue.sync_infos[release_seq_id].append(dst_sync_info)

            return self._sync_funcs.wait_stream(self_stream, dst_stream)

        @wraps(self._sync_funcs.wait_event)
        def wait_event(self_stream: torch.cuda.Stream, event: torch.cuda.Event):
            sync_func = _SyncOps.wait_event
            logger.debug(
                f"{sync_func} Self Stream {self_stream.stream_id} Event {id(event)}"
            )
            src_queue = self.streamid_to_queue.setdefault(
                self_stream.stream_id,
                _Queue(
                    self_stream.stream_id,
                    self_stream.priority,
                    _State.READY,
                    self.global_sync_events,
                    list(),
                ),
            )
            blocking_seq_id = src_queue.ops[-1].seq_id if len(src_queue.ops) > 0 else -1
            event_id = id(event)
            src_sync_info = _SyncInfo(
                _SyncAction.STREAM_WAIT,
                blocking_seq_id=blocking_seq_id,
                release_event_id=event_id,
            )
            src_queue.sync_infos[blocking_seq_id].append(src_sync_info)
            return self._sync_funcs.wait_event(self_stream, event)

        @wraps(self._sync_funcs.event_synchronize)
        def event_synchronize(self_event: torch.cuda.Event):
            sync_func = _SyncFunctions.event_synchronize
            logger.debug(f"{sync_func} Event {id(self_event)}")
            event_id = id(self_event)
            for queue in self.streamid_to_queue.values():
                blocking_seq_id = queue.ops[-1].seq_id if len(queue.ops) > 0 else -1
                sync_info = _SyncInfo(
                    _SyncAction.EVENT_WAIT,
                    blocking_seq_id=blocking_seq_id,
                    release_event_id=event_id,
                )
                queue.sync_infos[blocking_seq_id].append(sync_info)
            global_sync_info = _SyncInfo(
                _SyncAction.EVENT_WAIT, release_event_id=event_id
            )
            self.global_sync_events.append(global_sync_info)
            return self._sync_funcs.event_synchronize(self_event)

        @wraps(self._sync_funcs.event_record)
        def event_record(
            self_event: torch.cuda.Event, stream: torch.cuda.Stream = None
        ):
            sync_func = _SyncOps.event_record
            if stream is None:
                stream = torch.cuda.current_stream()
            logger.debug(
                f"{sync_func} Event {id(self_event)} Stream {stream.stream_id}"
            )
            event_op = self.create_and_queue_op(
                sync_func,
                OpType.sync,
                Resource.COMP,
                0.0,
                stream.stream_id,
                stream.priority,
            )
            src_queue = self.streamid_to_queue[stream.stream_id]
            event_id = id(self_event)
            event_sync_info = _SyncInfo(
                sync_action=_SyncAction.EVENT_RELEASE, release_event_id=event_id
            )
            src_queue.sync_infos[event_op.seq_id].append(event_sync_info)
            return self._sync_funcs.event_record(self_event, stream)

        @wraps(self._sync_funcs.event_wait)
        def event_wait(self_event: torch.cuda.Event, stream: torch.cuda.Stream = None):
            sync_func = _SyncOps.event_wait
            if stream is None:
                stream = torch.cuda.current_stream()
            logger.debug(
                f"{sync_func} Event {id(self_event)} Stream {stream.stream_id}"
            )
            src_queue = self.streamid_to_queue.setdefault(
                stream.stream_id,
                _Queue(
                    stream.stream_id,
                    stream.priority,
                    _State.READY,
                    self.global_sync_events,
                    list(),
                ),
            )
            blocking_seq_id = src_queue.ops[-1].seq_id if len(src_queue.ops) > 0 else -1
            event_id = id(self_event)
            src_sync_info = _SyncInfo(
                _SyncAction.EVENT_WAIT,
                blocking_seq_id=blocking_seq_id,
                release_event_id=event_id,
            )
            src_queue.sync_infos[blocking_seq_id].append(src_sync_info)
            return self._sync_funcs.event_wait(self_event, stream)

        @wraps(self._sync_funcs.synchronize)
        def synchronize(device: torch.cuda.device = None):
            sync_func = _SyncOps.sync
            logger.debug(f"{sync_func}")
            for queue in self.streamid_to_queue.values():
                blocking_seq_id = queue.ops[-1].seq_id if len(queue.ops) > 0 else -1
                sync_info = _SyncInfo(
                    _SyncAction.SYNCHRONIZE_WAIT,
                    blocking_seq_id=blocking_seq_id,
                )
                queue.sync_infos[blocking_seq_id].append(sync_info)
            global_sync_info = _SyncInfo(
                _SyncAction.SYNCHRONIZE_WAIT,
            )
            self.global_sync_events.append(global_sync_info)
            return self._sync_funcs.synchronize(device)

        @wraps(self._sync_funcs.stream_synchronize)
        def stream_synchronize(self_stream: torch.cuda.Stream):
            sync_func = _SyncOps.stream_sync
            dst_queue = self.streamid_to_queue.setdefault(
                self_stream.stream_id,
                _Queue(
                    self_stream.stream_id,
                    self_stream.priority,
                    _State.READY,
                    self.global_sync_events,
                    list(),
                ),
            )
            release_seq_id = dst_queue.ops[-1].seq_id if len(dst_queue.ops) > 0 else -1
            if release_seq_id != -1:
                for queue in self.streamid_to_queue.values():
                    blocking_seq_id = queue.ops[-1].seq_id if len(queue.ops) > 0 else -1
                    sync_info = _SyncInfo(
                        _SyncAction.STREAM_WAIT,
                        blocking_seq_id=blocking_seq_id,
                        release_seq_id=release_seq_id,
                    )
                    queue.sync_infos[blocking_seq_id].append(sync_info)
                global_sync_info = _SyncInfo(
                    _SyncAction.STREAM_WAIT, release_seq_id=release_seq_id
                )
                self.global_sync_events.append(global_sync_info)
                dst_sync_info = _SyncInfo(
                    _SyncAction.STREAM_RELEASE, release_seq_id=release_seq_id
                )
                dst_queue.sync_infos[release_seq_id].append(dst_sync_info)
                logger.debug(f"{sync_func} Stream {self_stream.stream_id}")
            return self._sync_funcs.stream_synchronize(self_stream)

        @wraps(self._sync_funcs.work_wait)
        def work_wait(*args, **kwargs) -> bool:
            # sync_func = _SyncFunctions.work_wait
            # self_work = args[0]
            # src_op_info = self.work_registry[id(self_work.boxed())]
            # dst_queue = self.streamid_to_queue[src_op_info.stream_id]
            # release_seq_id = src_op_info.seq_id
            # logger.debug(f"{sync_func} for {release_seq_id}")
            # assert release_seq_id != -1
            # for queue in self.streamid_to_queue.values():
            #     blocking_seq_id = queue.ops[-1].seq_id if len(queue.ops) > 0 else -1
            #     sync_info = _SyncInfo(
            #         _SyncAction.STREAM_WAIT,
            #         blocking_seq_id=blocking_seq_id,
            #         release_seq_id=release_seq_id
            #     )
            #     queue.sync_infos[blocking_seq_id].append(sync_info)
            # global_sync_info = _SyncInfo(
            #     _SyncAction.STREAM_WAIT,
            #     release_seq_id=release_seq_id
            # )
            # self.global_sync_events.append(global_sync_info)
            # dst_sync_info = _SyncInfo(
            #     _SyncAction.STREAM_RELEASE,
            #     release_seq_id=release_seq_id
            # )
            # dst_queue.sync_infos[release_seq_id].append(dst_sync_info)

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
