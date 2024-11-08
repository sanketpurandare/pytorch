from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import auto, Enum, StrEnum
from functools import wraps
import itertools
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Self, Set, Tuple

import torch
from torch._C._distributed_c10d import ProcessGroup, _register_process_group
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._tools.runtime_estimator import RuntimeEstimator, get_flattened_tensor
from torch.distributed._tools.fake_collectives import collective_op_funcs, CollectiveOp
from torch.distributed._tools.collective_model_deploy import predict_communication
import torch.utils._pytree as pytree

class _Resource(Enum):
    INTRA_COMM = auto()
    INTER_COMM = auto()
    INTER_INTRA_COMM = auto()
    COMP = auto()
    HOST_MEM = auto()
    DMA_MEM = auto()

class _State(Enum):
    WAITING = auto()
    RUNNABLE = auto()
    RUNNING = auto()
    READY = auto()

class _SyncOps(StrEnum):
    event_record = "event_record"
    event_wait = "event_wait"
    wait_event = "wait_event"
    wait_stream = "wait_stream"
    stream_sync = "stream_sync"
    sync = "sync"

class _SyncFunctions(NamedTuple):
    wait_stream: callable
    wait_event: callable
    event_record: callable
    event_wait: callable
    synchronize: callable
    stream_synchronize: callable

class _OpType(Enum):
    compute = auto()
    collective = auto()
    sync = auto()

@dataclass
class _OpInfo:
    seq_id: int
    func: str
    type: _OpType
    stream_id: int
    res: _Resource
    run_time: float
    rem_time: float
    sync_info: Tuple[int, int] = ()


class _Queue:
    def __init__(self, priority, state, ops=[]) -> None:      
        self.priority: int = priority
        self.state: _State = state
        self.ops: List[_OpInfo] = ops
        self.blocking_event_seq_dict: Dict[int, int] = {}
        self.blocking_event_ids: Set[int] = set()
        self.to_be_blocked_seq_ids: Set[int] = set()
        self.to_be_synced_seq_ids: Set[int] = set()

class SPMDRuntimeEstimator(RuntimeEstimator):

    def __init__(self, world_mesh: Optional[DeviceMesh]=None) -> None:
        super().__init__()
        self._streamid_to_queue: Dict[int, _Queue] = {}
        self._pg_name_to_resource: Dict[str, Tuple[ProcessGroup, _Resource]] = {}
        if world_mesh is not None:
            _ndims = world_mesh.ndim
            assert _ndims <= 3, "Does not support > 3D mesh"
            _mesh_dim_names = world_mesh.mesh_dim_names
            if _ndims == 3:
                # Assume HSDP + TP
                tp_pg = world_mesh.get_group("tp")
                dp_replicate_pg = world_mesh.get_group("dp_replicate")
                dp_shard_pg = world_mesh.get_group("dp_shard")

                self._pg_name_to_resource[tp_pg.group_name] = (tp_pg, _Resource.INTRA_COMM)
                self._pg_name_to_resource[dp_replicate_pg.group_name] = (dp_replicate_pg, _Resource.INTER_COMM)
                self._pg_name_to_resource[dp_shard_pg.group_name] = (dp_shard_pg, _Resource.INTER_COMM)

                
            elif _ndims == 2:
                # Can be HSDP or FSDP + TP
                if "tp" in _mesh_dim_names:
                    # Assume FSDP + TP
                    tp_pg = world_mesh.get_group("tp")
                    dp_pg = world_mesh.get_group("dp")
                    self._pg_name_to_resource[tp_pg.group_name] = (tp_pg, _Resource.INTRA_COMM)
                    self._pg_name_to_resource[dp_pg.group_name] = (dp_pg, _Resource.INTER_COMM)
                else:
                    # Assume HSDP
                    dp_replicate_pg = world_mesh.get_group("dp_replicate")
                    dp_shard_pg = world_mesh.get_group("dp_shard")
                    self._pg_name_to_resource[dp_replicate_pg.group_name] = (dp_replicate_pg, _Resource.INTER_COMM)
                    self._pg_name_to_resource[dp_shard_pg.group_name] = (dp_shard_pg, _Resource.INTER_INTRA_COMM)

            else:
                # Assume FSDP
                dp_pg = world_mesh.get_group("dp")
                print("Reached here")
                print(f"PG name: {dp_pg.group_name}")
                self._pg_name_to_resource[dp_pg.group_name] = (dp_pg, _Resource.INTER_INTRA_COMM)

        _default_pg = torch.distributed.group.WORLD
        _register_process_group("custom", _default_pg)
        print(_default_pg.group_name)
        if _default_pg is not None and _default_pg not in self._pg_name_to_resource:
            self._pg_name_to_resource[_default_pg.group_name] = (_default_pg, _Resource.INTER_INTRA_COMM)
        _default_stream = torch.cuda.default_stream()
        self._streamid_to_queue[_default_stream.stream_id] = _Queue(_default_stream.priority, _State.READY, list())
        self.mod_commtimes: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.total_comm_time: float = 0.0
        self.total_runtime: float = 0.0
        self.seq_id = 0
        self.collective_estimation_type: str

    def create_and_queue_sync_op(self, sync_func: _SyncOps, sync_info: Tuple[int, int], stream_id:int=None, stream_priority:int=None):
        resource = _Resource.COMP
        op_type = _OpType.sync
        op_time = sys.float_info.min
        op_info = self.create_and_queue_op(sync_func, op_type, resource, op_time, sync_info, stream_id, stream_priority)
        return op_info

    def create_and_queue_op(self, func: Any, op_type: _OpType, resource: _Resource, op_time:float, sync_info: Tuple[int, int]=(), stream_id:int=None, stream_priority:int=None):
        if stream_id is None:
            _current_stream = torch.cuda.current_stream()
            stream_id = _current_stream.stream_id
            stream_priority = _current_stream.priority
        queue = self._streamid_to_queue.setdefault(stream_id, _Queue(stream_priority, _State.READY, list()))
        op_info = _OpInfo(self.seq_id, str(func), op_type, stream_id, resource, op_time, op_time, sync_info)
        queue.ops.append(op_info)
        self.seq_id += 1
        return op_info

    def capture_sync_ops(self):
        self._sync_funcs = _SyncFunctions(
            wait_stream=torch.cuda.Stream.wait_stream,
            wait_event=torch.cuda.Stream.wait_event,
            event_record=torch.cuda.Event.record,
            event_wait=torch.cuda.Event.wait,
            synchronize=torch.cuda.synchronize,
            stream_synchronize=torch.cuda.Stream.synchronize
        )

        @wraps(self._sync_funcs.wait_stream)
        def wait_stream(self_stream: torch.cuda.Stream, stream: torch.cuda.Stream):
            sync_func = _SyncOps.wait_stream
            sync_info = (self_stream.stream_id, stream.stream_id)
            self.create_and_queue_sync_op(sync_func, sync_info, self_stream.stream_id, self_stream.priority)
            dst_queue = self._streamid_to_queue.get(stream.stream_id, None)
            if dst_queue is not None and len(dst_queue.ops) > 0:
                last_dst_op = dst_queue.ops[-1]
                src_queue = self._streamid_to_queue[self_stream.stream_id]
                src_queue.to_be_blocked_seq_ids.add(last_dst_op.seq_id)
            return self._sync_funcs.wait_stream(self_stream, stream)

        @wraps(self._sync_funcs.wait_event)
        def wait_event(self_stream: torch.cuda.Stream, event: torch.cuda.Event):
            sync_func = _SyncOps.wait_event
            sync_info = (self_stream.stream_id, id(event))
            self.create_and_queue_sync_op(sync_func, sync_info, self_stream.stream_id, self_stream.priority)
            return self._sync_funcs.wait_event(self_stream, event)

        @wraps(self._sync_funcs.event_record)
        def event_record(self_event: torch.cuda.Event, stream: torch.cuda.Stream=None):
            sync_func = _SyncOps.event_record
            if stream is None:
                stream_id = torch.cuda.current_stream().stream_id
            else:
                stream_id = stream.stream_id
            sync_info = (id(self_event), stream_id)
            self.create_and_queue_sync_op(sync_func, sync_info)
            return self._sync_funcs.event_record(self_event, stream)

        @wraps(self._sync_funcs.event_wait)
        def event_wait(self_event: torch.cuda.Event, stream: torch.cuda.Stream=None):
            sync_func = _SyncOps.event_wait
            if stream is None:
                _stream = torch.cuda.current_stream()
                stream_id = _stream.stream_id
                priority = _stream.priority
            else:
                stream_id = stream.stream_id
                priority = stream.priority
            sync_info = (id(self_event), stream_id)
            op = self.create_and_queue_sync_op(sync_func, sync_info)
            dst_queue = self._streamid_to_queue.setdefault(stream_id, _Queue(priority, _State.READY, []))
            dst_queue.to_be_blocked_seq_ids.add(op.seq_id)
            dst_queue.blocking_event_ids.add(id(self_event))
            dst_queue.blocking_event_seq_dict[id(self_event)] = op.seq_id
            return self._sync_funcs.event_wait(self_event, stream)

        @wraps(self._sync_funcs.synchronize)
        def synchronize(device: torch.cuda.device=None):
            sync_func = _SyncOps.sync
            sync_info = ()
            op = self.create_and_queue_sync_op(sync_func, sync_info)
            for queue in self._streamid_to_queue.values():
                queue.to_be_synced_seq_ids.add(op.seq_id)
            return self._sync_funcs.synchronize(device)

        @wraps(self._sync_funcs.stream_synchronize)
        def stream_synchronize(self_stream: torch.cuda.Stream):
            sync_func = _SyncOps.stream_sync
            sync_info = (self_stream.stream_id,)
            op = self.create_and_queue_sync_op(sync_func, sync_info, self_stream.stream_id, self_stream.priority)
            for queue in self._streamid_to_queue.values():
                queue.to_be_synced_seq_ids.add(op.seq_id)
            return self._sync_funcs.stream_synchronize(self_stream)

        torch.cuda.Stream.wait_stream = wait_stream
        torch.cuda.Stream.wait_event = wait_event
        torch.cuda.Event.record = event_record
        torch.cuda.Event.wait = event_wait
        torch.cuda.synchronize = synchronize
        torch.cuda.Stream.synchronize = stream_synchronize

    def restore_sync_ops(self):
        torch.cuda.Stream.wait_stream = self._sync_funcs.wait_stream
        torch.cuda.Stream.wait_event = self._sync_funcs.wait_event
        torch.cuda.Event.record = self._sync_funcs.event_record
        torch.cuda.Event.wait = self._sync_funcs.event_wait
        torch.cuda.synchronize = self._sync_funcs.synchronize
        torch.cuda.Stream.synchronize = self._sync_funcs.stream_synchronize

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        if func in collective_op_funcs:
            op_type = _OpType.collective
            kwargs = kwargs if kwargs else {}
            res = func(*args, **kwargs)
            desugared_args = pytree.tree_map_only(torch.Tensor, get_flattened_tensor, args)
            desugared_kwargs = pytree.tree_map_only(torch.Tensor, get_flattened_tensor, kwargs)
            desugared_res = pytree.tree_map_only(torch.Tensor, get_flattened_tensor, res)
            collective_data_size = CollectiveOp.get_tensor_size(func, desugared_res, desugared_args, desugared_kwargs)
            
            pg_name, pg_size, pg = CollectiveOp.get_process_group_properties(func, desugared_args)
            pg, resource = self._pg_name_to_resource.setdefault(pg_name, (pg, _Resource.INTER_INTRA_COMM))
            op_time = predict_communication(func, (collective_data_size/(2**20)), pg_size, (resource == _Resource.INTER_COMM))
            self.total_comm_time += op_time

            for par in self._mod_tracker.parents:
                if self._mod_tracker.is_bw:
                    self.mod_commtimes[par]["bw"] += op_time
                else:
                    self.mod_commtimes[par]["fw"] += op_time
        elif func == torch.ops._c10d_functional.wait_tensor:
            op_time = sys.float_info.min
            op_type = _OpType.sync
            kwargs = kwargs if kwargs else {}
            res = func(*args, **kwargs)
            resource = _Resource.COMP

        else:
            op_type = _OpType.compute
            resource = _Resource.COMP
            res, op_time = self._estimate(func, args, kwargs)
            for par in self._mod_tracker.parents:
                if self._mod_tracker.is_bw:
                    self.mod_runtimes[par]["bw"] += op_time
                else:
                    self.mod_runtimes[par]["fw"] += op_time
            self.total_compute_time += op_time
        self.create_and_queue_op(func, op_type, resource, op_time, ())
        return res
    
    def simulate_runtime(self) -> float:
        s_queues = deepcopy(self._streamid_to_queue)
        total_runtime = 0.0

        def get_runnable_queues(queues: Dict[int, _Queue]) -> List[_Queue]:
            runnable_queues = []
            for queue in queues.values():
                if queue.state in [_State.RUNNABLE, _State.RUNNING]:
                    runnable_queues.append(queue)
            return runnable_queues

       
        def get_queue_heads(queues: List[_Queue]) -> List[_OpInfo]:
            head_ops = []
            for queue in queues:
                if len(queue.ops) > 0:
                    head_ops.append(queue.ops[0])
            return head_ops
        
        def get_resource_indep_queues(runnable_queues: List[_Queue], s_queues: Dict[int, _Queue]) -> List[_Queue]:
            indep_queues:Dict[_Resource, _Queue] = {}
            queue_heads = get_queue_heads(runnable_queues)
            for op in queue_heads:
                op_queue = s_queues[op.stream_id]
                if current_queue := indep_queues.get(op.res, None):
                    if current_queue.state == _State.RUNNING:
                        continue
                    elif op_queue.state == _State.RUNNING:
                        indep_queues[op.res] = op_queue
                        continue
                    elif op_queue.priority > current_queue.priority:
                        continue
                    elif op_queue.priority < current_queue.priority:
                        indep_queues[op.res] = op_queue
                        continue
                    else:
                        current_head = current_queue.ops[0]
                        if current_head.seq_id < op.seq_id:
                            continue
                        else:
                            indep_queues[op.res] = op_queue
                            continue
                else:
                    indep_queues[op.res] = op_queue
            return list(indep_queues.values())

        def is_completed(queues: Dict[int, _Queue]) -> bool:
            is_completed = True
            for queue in queues.values():
                if len(queue.ops) > 0:
                    is_completed = False
                    break
            return is_completed

        def maybe_complete_sync(s_queues: Dict[int, _Queue]):
            if all(queue.state == _State.WAITING for queue in s_queues.values()):
                min_sync_seq_id = set()
                for queue in s_queues.values():
                    min_sync_seq_id.add(min(queue.to_be_synced_seq_ids))
                if len(min_sync_seq_id == 1):
                    min_sync_id = min_sync_seq_id.pop()
                    for queue in s_queues.values():
                        while(min_sync_id in queue.to_be_synced_seq_ids):
                            queue.to_be_synced_seq_ids.remove(min_sync_id)
                

        def get_waiting_queues_on_event(event_id: int) -> List[_Queue]:
            waiting_queues = []
            for queue in s_queues.values():
                if queue.state == _State.WAITING and event_id in queue.blocking_event_ids:
                    waiting_queues.append(queue)
            return waiting_queues
        
        def maybe_mark_runnable(queue: _Queue):
            if len(queue.ops) > 0:
                head_op = queue.ops[0]
                min1 = min2 = sys.int_info.default_max_str_digits
                if len(queue.to_be_synced_seq_ids) > 0:
                    min1 = min(queue.to_be_synced_seq_ids)
                if len(queue.to_be_blocked_seq_ids) > 0:
                    min2 = min(queue.to_be_blocked_seq_ids)
                b_seq_id = min(min1, min2)
                if head_op.seq_id < b_seq_id:
                    queue.state = _State.RUNNABLE

        def process_seq_id(seq_id: int, s_queues: Dict[int, _Queue]):
            for queue in s_queues.values():
                while(seq_id in queue.to_be_blocked_seq_ids):
                    queue.to_be_blocked_seq_ids.remove(seq_id)
                maybe_mark_runnable(queue)

        ## Start Simulation
        for queue in s_queues.values():
            if queue.state not in [_State.RUNNABLE, _State.RUNNING]:
                maybe_mark_runnable(queue)
        recorded_events = set()
        while(not is_completed(s_queues)):
            maybe_complete_sync(s_queues)
            runnable_queues = get_runnable_queues(s_queues)
            res_indep_queues = get_resource_indep_queues(runnable_queues, s_queues)
            if len(res_indep_queues) == 0:
                raise AssertionError(f"Maybe Deadlock")
            
            head_ops = get_queue_heads(res_indep_queues)
            min_op = min(head_ops, key=lambda info: info.rem_time)
            print(min_op)
            time_quantum = min_op.rem_time
            for op in head_ops:
                op.rem_time -= time_quantum
                queue = s_queues[op.stream_id]
                queue.state = _State.RUNNING
                if op.rem_time <= 0:
                    popped_op = queue.ops.pop(0)
                    assert op == popped_op, f"Removed {popped_op} but expected {op}."
                    if len(queue.ops) == 0:
                        queue.state = _State.RUNNABLE
                    else:
                        head_op = queue.ops[0]
                        to_block = True
                        min1 = min2 = sys.int_info.default_max_str_digits
                        if len(queue.to_be_synced_seq_ids) > 0:
                            min1 = min(queue.to_be_synced_seq_ids)
                        if len(queue.to_be_blocked_seq_ids) > 0:
                            min2 = min(queue.to_be_blocked_seq_ids)
                        b_seq_id = min(min1, min2)
                        if head_op.seq_id < b_seq_id:
                            to_block = False
                        if to_block:
                            queue.state = _State.WAITING
                        else:
                            queue.state = _State.RUNNABLE

            total_runtime += time_quantum

            if op.type == _OpType.sync:
                match op.func:
                    case 'wait_event':
                        src_stream_id, event_id = op.sync_info
                        if event_id not in recorded_events:
                            src_queue = s_queues[src_stream_id]
                            assert src_queue.state is not _State.RUNNING, "Stream calling wait_event can be different from where it is launched."
                            src_queue.state = _State.WAITING
                            src_queue.blocking_event_ids.add(event_id)
                            src_queue.blocking_event_seq_dict[event_id] = op.seq_id
                    case 'event_record':
                        event_id = op.sync_info[0]
                        recorded_events.add(event_id)
                        waiting_queues = get_waiting_queues_on_event(event_id)
                        for queue in waiting_queues:
                            while(event_id in queue.blocking_event_ids):
                                queue.blocking_event_ids.remove(event_id)
                            blocked_seq_id = queue.blocking_event_seq_dict.get(event_id, None)
                            if blocked_seq_id is not None:
                                while(blocked_seq_id in queue.to_be_blocked_seq_ids):
                                    queue.to_be_blocked_seq_ids.remove(blocked_seq_id)
                            while (event_id in queue.blocking_event_seq_dict):
                                queue.blocking_event_seq_dict.pop(event_id)
                            maybe_mark_runnable(queue)

            process_seq_id(op.seq_id, s_queues)
        self.total_runtime = total_runtime
        return self.total_runtime
    
    def __call__(self, estimate_mode_type: str, collective_mode_type: str) -> Self:
        self.collective_estimation_type = collective_mode_type
        return super().__call__(estimate_mode_type)

    def __enter__(self) -> Self:
        self.capture_sync_ops()
        _default_stream = torch.cuda.default_stream()
        self._streamid_to_queue.clear()
        self._streamid_to_queue[_default_stream.stream_id] = _Queue(_default_stream.priority, _State.RUNNABLE, list())
        self.mod_commtimes: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.total_comm_time: float = 0.0
        self.total_runtime: float = 0.0
        self.seq_id = 0
        return super().__enter__()
    
    def __exit__(self, *args: Any) -> None:
        self.simulate_runtime()
        self.restore_sync_ops()
        print(
            f"Estimated ({self.collective_estimation_type})"
            f" total_comm_time: {self.total_comm_time:.3f} ms"
        )
        print(
            f"Simulation Time: {self.total_runtime:.3f} ms"
        )
        return super().__exit__(*args)


        