import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ProcessGroup, Work
from torch.futures import Future
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.distributed._functional_collectives import all_reduce
from torch.utils._python_dispatch import TorchDispatchMode
from functools import wraps
from contextlib import contextmanager, nullcontext
import logging
from datetime import timedelta
from typing import cast, Optional, overload


class FakeWork(Work):
    def __init__(self):
        super().__init__()

    def get_future(self) -> Future:
        future = Future()
        future.set_result(None)
        return future

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        return True


def _broadcast_meta(*args):
    fakework = FakeWork()
    fakework.__setattr__("getFuture", fakework.get_future)
    fakework_script_obj = fakework.boxed()
    return (args[0], fakework_script_obj)


def _all_reduce_meta(*args):
    fakework = FakeWork()
    fakework.__setattr__("getFuture", fakework.get_future)
    fakework_script_obj = fakework.boxed()
    return (args[0], fakework_script_obj)


def _all_gather_meta(*args):
    fakework = FakeWork()
    fakework.__setattr__("getFuture", fakework.get_future)
    fakework_script_obj = fakework.boxed()
    return (args[0], fakework_script_obj)


def _all_gather_into_tensor_meta(*args):
    fakework = FakeWork()
    fakework.__setattr__("getFuture", fakework.get_future)
    fakework_script_obj = fakework.boxed()
    return (args[0], fakework_script_obj)


def _reduce_scatter_meta(*args):
    fakework = FakeWork()
    fakework.__setattr__("getFuture", fakework.get_future)
    fakework_script_obj = fakework.boxed()
    return (args[0], fakework_script_obj)


def _reduce_scatter_tensor_meta(*args):
    fakework = FakeWork()
    fakework.__setattr__("getFuture", fakework.get_future)
    fakework_script_obj = fakework.boxed()
    return (args[0], fakework_script_obj)


def _reduce_meta(*args):
    fakework = FakeWork()
    fakework.__setattr__("getFuture", fakework.get_future)
    fakework_script_obj = fakework.boxed()
    return fakework_script_obj


def _reduce_meta(*args):
    fakework = FakeWork()
    fakework.__setattr__("getFuture", fakework.get_future)
    fakework_script_obj = fakework.boxed()
    return fakework_script_obj


def _gather_meta(*args):
    fakework = FakeWork()
    fakework.__setattr__("getFuture", fakework.get_future)
    fakework_script_obj = fakework.boxed()
    return fakework_script_obj

def _scatter_meta(*args):
    fakework = FakeWork()
    fakework.__setattr__("getFuture", fakework.get_future)
    fakework_script_obj = fakework.boxed()
    return (args[0], fakework_script_obj)

def _alltoall_meta(*args):
    fakework = FakeWork()
    fakework.__setattr__("getFuture", fakework.get_future)
    fakework_script_obj = fakework.boxed()
    return (args[0], fakework_script_obj)

def _send_meta(*args):
    fakework = FakeWork()
    fakework.__setattr__("getFuture", fakework.get_future)
    fakework_script_obj = fakework.boxed()
    return fakework_script_obj


def _recv_meta(*args):
    fakework = FakeWork()
    fakework.__setattr__("getFuture", fakework.get_future)
    fakework_script_obj = fakework.boxed()
    return fakework_script_obj


def _barrier_meta(*args):
    fakework = FakeWork()
    fakework.__setattr__("getFuture", fakework.get_future)
    fakework_script_obj = fakework.boxed()
    return fakework_script_obj


if not torch._running_with_deploy():
    # Library MUST be defined at module scope or it doesn't work
    # Creating a "DEF" Library always crashes torch::deploy so we create our
    # Library instances here guarded against running inside it
    lib_impl = torch.library.Library("c10d", "IMPL")
    lib_impl.impl("broadcast_", _broadcast_meta, "Meta")
    lib_impl.impl("allreduce_", _all_reduce_meta, "Meta")
    lib_impl.impl("allgather_", _all_gather_meta, "Meta")
    lib_impl.impl("_allgather_base_", _all_gather_into_tensor_meta, "Meta")
    lib_impl.impl("reduce_scatter_", _reduce_scatter_meta, "Meta")
    lib_impl.impl("_reduce_scatter_base_", _reduce_scatter_tensor_meta, "Meta")
    lib_impl.impl("reduce_", _reduce_meta, "Meta")
    lib_impl.impl("gather_", _gather_meta, "Meta")
    lib_impl.impl("scatter_", _scatter_meta, "Meta")
    lib_impl.impl("alltoall_", _alltoall_meta, "Meta")
    lib_impl.impl("barrier", _barrier_meta, "Meta")
    lib_impl.impl("send", _send_meta, "Meta")
    lib_impl.impl("recv_", _recv_meta, "Meta")


class IgnoreDistMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        logging.info(f"Function name: {str(func.__name__)}")
        logging.info(f"Function type: {type(func)}")
        logging.info(f"Func: {func}")

        res = func(*args, **kwargs or {})
        return res


def run_test():
    try:
        rank = dist.get_rank()
    except:
        rank = 0
    logging.getLogger().setLevel(logging.DEBUG if rank == 0 else logging.CRITICAL)

    # with nullcontext():
    with FakeTensorMode():
        with IgnoreDistMode():
            test_tensor_list = [torch.randn(1000, device="cuda") for _ in range(3)]
            test_tensor = torch.randn(10000, device="cuda")

            # testing for collective operations
            dist.broadcast(test_tensor, src=0)
            dist.all_reduce(test_tensor)
            dist.all_gather(test_tensor_list, test_tensor)
            dist.all_gather_into_tensor(test_tensor, test_tensor)
            dist.reduce_scatter(test_tensor, test_tensor_list)
            dist.reduce_scatter_tensor(test_tensor, test_tensor)
            dist.reduce(test_tensor, dst=0)
            dist.gather(test_tensor, gather_list=test_tensor_list, dst=0)
            dist.scatter(test_tensor, scatter_list=test_tensor_list, src=0)
            dist.all_to_all(test_tensor_list, test_tensor_list)
            dist.barrier()
            dist.send(test_tensor, dst=1)
            dist.recv(test_tensor, src=1)

        dist.barrier()


if __name__ == "__main__":
    gpu_id = 0
    world_size = 4
    dims = (world_size,)
    names = ("dp",)
    store = FakeStore()
    dist.init_process_group("fake", rank=gpu_id, world_size=world_size, store=store)
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    try:
        run_test()
    finally:
        dist.destroy_process_group()
