import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ProcessGroup, Work
from torch.futures import Future
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.distributed._functional_collectives import *
from torch.utils._python_dispatch import TorchDispatchMode
from functools import wraps
from contextlib import contextmanager, nullcontext
import logging
from datetime import timedelta
from typing import cast, Optional, overload

aten = torch.ops.aten
c10d = torch.ops.c10d
_c10d_functional = torch.ops._c10d_functional

class FakeWork(Work):
    def __init__(self):
        super().__init__()

    def get_future(self) -> Future:
        future = Future()
        future.set_result(None)
        return future

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        return True


# Meta functions for collective operations with FakeWork


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

# Intercepting collectives

collective_op_funcs = [
    c10d.broadcast_.default,
    c10d.allreduce_.default,
    c10d.reduce_.default,
    c10d.send.default,
    c10d.recv_.default,
    c10d.allgather_.default,
    c10d.reduce_scatter_.default,
    c10d._reduce_scatter_base_.default,
    c10d._allgather_base_.default,
    c10d.gather_.default,
    c10d.scatter_.default,
    c10d.alltoall_.default,
    _c10d_functional.broadcast.default,
    _c10d_functional.all_reduce.default,
    _c10d_functional.all_to_all_single.default,
    _c10d_functional.all_gather_into_tensor.default,
    _c10d_functional.reduce_scatter_tensor.default
]

class IgnoreDistMode(TorchDispatchMode):

    class CollectiveOp:
        # CollectiveOp stores the name and data sizes of the collective operation
        # so that the cost of the operation can be calculated
        def __init__(self, func, res, *args, **kwargs):
            self.func = func
            self.tensor_args = self.parse_collective_args(args)
            self.data_size = self.get_data_sizes()
            self.pg_name, self.world_size = self.get_process_group_properties(args)
            self.output_data_size = self.get_output_data_size(args, res)

            # process group, world size of process group

            # logging.info(f"Data size: {self.data_size}")
            # logging.info(f"Process Group: {self.pg_name}, {self.world_size}")
            logging.info(f"res: {res}")
            logging.info(f"Output data size: {self.output_data_size}")

        # Extract the tensor from the collective operator arguments
        def parse_collective_args(self, args):
            if self.func in [
                c10d.broadcast_.default,
                _c10d_functional.broadcast.default,
                c10d.allreduce_.default,
                _c10d_functional.all_reduce.default,
                c10d.reduce_.default,
                c10d.send.default,
                c10d.recv_.default,
                _c10d_functional.all_to_all_single.default,
            ]:
                return args[0]
            elif self.func in [
                c10d.allgather_.default,
                c10d._allgather_base_.default,
                c10d.reduce_scatter_.default,
                c10d._reduce_scatter_base_.default,
                c10d.gather_.default,
                c10d.scatter_.default,
                c10d.alltoall_.default,
            ]:
                return args[1]
            elif self.func in [
                _c10d_functional.all_gather_into_tensor.default,
                _c10d_functional.reduce_scatter_tensor.default,
            ]:
                return args[0]
            else:
                return None

        def get_data_sizes(self):
            if self.func in [
                c10d.broadcast_.default,
                c10d.allreduce_.default,
                c10d.allgather_.default,
                c10d.gather_.default,
                c10d.reduce_.default,
                c10d.alltoall_.default,
                c10d.send.default,
                c10d.recv_.default,
            ]:  
                return [t.untyped_storage().nbytes() for t in self.tensor_args]
            elif self.func in [
                c10d.reduce_scatter_.default,
                c10d.scatter_.default,
            ]:
                return [t.untyped_storage().nbytes() for l in self.tensor_args for t in l]
            elif self.func in [
                _c10d_functional.broadcast.default,
                _c10d_functional.all_reduce.default,
                c10d._allgather_base_.default,
                _c10d_functional.all_gather_into_tensor.default,
                _c10d_functional.reduce_scatter_tensor.default,
                c10d._reduce_scatter_base_.default,
                _c10d_functional.all_to_all_single.default,
            ]:
                return self.tensor_args.untyped_storage().nbytes()
            else:
                return None
            
        def get_process_group_properties(self, args):
            if self.func in [
                c10d.broadcast_.default,
                c10d.allreduce_.default,
                c10d.reduce_.default,
                c10d.send.default,
                c10d.recv_.default,
            ]:
                pg = ProcessGroup.unbox(args[1])
            elif self.func in [
                c10d.allgather_.default,
                c10d._allgather_base_.default,
                c10d.reduce_scatter_.default,
                c10d._reduce_scatter_base_.default,
                c10d.gather_.default,
                c10d.scatter_.default,
                c10d.alltoall_.default,
            ]:
                pg = ProcessGroup.unbox(args[2])
            else:
                # TODO: functional collectives
                return None, None

            return pg.name(), pg.size()

        def get_output_data_size(self, args, res):
            if self.func in [
                c10d._allgather_base_.default,
                c10d._reduce_scatter_base_.default,
            ]:
                return args[0].untyped_storage().nbytes()
            elif self.func in [
                c10d.reduce_scatter_.default,
                c10d.scatter_.default,
                c10d.alltoall_.default,
            ]:
                return [t.untyped_storage().nbytes() for t in args[0]]
            elif self.func in [
                c10d.allgather_.default,
                c10d.gather_.default,
            ]:
                return [t.untyped_storage().nbytes() for l in args[0] for t in l]
            elif self.func in [
                c10d.broadcast_.default,
                c10d.allreduce_.default,
            ]:
                return [t.untyped_storage().nbytes() for t in res[0]]
            elif type(res) == list and type(res[0]) == torch.FakeTensor:
                return res[0].untyped_storage().nbytes()
            elif type(res) == FakeTensor:
                return res.untyped_storage().nbytes()
            else:
                return None

    collectives = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        logging.info(f"Function name: {str(func.__name__)}")
        logging.info(f"Function type: {type(func)}")
        logging.info(f"Func: {func}")
        logging.info(f"Args: {args}")

        res = func(*args, **kwargs or {})

        if func in collective_op_funcs:
            self.collectives.append(self.CollectiveOp(func, res, *args, **kwargs))
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
            test_tensor_list = [torch.randn(300, device="cuda") for _ in range(4)]
            test_tensor_list_2 = [torch.randn(300, device="cuda") for _ in range(4)]
            test_tensor = torch.randn(100, device="cuda")

            # testing for collective operations
            dist.broadcast(test_tensor, src=0)
            dist.all_reduce(test_tensor)
            dist.all_to_all(test_tensor_list_2, test_tensor_list)
            dist.all_gather(test_tensor_list, test_tensor)
            dist.all_gather_into_tensor(test_tensor, test_tensor)
            dist.reduce_scatter(test_tensor, test_tensor_list)
            dist.reduce_scatter_tensor(test_tensor, test_tensor)
            dist.reduce(test_tensor, dst=0)
            dist.scatter(test_tensor, scatter_list=test_tensor_list, src=0)
            dist.gather(test_tensor, gather_list=test_tensor_list, dst=0)
            dist.barrier()
            dist.send(test_tensor, dst=1)
            dist.recv(test_tensor, src=1)

            # testing for functional collectives
            output = wait_tensor(test_tensor)
            output = broadcast(test_tensor, src=0, group=dist.group.WORLD)
            output = all_reduce(test_tensor, reduceOp="avg", group=dist.group.WORLD)
            output = all_gather_tensor(
                test_tensor, gather_dim=0, group=dist.group.WORLD
            )
            output = reduce_scatter_tensor(
                test_tensor, scatter_dim=0, reduceOp="sum", group=dist.group.WORLD
            )
            output = all_to_all_single(
                test_tensor,
                output_split_sizes=[0],
                input_split_sizes=[1],
                group=dist.group.WORLD,
            )
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
