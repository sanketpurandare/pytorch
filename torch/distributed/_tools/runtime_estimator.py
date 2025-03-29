# Owner(s): ["module: unknown"]
import math
import os
from collections import defaultdict
from enum import auto, Enum
from typing import Any, Callable, Optional
from typing_extensions import Self

import torch
import torch.utils._pytree as pytree
from torch._guards import active_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed import DeviceMesh
from torch.distributed._tools.comm_models import predict_communication
from torch.distributed._tools.common_utils import get_flattened_tensor
from torch.distributed._tools.compute_models import (
    learned_estimate_predictor,
    LEARNED_OPS,
    PYTORCH_MIN_ALLOCATE,
)
from torch.distributed._tools.fake_collectives import (
    collective_ops,
    CollectiveOp,
    non_functional_collectives,
    sync_ops,
)
from torch.distributed._tools.mod_tracker import ModTracker
from torch.distributed._tools.runest_utils import (
    CREATE_OPS,
    get_estimation_configs,
    OPS_TO_ALWAYS_SKIP,
    REDUCTION_OPS,
    resolve_gpu_type,
    VIEW_OPS,
)
from torch.distributed._tools.runtime_simulator import Resource, Simulator
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry


aten = torch.ops.aten

_MiB = 2**20

_VIEW_OR_CREATE_OPS = VIEW_OPS | CREATE_OPS

__all__ = ["RuntimeEstimator"]

class ComputeEstimate(Enum):
    BENCHMARK = auto()
    ANALYTICAL = auto()
    LEARNED = auto()

class CommEstimate(Enum):
    ANALYTICAL = auto()
    LEARNED = auto()

class RuntimeEstimator(TorchDispatchMode):
    """
    Estimates the GPU runtime in milliseconds using various estimation methods under the ``FakeTensorMode``.

    This class provides a ``TorchDispatchMode`` based context manager that can be used to estimate the eager
    runtime of PyTorch functions. It supports two estimation modes, benchmarking (`operator-level-benchmark`) and
    roofline cost modeling (`operator-level-cost-model`).
    For modules executed under this context manager, it agggregates the forward and backward operation runtimes
    and also records their execution orders.

    Attributes:
        mod_runtimes (dict[str, dict[str, float]]): A dictionary of module runtimes. The key to the outer dictionary
            is the fully qualified name (FQN) of the module. For each module the forward and backward runtimes of the
            operations are aggregated in the inner dictionary keyed by 'fw' and 'bw'.
        mod_fw_pre_order (list[str]): list of module FQNs in pre-forward execution order.
        mod_bw_pre_order (list[str]): list of module FQNs in pre-backward execution order.
        mod_fw_post_order (list[str]): list of module FQNs in post-forward execution order.
        mod_bw_post_order (list[str]): list of module FQNs in post-backward execution order.
        total_compute_time (float): The total estimated compute time in milliseconds.

    Note:
        1) The benchmarking estimate mode will execute kernels on GPU and assumes that every operation can run in
            isolation without causing an OOM error. It is also designed to be used only under ``FakeTensorMode``.
        2) We only estimate the compute time, if your code has communication, it will not be considered. Again, we will
            support this in future PRs.

    Example usage:

        .. code-block:: python

            runtime_estimator = RuntimeEstimator()
            with FakeTensorMode():
                module = ...
                optimizer = ...
                inp = ...
                with runtime_estimator(estimate_mode_type="operator-level-cost-model"):
                    loss = module(inp)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                runtime_estimator.display_modulewise_stats()
    """

    _float_types: set[torch.dtype] = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    _peak_flops_reg: dict[torch.dtype, float]
    _peak_flops_factors: dict[torch.dtype, float]
    _peak_bandwidth: float
    _no_fallback_kernel: set[torch._ops._OpNamespace] = set()
    _gpu_type: str = ""
    _fake_mode: FakeTensorMode
    estimate_mode_type: str
    compute_estimate: Callable
    pg_to_resource: dict[ProcessGroup, set[Resource]] = defaultdict(set)

    def __init__(self, simulate: bool = False) -> None:
        super().__init__()
        self._mod_tracker = ModTracker()
        self.mod_comptimes: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: 0.0)
        )
        self.mod_commtimes: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: 0.0)
        )
        self.mod_fw_pre_order: list[str] = []
        self.mod_bw_pre_order: list[str] = []
        self.mod_fw_post_order: list[str] = []
        self.mod_bw_post_order: list[str] = []
        self.total_compute_time: float = 0.0
        self.total_comm_time: float = 0.0
        self.simulated_time: float = 0.0
        self.simulator = None
        self.simulate = simulate
        if simulate:
            self.simulator = Simulator(RuntimeEstimator.pg_to_resource)

    @classmethod
    def init_configs(
        cls,
        gpu_type: str = "",
        custom_config: Optional[
            tuple[dict[torch.dtype, float], dict[torch.dtype, float], float]
        ] = None,
    ) -> None:
        """
        Initialize the configuration for the GPU type, including peak FLOPS, FLOPS factors, and bandwidth.

        Args:
            gpu_type (str, optional):
                The type of GPU to configure specific settings (e.g., "H100_SXM_80GB").
                Defaults to an empty string, which triggers automatic configuration based on the available GPU.
            custom_config (Optional[tuple[dict[torch.dtype, float], dict[torch.dtype, float], float]], optional):
                A tuple containing:
                    - A dictionary mapping `torch.dtype` to peak FLOPS (in GFLOPS/s).
                    - A dictionary mapping `torch.dtype` to peak FLOPS factors.
                    - The peak bandwidth (in GB/s).
                If provided, this overrides the default estimation based on the GPU type.

        Returns:
            None
        Raises:
            TypeError: If `runtime_kwargs` contains invalid types for any of the supported keys.
        """
        if gpu_type and not isinstance(gpu_type, str):
            raise TypeError(f"`gpu_type` must be a str, got {type(gpu_type).__name__}")
        if custom_config:
            if not (isinstance(custom_config, tuple) and len(custom_config) == 3):
                raise TypeError("`custom_config` must be a tuple of length 3")
            if not all(isinstance(custom_config[i], dict) for i in range(2)):
                raise TypeError(
                    "The first two elements of `custom_config` must be dictionaries"
                )
            if not isinstance(custom_config[2], float):
                raise TypeError("The third element of `custom_config` must be a float")
        cls._gpu_type = resolve_gpu_type(gpu_type)
        (
            cls._peak_flops_reg,
            cls._peak_flops_factors,
            cls._peak_bandwidth,
        ) = (
            get_estimation_configs(cls._gpu_type)
            if not custom_config
            else custom_config
        )

    @classmethod
    def init_pg_resources(cls, world_mesh: Optional[DeviceMesh]):
        if world_mesh is not None:
            _ndims = world_mesh.ndim
            assert _ndims <= 3, "Does not support > 3D mesh"
            _mesh_dim_names = world_mesh.mesh_dim_names
            if _ndims == 3:
                # Assume HSDP + TP
                tp_pg = world_mesh.get_group("tp")
                dp_replicate_pg = world_mesh.get_group("dp_replicate")
                dp_shard_pg = world_mesh.get_group("dp_shard")

                cls.pg_to_resource[tp_pg].add(Resource.INTRA_COMM)
                cls.pg_to_resource[dp_replicate_pg].add(Resource.INTER_COMM)
                cls.pg_to_resource[dp_shard_pg].add(Resource.INTER_COMM)

            elif _ndims == 2:
                # Can be HSDP or FSDP + TP
                if "tp" in _mesh_dim_names:
                    # Assume FSDP + TP
                    tp_pg = world_mesh.get_group("tp")
                    dp_pg = world_mesh.get_group("dp")
                    cls.pg_to_resource[tp_pg].add(Resource.INTRA_COMM)
                    cls.pg_to_resource[dp_pg].add(Resource.INTER_COMM)
                else:
                    # Assume HSDP
                    dp_replicate_pg = world_mesh.get_group("dp_replicate")
                    dp_shard_pg = world_mesh.get_group("dp_shard")
                    cls.pg_to_resource[dp_replicate_pg].add(Resource.INTER_COMM)
                    cls.pg_to_resource[dp_shard_pg].update([Resource.INTER_COMM, Resource.INTER_COMM])

            else:
                # Assume FSDP
                dp_pg = world_mesh.get_group("dp")
                cls.pg_to_resource[dp_pg].update([Resource.INTER_COMM, Resource.INTER_COMM])

        _default_pg = torch.distributed.group.WORLD
        if _default_pg is not None and _default_pg not in cls.pg_to_resource:
            cls.pg_to_resource[_default_pg].update([Resource.INTER_COMM, Resource.INTER_COMM])

    # Adapted from: https://github.com/pytorch/pytorch/blob/9b902b3ee3bd608a19543362b66bf06c373dd374/torch/_subclasses/fake_tensor.py#L1969  # noqa: PGH004,B950
    # NB: returns fake tensors
    @classmethod
    def _maybe_run_and_benchmark_fallback_kernel(  # type: ignore[no-untyped-def]
        cls,
        func,
        args,
        kwargs,
        orig_not_implemented_exception,
    ):
        """
        Runs and benchmarks a fallback kernel for a given function.

        Args:
            func (Callable): The function to benchmark.
            args (tuple): The arguments to pass to the function.
            kwargs (dict[str, Any]): The keyword arguments to pass to the function.
            orig_not_implemented_exception (Exception): The original exception to raise if the fallback kernel
                is not implemented.

        Returns:
            tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
        # these should all be supported, just to be safe
        # avoid fallback for operators which inplace modify metadata
        # because the input fake tensors would be umodified
        if torch.Tag.inplace_view in func.tags:  # type: ignore[attr-defined]
            raise orig_not_implemented_exception

        inp_impls = {}
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))
        # Don't use in_kernel_invocation_manager(fake_mode) as we want to do
        # REAL compute (not with meta device)
        with no_dispatch():

            def to_real_tensor(e):  # type: ignore[no-untyped-def]
                if cls._fake_mode.is_our_fake(e):
                    if e.dtype in cls._float_types:
                        out = torch.rand_like(e, device=e.fake_device)
                    else:
                        out = torch.ones_like(e, device=e.fake_device)
                    if e.is_sparse:
                        out._coalesced_(e.is_coalesced())
                    inp_impls[id(out)] = e
                    return out
                return e

            flat_args = [to_real_tensor(a) for a in flat_args]
            args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
            r = func(*args, **kwargs)
            warmup_iters, actual_iters = 2, 3
            for _ in range(warmup_iters):
                func(*args, **kwargs)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream())
            for _ in range(actual_iters):
                func(*args, **kwargs)
            end_event.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            cuda_time = start_event.elapsed_time(end_event)
            mean_op_time = cuda_time / actual_iters

        storages = set()

        for e in flat_args:
            if isinstance(e, torch.Tensor):
                if not e.is_sparse:
                    storages.add(e._typed_storage()._cdata)

        # TODO: also check metadata change on inputs
        # proper aliasing/metadata relationship between outputs and inputs will
        # not be set up, bc of conversion to device, unless we can reuse an
        # input impl

        def map_out(e):  # type: ignore[no-untyped-def]
            if id(e) not in inp_impls and (
                isinstance(e, torch.Tensor)
                and not e.is_sparse
                and e._typed_storage()._cdata in storages
            ):
                raise orig_not_implemented_exception

            if isinstance(e, torch.Tensor):
                if id(e) in inp_impls:
                    return inp_impls[id(e)]
                else:
                    return cls._fake_mode.fake_tensor_converter.from_real_tensor(
                        cls._fake_mode, e
                    )
            else:
                return e

        return (pytree.tree_map(map_out, r), mean_op_time)

    @classmethod
    def _benchmark_estimate(cls, func, args, kwargs) -> tuple[Any, float]:  # type: ignore[no-untyped-def]
        """
        Estimates the runtime of a function using benchmarking.

        Args:
            func: The function to estimate.
            args: The arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.
            res: The result of the function.

        Returns:
            tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
        assert isinstance(cls.fake_mode, FakeTensorMode), (
            "Initialize/Assign FakeTensorMode before using this function"
        )
        mean_op_time = 0.0
        if func._overloadpacket not in _VIEW_OR_CREATE_OPS:
            try:
                res, mean_op_time = cls._maybe_run_and_benchmark_fallback_kernel(
                    func,
                    args,
                    kwargs,
                    NotImplementedError,
                )
                return (res, mean_op_time)
            except NotImplementedError:
                cls._no_fallback_kernel.add(func._overloadpacket)
        res = func(*args, **kwargs or {})
        return (res, mean_op_time)

    @classmethod
    def _desugar_args_kwargs_out(cls, args, kwargs, out) -> tuple[Any, Any, Any]:  # type: ignore[no-untyped-def]
        desugared_args = pytree.tree_map_only(torch.Tensor, get_flattened_tensor, args)
        desugared_kwargs = pytree.tree_map_only(
            torch.Tensor, get_flattened_tensor, kwargs
        )
        desugared_out = pytree.tree_map_only(torch.Tensor, get_flattened_tensor, out)
        return (desugared_args, desugared_kwargs, desugared_out)

    # Adapted from: https://github.com/pytorch/pytorch/blob/9b902b3ee3bd608a19543362b66bf06c373dd374/torch/_inductor/scheduler.py#L589  # noqa: PGH004,B950
    @classmethod
    def _roofline_or_learned_estimate(cls, func, args, kwargs) -> tuple[Any, float]:  # type: ignore[no-untyped-def]
        """
        Estimates the runtime of a function using a roofline and/or learned cost model.

        Args:
            func: The function to estimate.
            args: The arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.
            out: The output of the function.

        Returns:
            tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
        assert torch.cuda.is_available(), (
            "Roofline estimation needs to access CUDA capabilities to make estimations"
        )

        def get_num_bytes(t: torch.Tensor) -> int:
            """
            Calculates the memory consumption of a tensor.

            Args:
                t (torch.Tensor): The input tensor.

            Returns:
                int: The memory consumption of the tensor in bytes.
            """
            num_bytes = t.untyped_storage().nbytes()
            mem_consumed = (
                math.ceil(num_bytes / PYTORCH_MIN_ALLOCATE) * PYTORCH_MIN_ALLOCATE
            )
            return mem_consumed

        def get_compute_time(func_packet, args, kwargs, out, out_dtypes) -> float:  # type: ignore[no-untyped-def]
            """
            Estimates the compute time of an aten operator.

            Args:
                func_packet: The operator overload packet.
                args: The arguments to the operator.
                kwargs: The keyword arguments to the operator.
                out: The output of the operator.
                out_dtypes: The output data types.

            Returns:
                float: The estimated compute time in nanoseconds.
            """
            if func_packet in flop_registry:
                dtype = min(out_dtypes, key=lambda x: x.itemsize)
                # This gives GFLOPS/sec for the given dtype
                peak_gpu_flops = cls._peak_flops_reg[dtype]
                # factor determines the peak flops that are empirically attained by compute ops
                factor = cls._peak_flops_factors[dtype]
                peak_empirical_flops = factor * peak_gpu_flops
                flop_count_func = flop_registry[func_packet]
                # We divide by a factor of 2 to get the MACs (multiply and accumulate)
                flop_count = flop_count_func(*args, **kwargs, out_val=out) / 2
                # FLOPS/(GFLOPS/sec) gives us time in nanoseconds
                compute_time = flop_count / peak_empirical_flops
                return compute_time
            return 0.0

        def get_transfer_time(func_packet, flat_args_kwargs, flat_outs) -> float:  # type: ignore[no-untyped-def]
            """
            Estimates the memory transfer time of input and output tensors.

            Args:
                flat_args_kwargs (list[torch.Tensor]): The flat list of arguments and keyword arguments.
                flat_outs (list[torch.Tensor]): The flat list of outputs.

            Returns:
                float: The estimated memory transfer time in nanoseconds.
            """
            # The GPU memory bandwidth is in GB/s
            gpu_memory_bandwidth = cls._peak_bandwidth
            read_bytes = sum(
                get_num_bytes(t)
                for t in flat_args_kwargs
                if isinstance(t, torch.Tensor)
            )
            write_bytes = sum(
                get_num_bytes(t) for t in flat_outs if isinstance(t, torch.Tensor)
            )
            counted_bytes = read_bytes + write_bytes
            # The GPU memory bandwidth is in GB/s so the transfer time is in nanoseconds
            transfer_time = counted_bytes / gpu_memory_bandwidth
            if func_packet in REDUCTION_OPS:
                transfer_time *= 2
            return transfer_time

        # Roofline Cost Model Explanation

        # The roofline cost model estimates the execution time of an operator based on
        # the device's empirical maximum FLOPs/sec (pi) and device DRAM bandwidth (beta).

        # Variables:
        # - pi: Maximum empirical FLOPs/sec of the device
        # - beta: Maximum empirical device DRAM bandwidth (bytes/sec) of the device
        # - I: Arithmetic intensity of the operator (FLOPs/bytes)
        # - op_flops: FLOPs required by the operator
        # - op_bytes: Bytes transferred to and from DRAM for the operator

        # Calculation Steps:
        # 1. Calculate arithmetic intensity: I = op_flops / op_bytes
        # 2. Calculate estimated FLOPs/sec: est_flops_sec = min(pi, beta * I)
        # 3. Calculate estimated operator time: estimated_op_time = op_flops / est_flops_sec
        #    This simplifies to: estimated_op_time = max(op_flops / pi, op_flops / (beta * I))
        #    Further simplifying: estimated_op_time = max(op_flops / pi, op_bytes / beta)

        # Simplified Formulas:
        # - compute_time = op_flops / pi
        # - transfer_time = op_bytes / beta
        # - estimated_op_time = max(compute_time, transfer_time)

        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        op_time = 0.0
        func_packet = func._overloadpacket
        if func_packet not in _VIEW_OR_CREATE_OPS:
            desugared_args, desugared_kwargs, desugared_out = (
                cls._desugar_args_kwargs_out(args, kwargs, out)
            )

            flat_args_kwargs, _ = pytree.tree_flatten(
                (desugared_args, desugared_kwargs)
            )
            flat_outs, _ = pytree.tree_flatten(desugared_out)
            out_dtypes = {
                t.dtype for t in flat_outs if isinstance(t, torch.Tensor)
            } & cls._float_types
            if (
                cls.estimate_mode_type == "operator-level-learned-model"
                and func_packet in LEARNED_OPS
            ):
                op_time = learned_estimate_predictor(
                    func_packet,
                    desugared_args,
                    desugared_kwargs,
                    desugared_out,
                    out_dtypes,
                    cls._gpu_type,
                )
            else:
                transfer_time = (
                    get_transfer_time(func_packet, flat_args_kwargs, flat_outs) / 1.5
                )

                compute_time = get_compute_time(
                    func_packet,
                    desugared_args,
                    desugared_kwargs,
                    desugared_out,
                    out_dtypes,
                )
                # We get the estimated time as the max of the transfer time and
                # compute time. We divide by 1e6 to get the time in ms
                op_time = max(transfer_time, compute_time) / 1e6

        return (out, op_time)

    @classmethod
    def comm_estimate(cls, func, args, kwargs) -> tuple[Any, float]:
        kwargs = kwargs if kwargs else {}
        op_time = 0.0
        res = func(*args, **kwargs)
        if func not in sync_ops:
            desugared_args, desugared_kwargs, desugared_res = (
                cls._desugar_args_kwargs_out(args, kwargs, res)
            )
            process_group = CollectiveOp.get_process_group(func, desugared_args)
            comm_size = CollectiveOp.get_comm_tensor_size(
                func, desugared_res, desugared_args, desugared_kwargs
            )
            num_ranks = process_group.size()
            comm_resource = cls.pg_to_resource[process_group]
            internode_only = (comm_resource == {Resource.INTER_COMM})
            comm_size_mib = comm_size / _MiB
            analytical_mode = not (
                cls.estimate_mode_type == "operator-level-learned-model"
            )
            op_time = predict_communication(
                func, comm_size_mib, num_ranks, internode_only, analytical_mode
            )
        return (res, op_time)

    @classmethod
    def runtime_estimate(cls, func, args, kwargs) -> tuple[Any, float]:
        if func in collective_ops:
            res, op_time = cls.comm_estimate(func, args, kwargs)
        else:
            res, op_time = cls.compute_estimate(func, args, kwargs)
        return res, op_time

    def display_modulewise_stats(self, depth: int = 2) -> None:
        """
        Displays module-wise statistics collected by ``RuntimeEstimator``.

        Prints the pre-forward and pre-backward execution orders.
        Displays the module-wise forward and backward runtimes in milliseconds.

        Args:
            depth (int): The maximum depth of module hierarchy to display (default to 2).
        """
        print("Pre-Forward Execution Order: ")
        for mod_fqn in self.mod_fw_pre_order:
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(mod_fqn)
        print("Pre-Backward Execution Order: ")
        for mod_fqn in self.mod_bw_pre_order:
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(mod_fqn)
        for mod_fqn, runtimes in self.mod_comptimes.items():
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(
                f"{mod_fqn} fw: {runtimes.get('fw', 0.0):.3f}ms bw: {runtimes.get('bw', 0.0):.3f}ms"
            )

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):  # type: ignore[no-untyped-def]
        res, op_time = RuntimeEstimator.runtime_estimate(func, args, kwargs)
        # if func._overloadpacket not in OPS_TO_ALWAYS_SKIP:
        #     print(f"{func._overloadpacket}: {op_time:.3f}")
        is_comm_op = func in collective_ops
        for par in self._mod_tracker.parents:
            if self._mod_tracker.is_bw:
                if is_comm_op:
                    self.mod_commtimes[par]["bw"] += op_time
                else:
                    self.mod_comptimes[par]["bw"] += op_time
            else:
                if is_comm_op:
                    self.mod_commtimes[par]["fw"] += op_time
                else:
                    self.mod_comptimes[par]["fw"] += op_time
        if is_comm_op:
            self.total_comm_time += op_time
        else:
            self.total_compute_time += op_time

        if self.simulate and func not in OPS_TO_ALWAYS_SKIP:
            self.simulator.record_op(func, args, kwargs, res)
        return res

    def __call__(
        self,
        estimate_mode_type: str,
        gpu_type: str = "",
        custom_config: Optional[
            tuple[dict[torch.dtype, float], dict[torch.dtype, float], float]
        ] = None,
        world_mesh: Optional[DeviceMesh] = None,
    ) -> Self:
        """
        Configures the runtime estimation mode and initializes GPU-specific configurations.

        Supported Modes:
            - `"operator-level-benchmark"`: Estimates runtime using operator benchmarking.
            - `"operator-level-cost-model"`: Estimates runtime using a roofline cost model.
            - `"operator-level-learned-model"`: Estimates runtime using a learned cost model.


        Args:
            estimate_mode_type (str):
                The runtime estimation mode to use. Must be one of the supported modes.
            gpu_type (str, optional):
                The GPU type to configure specific settings (e.g., `"H100_SXM_80GB"`).
                Defaults to an empty string, which triggers automatic configuration based on the available GPU.
            custom_config (Optional[tuple[dict[torch.dtype, float], dict[torch.dtype, float], float]], optional):
                A tuple containing:
                    - A dictionary mapping `torch.dtype` to peak FLOPS (in GFLOPS/s).
                    - A dictionary mapping `torch.dtype` to peak FLOPS factors.
                    - The peak bandwidth (in GB/s).
                If provided, this overrides the default estimation based on the GPU type.

        Returns:
            Self:
                The current instance of `RuntimeEstimator` with the configured estimation mode.

        Raises:
            NotImplementedError:
                If `estimate_mode_type` is not a supported runtime estimation mode.
        """
        if estimate_mode_type == "operator-level-benchmark":
            RuntimeEstimator.compute_estimate = RuntimeEstimator._benchmark_estimate
        elif estimate_mode_type in [
            "operator-level-cost-model",
            "operator-level-learned-model",
        ]:
            RuntimeEstimator.compute_estimate = (
                RuntimeEstimator._roofline_or_learned_estimate
            )
        else:
            raise NotImplementedError(
                f"estimate_mode_type {estimate_mode_type} not supported"
            )
        RuntimeEstimator.estimate_mode_type = estimate_mode_type
        RuntimeEstimator.init_configs(gpu_type, custom_config)
        RuntimeEstimator.init_pg_resources(world_mesh)
        return self

    def __enter__(self) -> Self:
        fake_mode = active_fake_mode()
        assert isinstance(fake_mode, FakeTensorMode), (
            "No FakeTensorMode found, designed to used under FakeTensorMode"
        )
        RuntimeEstimator.fake_mode = fake_mode
        self.total_compute_time = 0.0
        self.total_comm_time = 0.0
        self.mod_comptimes = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.mod_commtimes = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.mod_fw_pre_order.clear()
        self.mod_bw_pre_order.clear()
        self.mod_fw_post_order.clear()
        self.mod_bw_post_order.clear()
        self._mod_tracker.register_user_hooks(
            pre_fw_hook=lambda mod, inp: self.mod_fw_pre_order.append(
                self._mod_tracker.get_known_fqn(mod)
            ),
            pre_bw_hook=lambda mod, g_out: self.mod_bw_pre_order.append(
                self._mod_tracker.get_known_fqn(mod) if mod is not None else ""
            ),
            post_fw_hook=lambda mod, inp, out: self.mod_fw_post_order.append(
                self._mod_tracker.get_known_fqn(mod)
            ),
            post_bw_hook=lambda mod, g_inp: self.mod_bw_post_order.append(
                self._mod_tracker.get_known_fqn(mod) if mod is not None else ""
            ),
        )
        if self.simulate:
            self.simulator.capture_sync_ops()
        self._mod_tracker.__enter__()
        super().__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if len(self._no_fallback_kernel) > 0:
            print("no_fallback_kernel: ", list(self._no_fallback_kernel))
        if self.simulate:
            self.simulated_time = self.simulator.simulate()
            self.simulator.restore_sync_ops()
        super().__exit__(*args)
        self._mod_tracker.clear_user_hooks()
        self._mod_tracker.__exit__()
