# Owner(s): ["module: unknown"]
import math
import os
import joblib
import subprocess
import numpy as np
import pandas as pd
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Set, Tuple
from typing_extensions import Self
import warnings
from math import prod
from torch.distributed._tools.run_est_utils import get_peak_flops_registry, peak_factors, get_flattened_tensor

import torch
import torch.utils._pytree as pytree
from torch._guards import active_fake_mode
from torch._inductor.utils import get_device_tflops, get_gpu_dram_gbps
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.mod_tracker import ModTracker
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry


aten = torch.ops.aten

# This value is hard-coded here:
# https://github.com/pytorch/pytorch/blob/5fba5d83f0703ff8077ab65448a998e9ad6598fd/c10/cuda/CUDACachingAllocator.cpp#L117
_PYTORCH_MIN_ALLOCATE = (
    2**9 if int(os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", 0)) == 0 else 1
)

# No fall-back kernel needed/exists for view ops
_VIEW_OPS = {
    aten.lift_fresh,
    aten.t,
    aten.transpose,
    aten.view,
    aten.detach,
    aten._unsafe_view,
    aten.split,
    aten.adjoint,
    aten.as_strided,
    aten.diagonal,
    aten.expand,
    aten.expand_as,
    aten.movedim,
    aten.permute,
    aten.select,
    aten.squeeze,
    aten.mT,
    aten.mH,
    aten.real,
    aten.imag,
    aten.view_as,
    aten.unflatten,
    aten.unfold,
    aten.unbind,
    aten.unsqueeze,
    aten.vsplit,
    aten.hsplit,
    aten.split_with_sizes,
    aten.swapaxes,
    aten.swapdims,
    aten.chunk,
}
# We can ignore benchmarking tensor create ops
_CREATE_OPS = {
    aten.randint,
    aten.randn,
    aten.rand,
    aten.randn_like,
    aten.rand_like,
    aten.randint_like,
    aten.arange,
    aten.ones_like,
    aten.zeros_like,
    aten.as_strided_,
}

_IGNORE_OPS = _VIEW_OPS | _CREATE_OPS

# Similar to `flop_registry`, stores the operators that have learned predictors
_LEARNED_OPS: Dict[Any, Any] = {}

# Caches the learned models that predict ops' runtimes.
_LEARNED_OPS_PREDICTORS: Dict[str, Any] = {}


__all__ = ["RuntimeEstimator"]


def get_learned_model(op: str, gpu_type: str) -> Any:
    if op not in _LEARNED_OPS_PREDICTORS:
        base_dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(base_dir, f"{gpu_type}_models", f"{op}.joblib")

        _LEARNED_OPS_PREDICTORS[op] = joblib.load(path)
    return _LEARNED_OPS_PREDICTORS[op]

from functools import wraps
from torch.utils._pytree import tree_map

def get_shape(i):
    if isinstance(i, torch.Tensor):
        return i.shape
    return i

def shape_wrapper(f):
    """
    Similar to flop_counter.shape_wrapper(), but also takes takes gflops.
    """
    @wraps(f)
    def nf(gpu_type, dtype, gflops, *args, out_val=None, **kwargs):
        args, kwargs, out_shape = tree_map(get_shape, (args, kwargs, out_val))
        return f(gpu_type, dtype, gflops, *args, out_shape=out_shape, **kwargs)
    return nf

def register_timing_formula(targets, get_raw=False):
    """
    Similar to flop_counter.register_flop_formula().
    """
    def register_fun(time_formula):
        if not get_raw:
            time_formula = shape_wrapper(time_formula)

        def register(target):
            if not isinstance(target, torch._ops.OpOverloadPacket):
                raise ValueError(
                    f"register_flop_formula(targets): expected each target to be "
                    f"OpOverloadPacket (i.e. torch.ops.mylib.foo), got "
                    f"{target} which is of type {type(target)}")
            if target in _LEARNED_OPS:
                raise RuntimeError(f"duplicate registrations for {target}")
            _LEARNED_OPS[target] = time_formula

        # To handle allowing multiple aten_ops at once
        torch.utils._pytree.tree_map_(register, targets)

        return time_formula

    return register_fun

def convert_dtype(dtype) -> Dict[str, int]:
    """
    Convert dtype to a one-hot encoding as a pandas Series.
    
    Learned model supports the dtypes:
        - torch.float16
        - torch.float32
        - torch.bfloat16
    """
    dtypes = [torch.float16, torch.float32, torch.bfloat16]
    dtype_one_hot = [1 if dtype == d else 0 for d in dtypes]
    dtype_names = ["dtype_16", "dtype_32", "dtype_b16"]
    return dict(zip(dtype_names, dtype_one_hot))

@register_timing_formula(aten.mm)
def mm_time(gpu_type, dtype, gflops, a_shape, b_shape, *args, out_shape=None, **kwargs) -> float:
    model = get_learned_model("mm", gpu_type)

    m, n = a_shape
    n2, p = b_shape
    assert n == n2

    nm = n * m
    mp = m * p
    np = n * p
    intensity = (gflops * 1e9) / (nm + mp + np)

    dtypes = convert_dtype(dtype)
    features = {
        "n": n,
        "m": m,
        "p": p,
        "gflops": gflops,
        "nm": nm,
        "mp": mp,
        "np": np,
        "intensity": intensity,
        "dtype_16": dtypes["dtype_16"],
        "dtype_32": dtypes["dtype_32"],
        "dtype_b16": dtypes["dtype_b16"],
    }
    
    features_df = pd.DataFrame([features])
    return float(model.predict(features_df)[0])

@register_timing_formula(aten.addmm)
def addmm_time(gpu_type, dtype, gflops, self_shape, a_shape, b_shape, out_shape=None, **kwargs) -> float:
    return mm_time(gpu_type, dtype, gflops, a_shape, b_shape)

@register_timing_formula(aten.bmm)
def bmm_time(gpu_type, dtype, gflops, a_shape, b_shape, out_shape=None, **kwargs) -> float:
    model = get_learned_model("bmm", gpu_type)

    b, m, n = a_shape
    b2, n2, p = b_shape
    assert b == b2 and n == n2
    
    bnm = b * n * m
    bmp = b * m * p
    bnp = b * n * p
    intensity = (gflops * 1e9) / (bnm + bmp + bnp)
    
    dtypes = convert_dtype(dtype)
    features = {
        "b": b,
        "n": n,
        "m": m,
        "p": p,
        "gflops": gflops,
        "bnm": bnm,
        "bmp": bmp,
        "bnp": bnp,
        "intensity": intensity,
        "dtype_16": dtypes["dtype_16"],
        "dtype_32": dtypes["dtype_32"],
        "dtype_b16": dtypes["dtype_b16"],
    }
    features_df = pd.DataFrame([features])
    return float(model.predict(features_df)[0])

@register_timing_formula(aten.baddbmm)
def baddbmm_time(gpu_type, dtype, gflops, self_shape, a_shape, b_shape, out_shape=None, **kwargs) -> float:
    return bmm_time(gpu_type, dtype, gflops, a_shape, b_shape)

def is_causal_sdpa(args: tuple) -> bool:
    """
    TODO: the way that flop_counter implements sdpa args/kwargs—namely, `is_causal`—should be updated.
    This is a heuristic hackaround.
    """
    if len(args) >= 2 and args[0] is not None:
        return True
    if len(args) > 2 and isinstance(args[-1], bool):
        return args[-1]
    return False

def build_sdpa_features(b, h, s_q, s_kv, d_qk, d_v, gflops, dtype, backend, is_causal: bool) -> pd.DataFrame:
    if backend == "cudnn":
        backends_ohe = [1, 0, 0]
    elif backend == "efficient":
        backends_ohe = [0, 1, 0]
    elif backend == "flash":
        backends_ohe = [0, 0, 1]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    dtypes = convert_dtype(dtype)
    is_causal_ohe = [0, 1] if is_causal else [1, 0]
    
    q = b * h * s_q * d_qk
    k = b * h * s_kv * d_qk
    v = b * h * s_kv * d_v
    output = b * h * s_q * d_v
    memory_accesses = q + k + v + output
    intensity = (gflops * 1e9) / memory_accesses

    features = {
        "b": b,
        "h": h,
        "s_q": s_q,
        "s_kv": s_kv,
        "d_qk": d_qk,
        "d_v": d_v,
        "gflops": gflops,
        "q": q,
        "k": k,
        "v": v,
        "output": output,
        "intensity": intensity,
        "dtype_16": dtypes["dtype_16"],
        "dtype_32": dtypes["dtype_32"],
        "dtype_b16": dtypes["dtype_b16"],
        "backend_cudnn": backends_ohe[0],
        "backend_efficient": backends_ohe[1],
        "backend_flash": backends_ohe[2],
        "is_causal_0": is_causal_ohe[0],
        "is_causal_1": is_causal_ohe[1]
    }
    return pd.DataFrame([features])


def check_sdpa_shapes(query_shape, key_shape, value_shape):
    b, h, s_q, d_qk = query_shape
    _b2, _h2, s_kv, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape
    assert b == _b2 == _b3 and h == _h2 == _h3 and d_qk == _d2 and s_kv == _s3 and d_qk == _d2
    return b, h, s_q, s_kv, d_qk, d_v

def check_sdpa_shapes_backward(query_shape, key_shape, value_shape, grad_out_shape):
    b, h, s_q, d_qk = query_shape
    _b2, _h2, s_kv, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape
    _b4, _h4, _s4, _d4 = grad_out_shape
    assert b == _b2 == _b3 == _b4 and h == _h2 == _h3 == _h4 and d_qk == _d2 and d_v == _d4 and s_kv == _s3 and s_q == _s4
    return b, h, s_q, s_kv, d_qk, d_v

@register_timing_formula(aten._scaled_dot_product_cudnn_attention)
def sdpa_cudnn_time(gpu_type, dtype, gflops, query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> float:
    model = get_learned_model("sdpa", gpu_type)
    b, h, s_q, s_kv, d_qk, d_v = check_sdpa_shapes(query_shape, key_shape, value_shape)
    features = build_sdpa_features(b, h, s_q, s_kv, d_qk, d_v, gflops, dtype, "cudnn", is_causal=is_causal_sdpa(args))
    return float(model.predict(features)[0])

@register_timing_formula(aten._scaled_dot_product_efficient_attention)
def sdpa_efficient_time(gpu_type, dtype, gflops, query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> float:
    model = get_learned_model("sdpa", gpu_type)
    b, h, s_q, s_kv, d_qk, d_v = check_sdpa_shapes(query_shape, key_shape, value_shape)
    features = build_sdpa_features(b, h, s_q, s_kv, d_qk, d_v, gflops, dtype, "efficient", is_causal=is_causal_sdpa(args))
    return float(model.predict(features)[0])

@register_timing_formula(aten._scaled_dot_product_flash_attention)
def sdpa_flash_time(gpu_type, dtype, gflops, query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> float:
    model = get_learned_model("sdpa", gpu_type)
    b, h, s_q, s_kv, d_qk, d_v = check_sdpa_shapes(query_shape, key_shape, value_shape)
    features = build_sdpa_features(b, h, s_q, s_kv, d_qk, d_v, gflops, dtype, "flash", is_causal=is_causal_sdpa(args))
    return float(model.predict(features)[0])

@register_timing_formula(aten._scaled_dot_product_cudnn_attention_backward)
def sdpa_backward_cudnn_time(gpu_type, dtype, gflops, grad_out_shape, query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> float:
    model = get_learned_model("sdpa_backward", gpu_type)
    b, h, s_q, s_kv, d_qk, d_v = check_sdpa_shapes_backward(query_shape, key_shape, value_shape, grad_out_shape)
    features = build_sdpa_features(b, h, s_q, s_kv, d_qk, d_v, gflops, dtype, "cudnn", is_causal=is_causal_sdpa(args))
    return float(model.predict(features)[0])

@register_timing_formula(aten._scaled_dot_product_efficient_attention_backward)
def sdpa_backward_efficient_time(gpu_type, dtype, gflops, grad_out_shape, query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> float:
    model = get_learned_model("sdpa_backward", gpu_type)
    b, h, s_q, s_kv, d_qk, d_v = check_sdpa_shapes_backward(query_shape, key_shape, value_shape, grad_out_shape)
    features = build_sdpa_features(b, h, s_q, s_kv, d_qk, d_v, gflops, dtype, "efficient", is_causal=is_causal_sdpa(args))
    return float(model.predict(features)[0])

@register_timing_formula(aten._scaled_dot_product_flash_attention_backward)
def sdpa_backward_flash_time(gpu_type, dtype, gflops, grad_out_shape, query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> float:
    model = get_learned_model("sdpa_backward", gpu_type)
    b, h, s_q, s_kv, d_qk, d_v = check_sdpa_shapes_backward(query_shape, key_shape, value_shape, grad_out_shape)
    features = build_sdpa_features(b, h, s_q, s_kv, d_qk, d_v, gflops, dtype, "flash", is_causal=is_causal_sdpa(args))
    return float(model.predict(features)[0])


def build_conv2d_features(gflops, dtype, x_shape, w_shape, _stride, _padding, _dilation, transposed, args, out_shape) -> pd.DataFrame:
    dtypes = convert_dtype(dtype)

    b, in_channels, iH, iW = x_shape
    _b, out_channels, oH, oW = out_shape

    assert b == _b, "Batch dimension doesn't match in input and output"

    if transposed:
        in_channels, out_channels_over_groups, *filter_size = w_shape
        groups = out_channels // out_channels_over_groups
    else:
        out_channels, in_channels_over_groups, *filter_size = w_shape
        groups = in_channels // in_channels_over_groups

    if len(filter_size) != 2:
        return None
    kH, kW = filter_size

    input_memory_accesses = b * in_channels * iH * iW
    kernel_memory_accesses = out_channels * (in_channels // groups) * kH * kW
    output_memory_accesses = b * out_channels * oH * oW
    memory_accesses = input_memory_accesses + kernel_memory_accesses + output_memory_accesses
    intensity = (gflops * 1e9) / memory_accesses

    stride = _stride if isinstance(_stride, int) else _stride[0]
    dilation = _dilation if isinstance(_dilation, int) else _dilation[0]

    features = {
        "b": b,
        "in_channels": in_channels,
        "iH": iH,
        "iW": iW,
        "out_channels": out_channels,
        "groups": groups,
        "kH": kH,
        "kW": kW,
        "stride": stride,
        "dilation": dilation,
        "oH": oH,
        "oW": oW,
        "gflops": gflops,
        "input": input_memory_accesses,
        "kernel": kernel_memory_accesses,
        "output": output_memory_accesses,
        "intensity": intensity,
        "dtype_16": dtypes["dtype_16"],
        "dtype_32": dtypes["dtype_32"],
        "dtype_b16": dtypes["dtype_b16"],
        "transposed_0": not transposed,
        "transposed_1": transposed,
    }
    return pd.DataFrame([features])


@register_timing_formula([aten.convolution, aten._convolution])
def conv_time(gpu_type, dtype, gflops, x_shape, w_shape, _bias, _stride, _padding, _dilation, transposed, *args, out_shape=None, **kwargs) -> float:
    """Only supports Conv2D for now."""
    if transposed:
        model = get_learned_model("conv", gpu_type)
    else:
        model = get_learned_model("conv_t", gpu_type)
    features = build_conv2d_features(gflops, dtype, x_shape, w_shape, _stride, _padding, _dilation, transposed, args, out_shape)
    return float(model.predict(features)[0])


@register_timing_formula(aten.convolution_backward)
def conv_backward_time(
    gpu_type,
    dtype,
    gflops,
    grad_out_shape,
    x_shape,
    w_shape,
    _bias,
    _stride,
    _padding,
    _dilation,
    transposed,
    _output_padding,
    _groups,
    output_mask,
    out_shape) -> float:
    """
    TODO: need to add support for higher dims.
    """
    args = None
    if transposed:
        model = get_learned_model("conv_backward", gpu_type)
    else:
        model = get_learned_model("conv_t_backward", gpu_type)
    features = build_conv2d_features(gflops, dtype, x_shape, w_shape, _stride, _padding, _dilation, transposed, args, out_shape)
    return float(model.predict(features)[0])


class RuntimeEstimator(TorchDispatchMode):
    """
    Estimates the GPU runtime in milliseconds using various estimation methods under the ``FakeTensorMode``.

    This class provides a ``TorchDispatchMode`` based context manager that can be used to estimate the eager
    runtime of PyTorch functions. It supports two estimation modes, benchmarking (`operator-level-benchmark`) and
    roofline cost modeling (`operator-level-cost-model`).
    For modules executed under this context manager, it agggregates the forward and backward operation runtimes
    and also records their execution orders.

    Attributes:
        mod_runtimes (Dict[str, Dict[str, float]]): A dictionary of module runtimes. The key to the outer dictionary
            is the fully qualified name (FQN) of the module. For each module the forward and backward runtimes of the
            operations are aggregated in the inner dictionary keyed by 'fw' and 'bw'.
        mod_fw_pre_order (List[str]): List of module FQNs in pre-forward execution order.
        mod_bw_pre_order (List[str]): List of module FQNs in pre-backward execution order.
        mod_fw_post_order (List[str]): List of module FQNs in post-forward execution order.
        mod_bw_post_order (List[str]): List of module FQNs in post-backward execution order.
        total_runtime (float): The total estimated runtime in milliseconds.

    Note:
        1) The benchmarking estimate mode will execute kernels on GPU and assumes that every operation can run in
            isolation without causing an OOM error. It is also designed to be used only under ``FakeTensorMode``.
        2) Currently wrapper tensor sub-classes such as ``DTensor`` won't produce correct estimates. We plan to support
            them in future PRs.
        3) We only estimate the compute time, if your code has communication, it will not be considered. Again, we will
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

    _float_types: Set[torch.dtype] = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    _no_fallback_kernel: Set[torch._ops._OpNamespace] = set()
    fake_mode: FakeTensorMode
    _peak_flops_reg: Dict[torch.dtype, int] = {}
    _factors: Dict[torch.dtype, float] = {}
    gpu_types: Dict[int, str] = {}
    count = {}

    def __init__(self) -> None:
        super().__init__()
        self._estimate: Callable
        self._estimate_mode_type: str
        self._mod_tracker = ModTracker()
        self.mod_runtimes: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: 0.0)
        )
        self.mod_fw_pre_order: List[str] = []
        self.mod_bw_pre_order: List[str] = []
        self.mod_fw_post_order: List[str] = []
        self.mod_bw_post_order: List[str] = []
        self.total_compute_time: float = 0.0

        gpu_id = torch.cuda.current_device()  # Get the current GPU ID
        if gpu_id not in RuntimeEstimator.gpu_types:
            RuntimeEstimator.gpu_types[gpu_id] = RuntimeEstimator.get_device_type()  # Initialize gpu_type for the GPU
        self.gpu_type = RuntimeEstimator.gpu_types[gpu_id]  # Assign gpu_type based on the current GPU
        RuntimeEstimator._factors = peak_factors[self.gpu_type]
        print(self.gpu_type)
    
    @classmethod
    def get_device_type(cls) -> str:
        try:
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
            gpu_name = result.decode('utf-8').strip()

            if "A100" in gpu_name:
                return "a100"
            elif "H100" in gpu_name:
                return "h100"
            else:
                raise ValueError("GPU type not supported")
        except subprocess.CalledProcessError as e:
            raise ValueError("Error retrieving GPU name")

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
            args (Tuple): The arguments to pass to the function.
            kwargs (Dict[str, Any]): The keyword arguments to pass to the function.
            orig_not_implemented_exception (Exception): The original exception to raise if the fallback kernel
                is not implemented.

        Returns:
            Tuple[Any, float]: A tuple containing the result of the function and
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
                if cls.fake_mode.is_our_fake(e):
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
                    return cls.fake_mode.fake_tensor_converter.from_real_tensor(
                        cls.fake_mode, e
                    )
            else:
                return e

        return (pytree.tree_map(map_out, r), mean_op_time)

    @classmethod
    def _benchmark_estimate(cls, func, args, kwargs) -> Tuple[Any, float]:  # type: ignore[no-untyped-def]
        """
        Estimates the runtime of a function using benchmarking.

        Args:
            func: The function to estimate.
            args: The arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.
            res: The result of the function.

        Returns:
            Tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
        assert isinstance(
            cls.fake_mode, FakeTensorMode
        ), "Initialize/Assign FakeTensorMode before using this function"
        mean_op_time = 0.0
        if func._overloadpacket not in _IGNORE_OPS:
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

    # Adapted from: https://github.com/pytorch/pytorch/blob/9b902b3ee3bd608a19543362b66bf06c373dd374/torch/_inductor/scheduler.py#L589  # noqa: PGH004,B950
    @classmethod
    def _get_transfer_time(cls, flat_args_kwargs, flat_outs) -> float:  # type: ignore[no-untyped-def]
        """
        Estimates the runtime of a function using a roofline cost model.

        Args:
            func: The function to estimate.
            args: The arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.
            out: The output of the function.

        Returns:
            Tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
        assert (
            torch.cuda.is_available()
        ), "Roofline estimation needs to access CUDA capabilities to make estimations"

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
                math.ceil(num_bytes / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE
            )
            return mem_consumed

        gpu_memory_bandwidth = get_gpu_dram_gbps()
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
        return transfer_time

    @classmethod
    def _get_compute_time(cls, func_packet, args, kwargs, out, out_dtypes) -> float:  # type: ignore[no-untyped-def]
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
            # assert (
            #     len(out_dtypes) == 1
            # ), f"Only support single out dtype got {out_dtypes} for {func_packet}"
            # dtype = out_dtypes.pop()
            float_dtypes = out_dtypes & cls._float_types
            dtype = min(float_dtypes, key=lambda x: x.itemsize)
            if len(cls._peak_flops_reg) == 0:
                cls._peak_flops_reg = get_peak_flops_registry(cls.get_device_type().upper())
            
            peak_gpu_flops = cls._peak_flops_reg[dtype]
            peak_empirical_flops = cls._factors[dtype] * peak_gpu_flops
            flop_count_func = flop_registry[func_packet]
            # We divide by a factor of 2 to get the MACs (multiply and accumulate)
            flop_count = flop_count_func(*args, **kwargs, out_val=out) / 2
            # We multiply by 1e9 to get the time in nano seconds
            compute_time = (flop_count / peak_empirical_flops) * 1e9
            return compute_time
        return 0.0

    # Adapted from: https://github.com/pytorch/pytorch/blob/9b902b3ee3bd608a19543362b66bf06c373dd374/torch/_inductor/scheduler.py#L589  # noqa: PGH004,B950
    @classmethod
    def _roofline_estimate(cls, func, args, kwargs) -> Tuple[Any, float]:  # type: ignore[no-untyped-def]
        """
        Estimates the runtime of a function using a roofline cost model.

        Args:
            func: The function to estimate.
            args: The arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.
            out: The output of the function.

        Returns:
            Tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
        assert (
            torch.cuda.is_available()
        ), "Roofline estimation needs to access CUDA capabilities to make estimations"

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
        if func_packet not in _IGNORE_OPS:
            desugared_args = pytree.tree_map_only(torch.Tensor, get_flattened_tensor, args)
            desugared_kwargs = pytree.tree_map_only(torch.Tensor, get_flattened_tensor, kwargs)
            desugared_out = pytree.tree_map_only(torch.Tensor, get_flattened_tensor, out)

            flat_args_kwargs, _ = pytree.tree_flatten((desugared_args, desugared_kwargs))
            flat_outs, _ = pytree.tree_flatten(desugared_out)
            transfer_time = cls._get_transfer_time(flat_args_kwargs, flat_outs) / 2

            out_dtypes = {
                t.dtype
                for t in flat_outs
                if isinstance(t, torch.Tensor) and t.dtype in cls._float_types
            }

            compute_time = cls._get_compute_time(func_packet, desugared_args, desugared_kwargs, desugared_out, out_dtypes)
            # We get the estimated time as the max of the transfer time and
            # compute time. We divide by 1e6 to get the time in ms
            op_time = max(transfer_time, compute_time) / 1e6

        return (out, op_time)
        
    @classmethod
    def _learned_estimate_predictor(cls, func_packet, args, kwargs, out, out_dtypes) -> float:  # type: ignore[no-untyped-def]
        """
        TODO:
            1) the order of the features
            2) where the models are stored
        
        
        Estimates the compute time of an aten operator.

        Args:
            func_packet: The operator overload packet.
            args: The arguments to the operator.
            kwargs: The keyword arguments to the operator.
            out: The output of the operator.
            out_dtypes: The output data types.

        Returns:
            float: The estimated compute time in milliseconds.
            
        
        # TODO: comments.
        Note: for the prediction functions, we mimic the arguments for mm_flop.
        """
        op_time = 0.0
        if func_packet in _LEARNED_OPS:
            # assert (
            #     len(out_dtypes) == 1
            # ), f"Only support single out dtype got {out_dtypes} for {func_packet}"
            # dtype = out_dtypes.pop()
            float_dtypes = out_dtypes & cls._float_types
            dtype = min(float_dtypes, key=lambda x: x.itemsize)

            flop_count_func = flop_registry[func_packet]
            gflops = flop_count_func(*args, **kwargs, out_val=out) / 1e9
            predictor_func = _LEARNED_OPS[func_packet]
            gpu_id = torch.cuda.current_device()
            op_time = predictor_func(cls.gpu_types[gpu_id], dtype, gflops, *args, **kwargs, out_val=out)
            cls.count[func_packet] = cls.count.get(func_packet, 0) + 1
        return op_time
    
    @classmethod
    def _learned_estimate(cls, func, args, kwargs) -> Tuple[Any, float]:  # type: ignore[no-untyped-def]
        """
        Estimates the runtime of a function using a learned estimator.

        Args:
            func: The function to estimate.
            args: The arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.
            res: The result of the function.

        Returns:
            Tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
        assert (
            torch.cuda.is_available()
        ), "Learned estimator needs to access CUDA capabilities to make estimations"
        
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        op_time = 0.0
        func_packet = func._overloadpacket
        if func_packet not in _IGNORE_OPS:
            desugared_args = pytree.tree_map_only(torch.Tensor, get_flattened_tensor, args)
            desugared_kwargs = pytree.tree_map_only(torch.Tensor, get_flattened_tensor, kwargs)
            desugared_out = pytree.tree_map_only(torch.Tensor, get_flattened_tensor, out)

            flat_args_kwargs, _ = pytree.tree_flatten((desugared_args, desugared_kwargs))
            flat_outs, _ = pytree.tree_flatten(desugared_out)

            out_dtypes = {
                t.dtype
                for t in flat_outs
                if isinstance(t, torch.Tensor) and t.dtype in cls._float_types
            }

            if func_packet in _LEARNED_OPS:
                op_time = cls._learned_estimate_predictor(func_packet, desugared_args, desugared_kwargs, desugared_out, out_dtypes)
            else:
                # Roofline estimate.
                transfer_time = cls._get_transfer_time(flat_args_kwargs, flat_outs) / 2.75
                compute_time = cls._get_compute_time(func_packet, desugared_args, desugared_kwargs, desugared_out, out_dtypes)

                # We get the estimated time as the max of the transfer time and
                # compute time. We divide by 1e6 to get the time in ms
                op_time = max(transfer_time, compute_time) / 1e6

        return (out, op_time)

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
        for mod_fqn, runtimes in self.mod_runtimes.items():
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(
                f"{mod_fqn} fw: {runtimes.get('fw', 0.0):.3f}ms bw: {runtimes.get('bw', 0.0):.3f}ms"
            )

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):  # type: ignore[no-untyped-def]
        # TODO: @sanketpurandare: Flatten tensors by desugaring the tensor subclasses
        # TODO: @sanketpurandare: Add logic for incorporating communication time
        res, op_time = self._estimate(func, args, kwargs)
        for par in self._mod_tracker.parents:
            if self._mod_tracker.is_bw:
                self.mod_runtimes[par]["bw"] += op_time
            else:
                self.mod_runtimes[par]["fw"] += op_time
        self.total_compute_time += op_time
        return res

    def __call__(self, estimate_mode_type: str) -> Self:
        """
        Sets the estimate mode type.

        Currently supported modes:
            - "operator-level-benchmark": Estimates runtime using operator benchmarking.
            - "operator-level-cost-model": Estimates runtime using roofline cost model.
            - "operator-level-learned-model": Estimates runtime using learned cost model.

        Args:
            estimate_mode_type (str): The type of estimate mode to use.

        Returns:
            RuntimeEstimator: The runtime estimator instance.

        Raises:
            NotImplementedError: If the estimate mode type is not supported.
        """
        if estimate_mode_type == "operator-level-benchmark":
            self._estimate = RuntimeEstimator._benchmark_estimate
        elif estimate_mode_type == "operator-level-cost-model":
            self._estimate = RuntimeEstimator._roofline_estimate
        elif estimate_mode_type == "operator-level-learned-model":
            self._estimate = RuntimeEstimator._learned_estimate
        else:
            raise NotImplementedError(
                f"estimate_mode_type {estimate_mode_type} not supported"
            )
        self._estimate_mode_type = estimate_mode_type
        return self

    def __enter__(self) -> Self:
        fake_mode = active_fake_mode()
        assert isinstance(
            fake_mode, FakeTensorMode
        ), "No FakeTensorMode found, designed to used under FakeTensorMode"
        RuntimeEstimator.fake_mode = fake_mode
        self.total_compute_time = 0.0
        self.mod_runtimes = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.mod_fw_pre_order.clear()
        self.mod_bw_pre_order.clear()
        self.mod_fw_post_order.clear()
        self.mod_bw_post_order.clear()
        self._mod_tracker.register_user_hooks(
            pre_fw_hook=lambda mod, inp: self.mod_fw_pre_order.append(
                self._mod_tracker.get_known_fqn(mod)
            ),
            pre_bw_hook=lambda mod, g_out: self.mod_bw_pre_order.append(
                self._mod_tracker.get_known_fqn(mod)
            ),
            post_fw_hook=lambda mod, inp, out: self.mod_fw_post_order.append(
                self._mod_tracker.get_known_fqn(mod) if mod is not None else ""
            ),
            post_bw_hook=lambda mod, g_inp: self.mod_bw_post_order.append(
                self._mod_tracker.get_known_fqn(mod) if mod is not None else ""
            ),
        )
        self._mod_tracker.__enter__()
        super().__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        print(
            f"Estimated ({self._estimate_mode_type})"
            f" total_compute_time: {self.total_compute_time:.3f} ms"
        )
        if self._estimate_mode_type == 'operator-level-learned-model':
            print("count", self.count)
        if len(self._no_fallback_kernel) > 0:
            print("no_fallback_kernel: ", list(self._no_fallback_kernel))
        super().__exit__(*args)
        self._mod_tracker.clear_user_hooks()
        self._mod_tracker.__exit__()
