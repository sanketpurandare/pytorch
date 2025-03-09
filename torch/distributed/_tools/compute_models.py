# Owner(s): ["module: unknown"]
from functools import wraps
import os
import joblib
import pandas as pd
from typing import Any
import torch
from torch.utils._pytree import tree_map_, tree_map
from torch.utils.flop_counter import flop_registry


aten = torch.ops.aten

# Similar to `flop_registry`, stores the operators that have learned predictors
LEARNED_OPS: dict[torch._ops.OpOverloadPacket, Any] = {}

# Caches the learned models that predict ops' runtimes.
_LEARNED_OPS_PREDICTORS: dict[str, Any] = {}

_MODEL_BASE_DIR = "/n/netscratch/idreos_lab/Everyone"


def get_learned_model(op: str, gpu_type: str) -> Any:
    if op not in _LEARNED_OPS_PREDICTORS:
        path = os.path.join(_MODEL_BASE_DIR, f"{gpu_type}_models", f"{op}.joblib")

        _LEARNED_OPS_PREDICTORS[op] = joblib.load(path)
    return _LEARNED_OPS_PREDICTORS[op]

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
            if target in LEARNED_OPS:
                raise RuntimeError(f"duplicate registrations for {target}")
            LEARNED_OPS[target] = time_formula

        # To handle allowing multiple aten_ops at once
        tree_map_(register, targets)

        return time_formula

    return register_fun

def convert_dtype(dtype) -> dict[str, int]:
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

# @register_timing_formula([aten.convolution, aten._convolution])
# def conv_time(gpu_type, dtype, gflops, x_shape, w_shape, _bias, _stride, _padding, _dilation, transposed, *args, out_shape=None, **kwargs) -> float:
#     """
#     TODO: need to add support for higher dims.
#     """
#     model = get_learned_model("conv", gpu_type)

#     # batch_size = x_shape[0]
#     # conv_shape = (x_shape if transposed else out_shape)[2:]
#     # c_out, c_in, *filter_size = w_shape

#     # features = np.array([[b, m, k, n, gflops]])
#     return model.predict(features)

# @register_timing_formula(aten.convolution_backward)
# def conv_backward_time(
#     gpu_type,
#     dtype,
#     gflops,
#     grad_out_shape,
#     x_shape,
#     w_shape,
#     _bias,
#     _stride,
#     _padding,
#     _dilation,
#     transposed,
#     _output_padding,
#     _groups,
#     output_mask,
#     out_shape) -> float:
#     """
#     TODO: need to add support for higher dims.
#     """
#     model = get_learned_model("conv_backward", gpu_type)

#     # features = np.array([[b, m, k, n, gflops]])
#     return model.predict(features)

def learned_estimate_predictor(func_packet, args, kwargs, out, out_dtypes, gpu_type) -> float:  # type: ignore[no-untyped-def]
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
    # assert (
    #     len(out_dtypes) == 1
    # ), f"Only support single out dtype got {out_dtypes} for {func_packet}"
    # dtype = out_dtypes.pop()
    dtype = min(out_dtypes, key=lambda x: x.itemsize)
    flop_count_func = flop_registry[func_packet]
    gflops = flop_count_func(*args, **kwargs, out_val=out) / 1e9
    predictor_func = LEARNED_OPS[func_packet]
    op_time = predictor_func(gpu_type, dtype, gflops, *args, **kwargs, out_val=out)
    return op_time
    


