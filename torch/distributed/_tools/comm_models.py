import os
import pickle
from typing import Any, Callable

import numpy as np

import torch


aten = torch.ops.aten
c10d = torch.ops.c10d
_c10d_functional = torch.ops._c10d_functional
_c10d_functional_autograd = torch.ops._c10d_functional_autograd
_dtensor = torch.ops._dtensor


def listdir(p: str):
    return [os.path.join(p, f) for f in os.listdir(p)]


# substitute for your param path and num_gpus per node constants
# gpus_per_node = 4
# path = "/n/netscratch/idreos_lab/Everyone/FASRC_COLLECTIVE_MODELS"
gpus_per_node = 8
path = "/n/netscratch/idreos_lab/Everyone/MAST_COLLECTIVE_MODELS"

modeled_collectives = ["all_reduce", "all_gather", "reduce_scatter"]


# read saved parameters
def read_bw_params(path: str):
    interbw_params = np.load(os.path.join(path, "interbw_params.npy"))
    intrabw_params = np.load(os.path.join(path, "intrabw_params.npy"))
    return interbw_params, intrabw_params


def read_latency_params(path: str) -> tuple[float, float]:
    latency_params = np.load(os.path.join(path, "latency.npy"))
    return latency_params


def read_collective_params(path: str, collective: str, two_d: bool):
    with open(
        os.path.join(path, f"{collective}_{'2D' if two_d else '1D'}_params.pkl"), "rb"
    ) as f:
        return pickle.load(f)


# read constants and predict
interbw_params, intrabw_params = read_bw_params(path)
internode_latency, intranode_latency = read_latency_params(path)
saved_params: dict[tuple[str, bool], Any] = {
    ("all_reduce", False): read_collective_params(path, "all_reduce", False),
    ("all_reduce", True): read_collective_params(path, "all_reduce", True),
    ("all_gather", False): read_collective_params(path, "all_gather", False),
    ("all_gather", True): read_collective_params(path, "all_gather", True),
    ("reduce_scatter", False): read_collective_params(path, "reduce_scatter", False),
    ("reduce_scatter", True): read_collective_params(path, "reduce_scatter", True),
}


# bandwidth models
def sigmoid(x, L, x0, k) -> float:
    y = L / (1 + np.exp(-k * (x - x0)))
    return y


def log_sigmoid(x, L, x0, k) -> float:
    return sigmoid(np.log(x), L, x0, k)


def inter_bw(x) -> float:
    return log_sigmoid(x, *interbw_params)


def intra_bw(x) -> float:
    return log_sigmoid(x, *intrabw_params)


# collective models
def all_reduce_model(
    data_size: float, N: int, internode_only: bool = False
) -> tuple[float, float]:
    if internode_only:
        N //= gpus_per_node
        time_tree = (
            (2 * data_size) / (inter_bw(data_size + 1) / gpus_per_node) if N > 1 else 0
        )
        latency = (gpus_per_node - 1) * internode_latency
        return time_tree + latency, latency
    else:
        time_tree = (2 * data_size) / (inter_bw(data_size + 1)) if N > 1 else 0
        time_chain = (2 * data_size) / (intra_bw(data_size + 1))

        latency = (np.floor(np.log2(N)) + 1) * intranode_latency + (
            gpus_per_node - 1
        ) * internode_latency
        return time_tree + time_chain + latency, latency


def reduce_scatter_model(
    data_size: float, N: int, internode_only: bool = False
) -> tuple[float, float]:
    global internode_latency, intranode_latency
    n_nodes = N // gpus_per_node

    time_intra = ((N - n_nodes - 1) * data_size) / (
        (N - n_nodes) * intra_bw(data_size + 1)
    )
    time_inter = ((n_nodes - 1) * data_size) / ((n_nodes) * inter_bw(data_size + 1))

    latency = (
        n_nodes * internode_latency + (gpus_per_node - 1) * intranode_latency
        if not internode_only
        else n_nodes * internode_latency
    )

    if internode_only:
        return time_inter + latency, latency
    else:
        return time_intra + time_inter + latency, latency


def all_gather_model(
    data_size: float, N: int, internode_only: bool = False
) -> tuple[float, float]:
    global internode_latency, intranode_latency
    n_nodes = N // gpus_per_node

    time_intra = ((N - n_nodes - 1) * data_size) / (
        (N - n_nodes) * intra_bw(data_size + 1)
    )
    time_inter = ((n_nodes - 1) * data_size) / ((n_nodes) * inter_bw(data_size + 1))

    latency = (
        n_nodes * internode_latency + (gpus_per_node - 1) * intranode_latency
        if not internode_only
        else n_nodes * internode_latency
    )

    if internode_only:
        return time_inter + latency, latency
    else:
        return time_intra + time_inter + latency, latency


def broadcast_model(
    data_size: float, N: int, internode: bool = True
) -> tuple[float, float]:
    n_nodes = N // gpus_per_node

    if internode:
        return data_size / (inter_bw(data_size + 1) / gpus_per_node) + (
            n_nodes * internode_latency
        ), (n_nodes * internode_latency)
    else:
        return (
            data_size / intra_bw(data_size + 1)
            + ((gpus_per_node - 1) * intranode_latency)
        ), (gpus_per_node - 1) * intranode_latency


def scatter_model(
    data_size: float, N: int, internode: bool = True
) -> tuple[float, float]:
    n_nodes = N // gpus_per_node

    if internode:
        return data_size / (inter_bw(data_size + 1) / gpus_per_node) + (
            n_nodes * internode_latency
        ), (n_nodes * internode_latency)
    else:
        return (
            data_size / intra_bw(data_size + 1)
            + ((gpus_per_node - 1) * intranode_latency),
            (gpus_per_node - 1) * intranode_latency,
        )


def gather_model(
    data_size: float, N: int, internode: bool = True
) -> tuple[float, float]:
    n_nodes = N // gpus_per_node

    if internode:
        return data_size / (inter_bw(data_size + 1) / gpus_per_node) + (
            n_nodes * internode_latency
        ), (n_nodes * internode_latency)
    else:
        return (
            data_size / intra_bw(data_size + 1)
            + ((gpus_per_node - 1) * intranode_latency),
            (gpus_per_node - 1) * intranode_latency,
        )


def reduce_model(
    data_size: float, N: int, internode: bool = True
) -> tuple[float, float]:
    n_nodes = N // gpus_per_node

    if internode:
        return data_size / (inter_bw(data_size + 1) / gpus_per_node) + (
            n_nodes * internode_latency
        ), (n_nodes * internode_latency)
    else:
        return (
            data_size / intra_bw(data_size + 1)
            + ((gpus_per_node - 1) * intranode_latency),
            (gpus_per_node - 1) * intranode_latency,
        )


model_map: dict[str, Callable] = {
    "broadcast": broadcast_model,
    "all_reduce": all_reduce_model,
    "reduce": reduce_model,
    "all_gather": all_gather_model,
    "reduce_scatter": reduce_scatter_model,
    "gather": gather_model,
    "scatter": scatter_model,
    "send_recv": broadcast_model,  # TODO: Verify if this makes sense
}


def get_collective_model(func: torch._ops.OpOverload) -> Callable:
    broadcast_ops = {
        c10d.broadcast_.default,
        _c10d_functional.broadcast.default,
        _c10d_functional.broadcast_.default,
    }

    all_reduce_ops = {
        c10d.allreduce_.default,
        _c10d_functional.all_reduce.default,
        _c10d_functional.all_reduce_.default,
        _c10d_functional.all_reduce_coalesced.default,
        _c10d_functional.all_reduce_coalesced_.default,
    }

    reduce_ops = {c10d.reduce_.default}

    all_gather_ops = {
        c10d.allgather_.default,
        c10d._allgather_base_.default,
        c10d.alltoall_.default,
        c10d.alltoall_base_.default,
        _c10d_functional.all_to_all_single.default,
        _c10d_functional.all_gather_into_tensor.default,
        _c10d_functional_autograd.all_to_all_single.default,
        _dtensor.shard_dim_alltoall.default,
        _c10d_functional_autograd.all_gather_into_tensor.default,
        _c10d_functional.all_gather_into_tensor_coalesced.default,
        _c10d_functional.all_gather_into_tensor_out.default,
    }

    reduce_scatter_ops = {
        c10d.reduce_scatter_.default,
        c10d._reduce_scatter_base_.default,
        _c10d_functional.reduce_scatter_tensor.default,
        _c10d_functional.reduce_scatter_tensor_coalesced.default,
        _c10d_functional_autograd.reduce_scatter_tensor.default,
    }

    gather_ops = {c10d.gather_.default}
    scatter_ops = {c10d.scatter_.default}
    send_recv_ops = {
        c10d.send.default,
        c10d.recv_.default,
        c10d.recv_any_source_.default,
    }

    # Map the function to the appropriate model based on the defined sets
    if func in broadcast_ops:
        return model_map["broadcast"]
    elif func in all_reduce_ops:
        return model_map["all_reduce"]
    elif func in reduce_ops:
        return model_map["reduce"]
    elif func in all_gather_ops:
        return model_map["all_gather"]
    elif func in reduce_scatter_ops:
        return model_map["reduce_scatter"]
    elif func in gather_ops:
        return model_map["gather"]
    elif func in scatter_ops:
        return model_map["scatter"]
    elif func in send_recv_ops:
        return model_map["send_recv"]

    raise ValueError(f"Unknown collective operation: {func}")


def predict_communication(
    collective: Any,
    data_size: float,
    N: int,
    internode_only: bool = False,
    analytical_mode: bool = False,
) -> float:
    global saved_params
    if isinstance(collective, str):
        model_func = model_map[collective]
    else:
        model_func = get_collective_model(collective)

    if collective in modeled_collectives and not analytical_mode:
        analytical, latency = model_func(data_size, N, internode_only)
        min_params, straggle_params = saved_params[(collective, internode_only)]

        min_model = (
            min_params["Intercept"]
            + min_params["model"] * analytical
            + min_params["size"] * data_size
            + min_params["model:size"] * analytical * data_size
            + min_params["N"] * N
            + min_params["model:N"] * analytical * N
            + min_params["latency"] * latency
            + min_params["latency:size"] * data_size * latency
        )

        min_model = min_model if min_model > 0 else analytical

        straggle_model = (
            straggle_params["Intercept"]
            + straggle_params["np.log(size)"] * np.log(data_size)
            + straggle_params["N"] * N
        )

        model_pred = min_model * straggle_model
        return model_pred
    else:
        analytical, _ = model_func(data_size, N, internode_only)
        return analytical


if __name__ == "__main__":
    prediction = predict_communication("all_gather", 711308800 / 2**20, 128, False)
    print(f"Predicted all_gather time: {prediction}")

    prediction = predict_communication("all_reduce", 422617600 / 2**20, 64, False)
    print(f"Predicted all_reduce time: {prediction}")

    prediction = predict_communication("reduce_scatter", 511308800 / 2**20, 256, False)
    print(f"Predicted reduce_scatter time: {prediction}")
