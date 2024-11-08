import numpy as np
import os
import pickle
from typing import Any, Callable
import torch
aten = torch.ops.aten
c10d = torch.ops.c10d
_c10d_functional = torch.ops._c10d_functional

def listdir(p: str):
    return [os.path.join(p, f) for f in os.listdir(p)]

# constants
gpus_per_node = 4
# substitute for your param path
path = "/n/holyscratch01/idreos_lab/Users/spurandare/pytorch/torch/distributed/_tools/model_params"
nccl_path = "/n/holyscratch01/idreos_lab/Users/spurandare/pytorch/torch/distributed/_tools/nccl_debug"

modeled_collectives = ["all_reduce", "all_gather", "reduce_scatter"]
nccl_map = {
    "all_reduce": "AllReduce",
    "reduce_scatter": "ReduceScatter",
    "all_gather": "AllGather",
}


# read saved parameters
def read_bw_params(path: str):
    interbw_params = np.load(os.path.join(path, "interbw_params.npy"))
    intrabw_params = np.load(os.path.join(path, "intrabw_params.npy"))
    return interbw_params, intrabw_params


def read_latency_params(path: str):
    latency_params = np.load(os.path.join(path, "latency.npy"))
    return latency_params


def read_collective_params(path: str, collective: str, two_d: bool):
    with open(
        os.path.join(path, f'{collective}_{"2D" if two_d else "1D"}_params.pkl'), "rb"
    ) as f:
        return pickle.load(f)


# read constants and predict
interbw_params, intrabw_params = read_bw_params(path)
internode_latency, intranode_latency = read_latency_params(path)
saved_params = {
    ("all_reduce", False): read_collective_params(path, "all_reduce", False),
    ("all_reduce", True): read_collective_params(path, "all_reduce", True),
    ("all_gather", False): read_collective_params(path, "all_gather", False),
    ("all_gather", True): read_collective_params(path, "all_gather", True),
    ("reduce_scatter", False): read_collective_params(path, "reduce_scatter", False),
    ("reduce_scatter", True): read_collective_params(path, "reduce_scatter", True),
}


# bandwidth models
def sigmoid(x, L, x0, k):
    y = L / (1 + np.exp(-k * (x - x0)))
    return y


def log_sigmoid(x, L, x0, k):
    return sigmoid(np.log(x), L, x0, k)


def inter_bw(x):
    return log_sigmoid(x, *interbw_params)


def intra_bw(x):
    return log_sigmoid(x, *intrabw_params)


# collective models
def all_reduce_model(data_size: float, N: int, twod=False):
    if twod:
        N //= gpus_per_node
        time_tree = (
            (2 * data_size) / (inter_bw(data_size + 1) / gpus_per_node) if N > 1 else 0
        )
        latency = (gpus_per_node - 1) * internode_latency
        print("LATENCY:", latency)
        return time_tree + latency, latency
    else:
        time_tree = (2 * data_size) / (inter_bw(data_size + 1)) if N > 1 else 0
        time_chain = (2 * data_size) / (intra_bw(data_size + 1))

        latency = (np.floor(np.log2(N)) + 1) * intranode_latency + (
            gpus_per_node - 1
        ) * internode_latency
        return time_tree + time_chain + latency, latency


def reduce_scatter_model(data_size: float, N: int, twod=False):
    global internode_latency, intranode_latency
    n_nodes = N // gpus_per_node

    time_intra = ((N - n_nodes - 1) * data_size) / (
        (N - n_nodes) * intra_bw(data_size + 1)
    )
    time_inter = ((n_nodes - 1) * data_size) / ((n_nodes) * inter_bw(data_size + 1))

    latency = (
        n_nodes * internode_latency + (gpus_per_node - 1) * intranode_latency
        if not twod
        else n_nodes * internode_latency
    )

    if twod:
        return time_inter + latency, latency
    else:
        return time_intra + time_inter + latency, latency


def all_gather_model(data_size: float, N: int, twod=False):
    global internode_latency, intranode_latency
    n_nodes = N // gpus_per_node

    time_intra = ((N - n_nodes - 1) * data_size) / (
        (N - n_nodes) * intra_bw(data_size + 1)
    )
    time_inter = ((n_nodes - 1) * data_size) / ((n_nodes) * inter_bw(data_size + 1))

    latency = (
        n_nodes * internode_latency + (gpus_per_node - 1) * intranode_latency
        if not twod
        else n_nodes * internode_latency
    )

    if twod:
        return time_inter + latency, latency
    else:
        return time_intra + time_inter + latency, latency


def broadcast_model(data_size: float, N: int, inter=True):
    n_nodes = N // gpus_per_node

    if inter:
        return data_size / (inter_bw(data_size + 1) / gpus_per_node) + (
            n_nodes * internode_latency
        ), (n_nodes * internode_latency)
    else:
        return (
            data_size / intra_bw(data_size + 1)
            + ((gpus_per_node - 1) * intranode_latency)
        ), (gpus_per_node - 1) * intranode_latency


def scatter_model(data_size: float, N: int, inter=True):
    n_nodes = N // gpus_per_node

    if inter:
        return data_size / (inter_bw(data_size + 1) / gpus_per_node) + (
            n_nodes * internode_latency
        ), (n_nodes * internode_latency)
    else:
        return (
            data_size / intra_bw(data_size + 1)
            + ((gpus_per_node - 1) * intranode_latency),
            (gpus_per_node - 1) * intranode_latency,
        )


def gather_model(data_size: float, N: int, inter=True):
    n_nodes = N // gpus_per_node

    if inter:
        return data_size / (inter_bw(data_size + 1) / gpus_per_node) + (
            n_nodes * internode_latency
        ), (n_nodes * internode_latency)
    else:
        return (
            data_size / intra_bw(data_size + 1)
            + ((gpus_per_node - 1) * intranode_latency),
            (gpus_per_node - 1) * intranode_latency,
        )


def reduce_model(data_size: float, N: int, inter=True):
    n_nodes = N // gpus_per_node

    if inter:
        return data_size / (inter_bw(data_size + 1) / gpus_per_node) + (
            n_nodes * internode_latency
        ), (n_nodes * internode_latency)
    else:
        return (
            data_size / intra_bw(data_size + 1)
            + ((gpus_per_node - 1) * intranode_latency),
            (gpus_per_node - 1) * intranode_latency,
        )


def get_collective_model(func: Any) -> Callable:
    match func:
        case c10d.broadcast_.default | _c10d_functional.broadcast.default:
            return broadcast_model
        case c10d.allreduce_.default | _c10d_functional.all_reduce.default:
            return all_reduce_model
        case c10d.reduce_.default:
            return reduce_model
        case c10d.allgather_.default | c10d._allgather_base_.default | c10d.alltoall_.default | c10d.alltoall_base_.default | _c10d_functional.all_to_all_single.default | _c10d_functional.all_gather_into_tensor.default:
            return all_gather_model
        case c10d.reduce_scatter_.default | c10d._reduce_scatter_base_.default | _c10d_functional.reduce_scatter_tensor.default:
            return reduce_scatter_model
        case c10d.gather_.default:
            return gather_model
        case c10d.scatter_.default:
            return scatter_model
        case c10d.send.default | c10d.recv_.default:
            # TODO: Check if this makes sense
            return broadcast_model

def get_model(collective: str):
    if collective == "all_reduce":
        return all_reduce_model
    elif collective == "all_gather":
        return all_gather_model
    elif collective == "reduce_scatter":
        return reduce_scatter_model
    elif collective == "broadcast":
        return broadcast_model
    elif collective == "scatter":
        return scatter_model
    elif collective == "gather":
        return gather_model
    elif collective == "reduce":
        return reduce_model


def predict_communication(
    collective: Any, data_size: float, N: int, twod: bool = False
):
    global saved_params
    if isinstance(collective, str):
        model_func = get_model(collective)
    else:
        model_func = get_collective_model(collective)

    if collective in modeled_collectives:
        analytical, latency = model_func(data_size, N, twod)
        min_params, straggle_params = saved_params[(collective, twod)]

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
        analytical, _ = model_func(data_size, N, twod)
        return analytical


# +++++++++

# nccl models
def find_table_start(text):
    for line_num, line in enumerate(text):
        if "Algorithm" in line:
            return line_num
    return None  # Returns None if "Algorithm" is not found

def extract_algorithm_table(log_text):
    # Use regular expression to find the table starting at "Algorithm"
    table_start = find_table_start(log_text)
    log_text = log_text[table_start:table_start + 16]

    start_index = log_text[0].find("INFO")
    log_text = [line[start_index + 5:] for line in log_text]
    return log_text

# NCCL strings for functions, algorithms, and protocols
FUNC = ["Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce"]
ALGO = ["Tree", "Ring", "CollNetDirect", "CollNetChain", "NVLS", "NVLSTree"]
PROTO = ["LL", "LL128", "Simple"]

def read_tuning_table(tuning_table: list[str]):
    # String Lengths
    FUNC_NAME_LEN = 15
    LATENCY_LEN = 8
    BANDWIDTH_LEN = 7
    ENTRY_LEN = LATENCY_LEN + BANDWIDTH_LEN + 2
    ALGO_ENTRY_LEN = ENTRY_LEN * 3

    del tuning_table[0:3]
    del tuning_table[len(FUNC) : len(FUNC) + 3]
    
    # Parse tuning table
    latencies = {}
    bandwidths = {}

    for block in range(2):
        for ba in range(len(ALGO) // 2):
            a = ba + block * 3
            for p in range(len(PROTO)):
                for f in range(len(FUNC)):
                    algo = ALGO[a]
                    proto = PROTO[p]
                    func = FUNC[f]

                    string_pos = FUNC_NAME_LEN + ALGO_ENTRY_LEN * ba + ENTRY_LEN * p
                    latency = tuning_table[f + block * len(FUNC)][
                        string_pos : string_pos + LATENCY_LEN
                    ]
                    bandwidth = tuning_table[f + block * len(FUNC)][
                        string_pos
                        + LATENCY_LEN
                        + 1 : string_pos
                        + LATENCY_LEN
                        + BANDWIDTH_LEN
                        + 1
                    ]

                    latencies[tuple([algo, proto, func])] = (
                        float(latency) / 1e6
                    )  # convert us to s
                    bandwidths[tuple([algo, proto, func])] = float(bandwidth) * 8

    return (latencies, bandwidths)

def read_nccl(path: str):
    # Read log file content as a string
    with open(path, 'r') as file:
        log_content = file.readlines()

    # Extract the table
    algorithm_table = extract_algorithm_table(log_content)
    return read_tuning_table(algorithm_table)

nccl_params = {
    int(os.path.basename(p).split("_")[-1]): read_nccl(p) for p in listdir(nccl_path)
}

def nccl_model(data_size: float, num_processors: int, collective: str):
    global nccl_params
    latencies, bandwidths = nccl_params[num_processors // gpus_per_node]

    data_size /= 1000
    algo_str = "Tree" if collective == "AllReduce" else "Ring"
    nccl_lat_bw_cand = [
        (
            latencies[tuple([algo_str, proto, collective])],
            bandwidths[tuple([algo_str, proto, collective])],
        )
        for proto in PROTO
        if bandwidths[tuple([algo_str, proto, collective])] != 0 and latencies[tuple([algo_str, proto, collective])] != 0
    ]
    nccl_latency, nccl_bw = min(nccl_lat_bw_cand, key=lambda x: x[0] + data_size/x[1])
    return (nccl_latency + (data_size * num_processors)/nccl_bw) * 1e3


if __name__ == "__main__":
    prediction = predict_communication("all_gather", 1711308800, 128, False)
    print(f"Predicted communication time: {prediction}")

    prediction = predict_communication("all_reduce", 3422617600, 128, False)
    print(f"Predicted communication time: {prediction}")
