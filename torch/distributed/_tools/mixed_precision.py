from functools import partial
from typing import Any

import torch
from torch import nn
from torch.distributed._composable import contract
from torch.distributed._composable_state import (
    _get_module_state,
    _insert_module_state,
    _module_state_mapping,
    _State,
)
from torch.distributed.fsdp import MixedPrecisionPolicy

from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.utils import _to_kwargs
from torch.utils._pytree import tree_flatten, tree_map
from torch.distributed.fsdp._fully_shard._fsdp_common import _cast_fp_tensor

class MPState(_State):
    def __init__(self):
        super().__init__()

    def init(
        self,
        modules: tuple[nn.Module, ...],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
    ) -> None:
        for module in modules:
            _insert_module_state(module, self)
        self._modules = modules
        self._device = device
        self._device_handle = _get_device_handle(device.type)
        self._mp_policy = mp_policy

    def _root_pre_forward(
            self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        pass

    def _pre_forward(
            self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        args, kwargs = self._root_pre_forward(module, args, kwargs)
        if self._mp_policy.cast_forward_inputs and self._mp_policy.param_dtype:
            cast_fn = partial(
                _cast_fp_tensor, self._mp_policy.param_dtype
            )
            args, kwargs = tree_map(cast_fn, args), tree_map(cast_fn, kwargs)