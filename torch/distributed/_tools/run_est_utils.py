from typing import Dict
import subprocess
import warnings
import torch 
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

peak_factors: Dict[str, Dict[torch.dtype, float]] = {
    "h100": {
        torch.float16: 0.6,
        torch.bfloat16: 0.6,
        torch.float32: 0.5,
        torch.float64: 0.5
    },
    "a100": {
        torch.float16: 0.75,
        torch.bfloat16: 0.75,
        torch.float32: 0.65,
        torch.float64: 0.65
    }
}

def get_peak_flops_registry(device_name: str) -> Dict[torch.dtype, int]:
    """
    Returns peak FLOPS for given device and data type.

    Args:
        device_name (str): Device name (e.g., "H100", "A100").

    Returns:
        Dict[torch.dtype, int]: Peak FLOPS reistry for the device.

    Raises:
        ValueError: If device is not supported.
    """
    try:
        # Run lspci command and capture output
        result = subprocess.run(["lspci"], stdout=subprocess.PIPE, text=True)
        
        # Filter output for lines containing device name
        device_lines = [
            line
            for line in result.stdout.splitlines()
            if device_name in line
        ]
        
        if not device_lines:
            raise ValueError(f"Device {device_name} not found")
        
        # Determine model type (NVL or SXM) for H100
        model_type = None
        if device_name == "H100":
            for line in device_lines:
                if "NVL" in line:
                    model_type = "NVL"
                    break
                elif "SXM" in line:
                    model_type = "SXM"
                    break
            if model_type is None:
                raise ValueError(f"Unable to determine model type for device {device_name}")
        
        # Define peak FLOPS registry
        peak_flops_registry = {
            "A100": {
                torch.float64: 9.7e12,
                torch.float32: 19.5e12,
                torch.bfloat16: 312e12,
                torch.float16: 312e12,
                torch.int8: 624e12,
            },
            "H100 SXM": {
                torch.float64: 34e12,
                torch.float32: 67e12,
                torch.bfloat16: 1979e12,
                torch.float16: 1979e12,
                torch.int8: 3958e12,
            },
            "H100 NVL": {
                torch.float64: 30e12,
                torch.float32: 60e12,
                torch.bfloat16: 1671e12,
                torch.float16: 1671e12,
                torch.int8: 3341e12,
            },
        }
        
        # Get peak FLOPS for device and data type
        device_key = device_name if device_name == "A100" else f"{device_name} {model_type}"
        peak_flops_reg = peak_flops_registry.get(device_key, {})
        
        if len(peak_flops_reg) == 0:
            raise ValueError(f"Unsupported device {device_name}")
        
        return peak_flops_reg
    
    except subprocess.CalledProcessError as e:
        print(f"Error running lspci: {e}")
        raise
    except Exception as e:
        print(e)
        raise

  
def get_flattened_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Recursively extracts flattened tensor from a traceable wrapper-subclass of tensor.

    Args:
        t (torch.Tensor): The tensor to extract from.

    Returns:
        torch.Tensor: A flattened tensor.
    """
    unflattened_tensors = [t]
    flattened_tensor = None
    while len(unflattened_tensors) > 0:
        obj = unflattened_tensors.pop()
        if is_traceable_wrapper_subclass(obj):
            attrs, _ = obj.__tensor_flatten__()  # type: ignore[attr-defined]
            unflattened_tensors.extend([getattr(obj, attr) for attr in attrs])
        else:
            if not hasattr(obj, "untyped_storage"):
                warnings.warn(
                    f"Expected a tensor or a traceable wrapper-subclass of tensor, but got {type(obj)}",
                    category=UserWarning,
                    stacklevel=2,
                )
            else:
                flattened_tensor = obj
                assert len(unflattened_tensors) == 0, "More than one flattened tensors"
                break
    return flattened_tensor