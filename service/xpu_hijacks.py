## code credit goes to https://github.com/vladmandic/automatic/blob/master/modules/intel/ipex/hijacks.py
##

from pathlib import Path
import logging
from functools import wraps
from contextlib import nullcontext
import torch
import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import
import numpy as np
from typing import Union, IO, Callable, Optional, Tuple, Any
from pydantic import BaseSettings, Field

"""
PyTorch Compatibility Layer for Intel XPUs

This module provides a compatibility layer to ensure that PyTorch code designed for CUDA devices
can run smoothly on Intel XPU devices. It accomplishes this by hijacking and overriding various
PyTorch functions, properties, and attributes, adapting them to work with XPU-specific features
and limitations.

Key features of this module include:

- Redirecting CUDA Operations to XPU: It globally redirects PyTorch's CUDA functions and
  attributes to use Intel XPUs. This enables compatibility with existing CUDA-based PyTorch code.
- Data Type Handling: It handles data type conversions and ensures consistency in data types across
  different operations, particularly when working with XPUs that might have limitations on supported data types.
- Workarounds for XPU Limitations: It implements specific workarounds for known issues with certain
  XPU architectures (e.g., Alchemist GPUs), such as lack of FP64 support or limitations in specific
  operations.
- Performance Optimization: It includes optimizations for specific operations (e.g., attention mechanisms)
  to improve performance on XPUs, potentially using techniques like slicing or CPU offloading.
- Preservation of Function Metadata: It uses the `@wraps` decorator to preserve the documentation
  and metadata of the original PyTorch functions being overridden, ensuring clarity and consistency.

Usage:
    To enable these compatibility patches, simply import this module and call the `ipex_hijacks()` function.
"""

# Define a Pydantic model for settings
class XPUConfig(BaseSettings):
    """Configuration settings for XPU compatibility."""
    force_attention_slice: Optional[bool] = Field(
        default=None, env="IPEX_FORCE_ATTENTION_SLICE"
    )

# Instantiate the settings object. This will load the environment variables and apply any
# validation rules defined in the XPUConfig class.
xpu_config = XPUConfig()

# Check if the current XPU device supports 64-bit floating-point precision (FP64)
device_supports_fp64 = torch.xpu.has_fp64_dtype()

# Globally redirect PyTorch CUDA operations to use XPU operations.
# This makes any code using `torch.cuda` run on XPU instead.
torch.cuda = torch.xpu

def return_null_context(*args: Optional[tuple], **kwargs: Optional[dict]) -> nullcontext:  # pylint: disable=unused-argument
    """
    Returns a null context manager.

    This is a utility function to disable certain context-based operations, such as CUDA's SDP kernel,
    when they are not needed or supported on XPUs.

    Args:
        *args (Optional[tuple]): Variable length positional arguments (ignored).
        **kwargs (Optional[dict]): Variable length keyword arguments (ignored).

    Returns:
        nullcontext: A context manager that does nothing.
    """
    return nullcontext()

@wraps(torch.cuda.is_available)
def is_available() -> bool:
    """
    Explicitly checks if an XPU device is available using IPEX.

    This function overrides `torch.cuda.is_available` to check for XPU availability. 
    It provides an explicit check, even though a check is already performed implicitly 
    during the global redirection of `torch.cuda` to `torch.xpu`.

    Returns:
        bool: True if an XPU device is available, False otherwise.
    """
    return ipex.has_xpu()

@property
def is_cuda(self) -> bool:
    """
    Checks if a tensor's device is either an XPU or a CUDA device.

    This property is added to PyTorch's `Tensor` object to provide a unified way
    to check if a tensor is on a device suitable for acceleration (either CUDA or XPU).

    Returns:
        bool: True if the tensor's device is 'xpu' or 'cuda', False otherwise.
    """
    return self.device.type in {'xpu', 'cuda'}

def check_device(device: Union[torch.device, str, int]) -> bool:
    """
    Checks if the provided device specification is for a CUDA or XPU device.

    Args:
        device (torch.device or str or int): The device specification to check. Can be a `torch.device` object,
                                             a string like 'cuda:0', 'xpu:0', or an integer representing a device index.

    Returns:
        bool: True if the device specification is for 'cuda' or 'xpu', False otherwise.
    """
    if isinstance(device, torch.device):
        return device.type in {"cuda", "xpu"}
    if isinstance(device, str):
        return device.startswith("cuda") or device.startswith("xpu")  # Ensure "cuda" or "xpu" is at the start
    if isinstance(device, int):
        return True  # Assuming int is used for device index and valid for CUDA/XPU
    return False

def return_xpu(device: Union[str, int, torch.device]) -> Union[str, torch.device]:
    """
    Converts a given device specification to an XPU device specification.

    Args:
        device (Union[str, int, torch.device]): The device to convert.

    Returns:
        Union[str, torch.device]: The XPU device specification.

    Raises:
        ValueError: If the provided device is not a string, integer, or torch.device
    """
    if isinstance(device, str) and ":" in device:
        return f"xpu:{device.split(':')[-1]}"
    if isinstance(device, int):
        return f"xpu:{device}"
    if isinstance(device, torch.device):
        return torch.device("xpu")
    raise ValueError("Invalid device specification. Expected torch.device, str, or int.")


# --- Autocast Modifications ---

# Store a reference to the original `torch.amp.autocast_mode.autocast.__init__`
original_autocast_init = torch.amp.autocast_mode.autocast.__init__

# Override autocast initialization to use bfloat16 for XPU devices if no dtype is specified
@wraps(torch.amp.autocast_mode.autocast.__init__)
def autocast_init(self, device_type, dtype=None, enabled=True, cache_enabled=None):
    """
    Initializes the autocast context manager.

    If the device type is 'cuda' or 'xpu' and no dtype is specified, the default dtype
    is set to `torch.bfloat16` (bfloat16) for XPU devices.

    This override ensures that autocasting on XPUs uses bfloat16, which is often
    preferred for its memory efficiency on these devices.
    """
    if device_type in ("cuda", "xpu"):  # Ensure any request for "cuda" or "xpu" is treated as "xpu"
        if dtype is None:
            dtype = torch.bfloat16
        return original_autocast_init(self, device_type="xpu", dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
    else:
        return original_autocast_init(self, device_type=device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)


# --- Latent Antialias CPU Offload ---

# Store a reference to the original `torch.nn.functional.interpolate`
original_interpolate = torch.nn.functional.interpolate

# Override `torch.nn.functional.interpolate` to offload to CPU for certain cases
@wraps(torch.nn.functional.interpolate)
def interpolate(tensor, size=None, scale_factor=None, mode='nearest', align_corners=None,
                recompute_scale_factor=None, antialias=False): # pylint: disable=too-many-arguments
    """
    Performs interpolation on the input tensor.

    This function overrides the standard PyTorch interpolation to offload the operation
    to the CPU with `torch.float32` precision for specific conditions:

    - If `antialias` is True.
    - If `align_corners` is not None.
    - If `mode` is 'bicubic'.

    This offloading is likely done due to limited XPU support or potential performance
    issues for these specific configurations.

    Args:
        tensor (torch.Tensor): The input tensor to be interpolated.
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional): Output spatial size.
        scale_factor (float or Tuple[float], optional): Multiplier for spatial size. 
        mode (str, optional): Algorithm used for interpolation. Defaults to 'nearest'.
        align_corners (bool, optional): Geometrically, we consider the points of the input and output tensors 
            as being centers of their respective grid cells. Defaults to None.
        recompute_scale_factor (bool, optional): Recompute the scale_factor for use in the interpolation 
            calculation. Defaults to None.
        antialias (bool, optional):  flag to apply antialiasing. Defaults to False.
    Returns:
        torch.Tensor: The interpolated tensor.
    """
    if antialias or align_corners is not None or mode == 'bicubic':
        return_device = tensor.device
        return_dtype = tensor.dtype
        # Offload to CPU for better compatibility
        return original_interpolate(tensor.to("cpu", dtype=torch.float32), size=size, scale_factor=scale_factor, mode=mode,
                                   align_corners=align_corners, recompute_scale_factor=recompute_scale_factor,
                                   antialias=antialias).to(return_device, dtype=return_dtype)
    return original_interpolate(tensor, size=size, scale_factor=scale_factor, mode=mode,
                                   align_corners=align_corners, recompute_scale_factor=recompute_scale_factor,
                                   antialias=antialias)


# --- Diffusers Float64 Workaround ---

# Preserve the original `torch.from_numpy` function to maintain original functionality
original_from_numpy = torch.from_numpy

@wraps(original_from_numpy)
def from_numpy(ndarray: np.ndarray) -> torch.Tensor:
    """
    Converts a NumPy ndarray to a PyTorch tensor, ensuring compatibility with XPU devices.

    This override handles potential data type incompatibilities with certain XPU architectures.
    If the NumPy array's data type is `float`, it is converted to `float32` before creating
    the PyTorch tensor, to ensure compatibility.

    Args:
        ndarray (np.ndarray): The NumPy array to convert.

    Returns:
        torch.Tensor: The converted tensor.
    """
    if ndarray.dtype == float:
        ndarray = ndarray.astype(np.float32)  # Convert to float32 for compatibility
    return original_from_numpy(ndarray)

# Preserve the original `torch.as_tensor` function
original_as_tensor = torch.as_tensor

@wraps(original_as_tensor)
def as_tensor(data: Union[np.ndarray, list, tuple, float, int], 
              dtype: Optional[torch.dtype] = None, 
              device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """
    Converts various data formats to a PyTorch tensor, ensuring compatibility with XPU devices.

    This override handles device and data type conversions, particularly converting
    NumPy arrays with a `float` data type to `float32` if the target device is not "cpu".

    Args:
        data (array_like): The data to convert to a tensor (e.g., list, tuple, NumPy ndarray, scalar).
        dtype (torch.dtype, optional): The desired data type of the tensor. Inferred if None.
        device (torch.device or str, optional): The device for the tensor. Defaults to the current device.

    Returns:
        torch.Tensor: The resulting tensor.
    """
    if check_device(device):
        device = return_xpu(device)
    if isinstance(data, np.ndarray) and data.dtype == float and not (
        (isinstance(device, torch.device) and device.type == "cpu") or 
        (isinstance(device, str) and "cpu" in device)):
        data = data.astype(np.float32)  # Convert to float32 for compatibility
    return original_as_tensor(data, dtype=dtype, device=device)

# --- 32-Bit Attention Workarounds ---

# Define a function to conditionally select the appropriate attention functions
def _select_attention_functions():
    """
    Selects the appropriate functions for `torch.bmm` and `scaled_dot_product_attention`.

    This function determines whether to use the original PyTorch functions or optimized 32-bit
    versions based on device capabilities and configuration. If the device supports FP64 and 
    slicing is not enforced (via the `xpu_config.force_attention_slice` setting), it uses the 
    original functions. Otherwise, it attempts to load the 32-bit optimized versions from the
    `attention` module. If the import fails, it falls back to the original functions and logs a warning.

    Returns:
        Tuple[Callable, Callable]: A tuple containing the selected `torch.bmm` and 
                                   `scaled_dot_product_attention` functions. 
    """
    if device_supports_fp64 and xpu_config.force_attention_slice is None:
        return torch.bmm, torch.nn.functional.scaled_dot_product_attention

    try:
        from attention import torch_bmm_32_bit, scaled_dot_product_attention_32_bit
        return torch_bmm_32_bit, scaled_dot_product_attention_32_bit
    except ImportError as e:
        logger.warning(f"32-bit attention functions not found: {e}. Falling back to original functions.")
        return torch.bmm, torch.nn.functional.scaled_dot_product_attention

# Apply the selected attention functions based on device and configuration.
original_torch_bmm, original_scaled_dot_product_attention = _select_attention_functions() 

# --- Data Type Errors: ---

def _ensure_matching_dtype(*tensors: torch.Tensor):
    """Ensures that all input tensors have the same data type as the first tensor.

    This helper function checks the data type of the first tensor in the list
    and converts all subsequent tensors to that type if they don't already match.

    Args:
        *tensors: A variable number of PyTorch tensors.
    """
    base_dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        if tensor is not None and tensor.dtype != base_dtype:
            tensor.to(base_dtype) 

# Override `torch.bmm` to ensure matching data types for inputs
@wraps(torch.bmm)
def torch_bmm(input: torch.Tensor, mat2: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Performs a batch matrix-matrix product of matrices stored in input and mat2.

    This function overrides the standard `torch.bmm` to ensure that both input tensors
    have the same data type before performing the operation. If the data types of `input`
    and `mat2` do not match, `mat2` is converted to the data type of `input`.

    Args:
        input (torch.Tensor): The first batch of matrices to be multiplied.
        mat2 (torch.Tensor): The second batch of matrices to be multiplied.
        out (torch.Tensor, optional): The output tensor.

    Returns:
        torch.Tensor: The result of the batch matrix multiplication.
    """
    _ensure_matching_dtype(input, mat2)
    return original_torch_bmm(input, mat2, out=out)  

# Override `torch.nn.functional.scaled_dot_product_attention` to ensure consistent data types across all inputs
@wraps(torch.nn.functional.scaled_dot_product_attention)
def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                                 attn_mask: Optional[torch.Tensor] = None, dropout_p: float = 0.0, 
                                 is_causal: bool = False, **kwargs) -> torch.Tensor:
    """
    Computes scaled dot product attention on input tensors, ensuring data type consistency.

    This function ensures that the `query`, `key`, `value`, and `attn_mask` (if provided) 
    tensors all have the same data type. If the data types do not match, the `key`, 
    `value`, and `attn_mask` tensors are converted to the data type of the `query` tensor.

    Args:
        query (torch.Tensor): Query tensor.
        key (torch.Tensor): Key tensor.
        value (torch.Tensor): Value tensor.
        attn_mask (torch.Tensor, optional): Optional attention mask tensor. Default: None.
        dropout_p (float, optional): Dropout probability. Default: 0.0
        is_causal (bool, optional): If True, applies a causal mask to the attention scores. Default: False
        **kwargs:  Additional keyword arguments to be passed to the original scaled dot product attention function.

    Returns:
        torch.Tensor: The output tensor resulting from the scaled dot product attention operation.
    """
    _ensure_matching_dtype(query, key, value, attn_mask)
    return original_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)

def _ensure_tensor_dtype(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Converts the tensor to the specified target data type if necessary.

    Args:
        tensor: The tensor to potentially convert.
        target_dtype: The desired data type.

    Returns:
        torch.Tensor: The input tensor, potentially converted to the target dtype.
    """
    if tensor.dtype != target_dtype:
        return tensor.to(dtype=target_dtype)
    return tensor

# --- A1111 FP16 Workaround ---
original_functional_group_norm = torch.nn.functional.group_norm

@wraps(torch.nn.functional.group_norm)
def functional_group_norm(input: torch.Tensor, num_groups: int, weight: Optional[torch.Tensor] = None, 
                          bias: Optional[torch.Tensor] = None, eps: float = 1e-05) -> torch.Tensor:
    """
    Applies Group Normalization over a mini-batch of inputs, ensuring data type consistency.

    This override ensures data type compatibility between input, weight, and bias, potentially converting
    the input tensor to the data type of the weight tensor. This is especially important when working with 
    FP16 precision, where data type mismatches can cause issues.

    Args:
        input (torch.Tensor): Input tensor of shape (N, C, *).
        num_groups (int): Number of groups to separate the channels into.
        weight (torch.Tensor, optional): Scale factor. Default: None.
        bias (torch.Tensor, optional): Additive bias. Default: None.
        eps (float, optional): Value added to the denominator for numerical stability. Default: 1e-5.

    Returns:
        torch.Tensor: Output tensor of the same shape as input.
    """
    input = _ensure_tensor_dtype(input, weight.data.dtype if weight is not None else None)
    if bias is not None:
        bias.data = _ensure_tensor_dtype(bias, weight.data.dtype if weight is not None else None)
    return original_functional_group_norm(input, num_groups, weight=weight, bias=bias, eps=eps)

# --- A1111 BF16 Workaround ---
original_functional_layer_norm = torch.nn.functional.layer_norm

@wraps(torch.nn.functional.layer_norm)
def functional_layer_norm(input: torch.Tensor, normalized_shape: Union[int, list, torch.Size], weight: Optional[torch.Tensor] = None,
                          bias: Optional[torch.Tensor] = None, eps: float = 1e-05) -> torch.Tensor:
    """
    Applies Layer Normalization over a mini-batch of inputs, ensuring data type consistency.

    This override ensures data type consistency between input, weight, and bias, converting the input
    to the data type of the weight if necessary. This is particularly relevant when working with bfloat16 (BF16)
    precision.

    Args:
        input (torch.Tensor): Input tensor of shape (N, *).
        normalized_shape (int or list or torch.Size): Input shape from an expected input of size.
        weight (torch.Tensor, optional): Scale factor. Default: None.
        bias (torch.Tensor, optional): Additive bias. Default: None.
        eps (float, optional): Value added to the denominator for numerical stability. Default: 1e-5.

    Returns:
        torch.Tensor: Output tensor of the same shape as input.
    """
    input = _ensure_tensor_dtype(input, weight.data.dtype if weight is not None else None)
    if bias is not None:
        bias.data = _ensure_tensor_dtype(bias, weight.data.dtype if weight is not None else None)
    return original_functional_layer_norm(input, normalized_shape, weight=weight, bias=bias, eps=eps)

# --- Data Type Handling for Tensor Operations ---

def _ensure_matching_dtype(*tensors: torch.Tensor) -> None:
    """Ensures that all input tensors have the same data type as the first tensor.

    This helper function iterates through the provided tensors and converts them to the 
    data type of the first tensor if their data types do not match.

    Args:
        *tensors: A variable number of PyTorch tensors. 
    """
    base_dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        if tensor is not None and tensor.dtype != base_dtype:
            tensor.to(base_dtype)

# --- Training Workaround ---
original_functional_linear = torch.nn.functional.linear

@wraps(torch.nn.functional.linear)
def functional_linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This function overrides the standard `torch.nn.functional.linear` to ensure that the input
    tensor and weight matrix have the same data type. 

    Args:
        input (torch.Tensor): Input tensor of shape (N, *, in_features).
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features).
        bias (torch.Tensor, optional): Bias vector of shape (out_features). Default: None.

    Returns:
        torch.Tensor: Output tensor of shape (N, *, out_features).
    """
    input = _ensure_tensor_dtype(input, weight)  # Ensure data type consistency
    if bias is not None:
        bias.data = _ensure_tensor_dtype(bias, weight.data.dtype)
    return original_functional_linear(input, weight, bias=bias)

# --- Convolution Workaround ---
original_functional_conv2d = torch.nn.functional.conv2d

@wraps(torch.nn.functional.conv2d)
def functional_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, 
                       stride: Union[int, Tuple] = 1, padding: Union[int, Tuple] = 0, 
                       dilation: Union[int, Tuple] = 1, groups: int = 1) -> torch.Tensor:
    """Applies a 2D convolution over an input signal.

    This function overrides the standard `torch.nn.functional.conv2d` to ensure that
    the input tensor, weight tensor, and bias tensor all have the same data type. 

    Args:
        input (torch.Tensor): Input tensor of shape (N, C_in, H_in, W_in).
        weight (torch.Tensor): Filters of shape (C_out, C_in, kernel_size[0], kernel_size[1]).
        bias (torch.Tensor, optional): Optional bias tensor of shape (C_out). Default: None.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.

    Returns:
        torch.Tensor: Output tensor of shape (N, C_out, H_out, W_out).
    """
    input = _ensure_tensor_dtype(input, weight.data.dtype)  # Ensure data type consistency with the weight tensor
    if bias is not None:
        bias.data = _ensure_tensor_dtype(bias, weight.data.dtype) # Ensure bias has the same data type as weight
    return original_functional_conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups) 

# --- Embedding Workaround ---
original_torch_cat = torch.cat

@wraps(torch.cat)
def torch_cat(tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Concatenates the given sequence of tensors in the given dimension.

    This override addresses potential issues with data type mismatches during tensor concatenation
    by converting all tensors to the same data type as the first tensor. 

    Args:
        tensors (sequence of Tensors):  Sequence of tensors to concatenate.
        dim (int, optional): The dimension over which the tensors are concatenated.
        out (Tensor, optional): The output tensor.

    Returns:
        torch.Tensor: The concatenated tensor.
    """
    _ensure_matching_dtype(*tensors) # Make sure all input tensors have matching data types.
    return original_torch_cat(tensors, dim=dim, out=out) 

# --- Padding Workaround ---
original_functional_pad = torch.nn.functional.pad

@wraps(torch.nn.functional.pad)
def functional_pad(input: torch.Tensor, pad: Tuple, mode: str = 'constant', value: Optional[float] = None) -> torch.Tensor:
    """Pads tensor.

    This override handles potential issues with padding operations when using bfloat16 data types
    by converting the input tensor to float32 before padding, and then converting the result back
    to bfloat16.

    Args:
        input (torch.Tensor): N-dimensional tensor.
        pad (tuple): m-elements tuple, where m/2 <= input dimensions and m is even.
        mode (string, optional): 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        value (scalar, optional): Fill value for 'constant' padding. Default: 0

    Returns:
        torch.Tensor: Padded tensor.
    """
    if mode == 'reflect' and input.dtype == torch.bfloat16:  # Check for 'reflect' mode and bfloat16 data type
        input = _ensure_tensor_dtype(input, torch.float32) # Convert to float32
        return original_functional_pad(input, pad, mode=mode, value=value).to(dtype=torch.bfloat16)  # Pad and convert back to bfloat16
    return original_functional_pad(input, pad, mode=mode, value=value)  # Use original padding function 

# --- Tensor Creation Workarounds ---

# Use a helper function to handle device and dtype checks and conversions
def _handle_device_and_dtype(data, dtype=None, device=None):
    """Handles device and dtype conversions for tensor creation functions.

    This helper function checks if the device is a CUDA/XPU device, and if so, 
    converts it to the corresponding XPU device string. It also handles potential
    dtype conversions for XPU devices that don't support FP64.

    Args:
        data (array_like): The input data for tensor creation.
        dtype (torch.dtype, optional):  The desired data type.
        device (torch.device or str or int, optional):  The desired device.

    Returns:
        Tuple[torch.dtype, Union[str, torch.device]]: A tuple containing the potentially
            modified dtype and device.
    """
    if check_device(device):
        device = return_xpu(device)

    if not device_supports_fp64 and _is_xpu_device(device):
        if dtype == torch.float64:
            dtype = torch.float32
        elif dtype is None and hasattr(data, "dtype") and data.dtype in (torch.float64, float):
            dtype = torch.float32
    return dtype, device

# Helper to check if the device is an XPU device
def _is_xpu_device(device) -> bool:
    """Checks if the provided device is an XPU device.

    Args:
        device (torch.device or str or int):  The device to check.

    Returns:
        bool: True if the device is an XPU device, False otherwise.
    """
    return (isinstance(device, torch.device) and device.type == "xpu") or \
           (isinstance(device, str) and "xpu" in device) 

original_torch_tensor = torch.tensor

@wraps(torch.tensor)
def torch_tensor(data, *args, dtype=None, device=None, **kwargs) -> torch.Tensor:
    """
    Constructs a tensor from data, ensuring compatibility with XPU devices.

    This function overrides `torch.tensor` to ensure that tensors are created on the appropriate
    device, particularly handling XPU devices. If the target device is an XPU and the device doesn't
    support FP64, it converts the data type to `float32` for compatibility.

    Args:
        data (array_like): Initial data for the tensor.
        dtype (torch.dtype, optional): The desired data type of the returned tensor.
        device (torch.device, optional): The desired device of the returned tensor.

    Returns:
        torch.Tensor: The constructed tensor.
    """
    dtype, device = _handle_device_and_dtype(data, dtype, device)
    return original_torch_tensor(data, *args, dtype=dtype, device=device, **kwargs)

original_Tensor_to = torch.Tensor.to

@wraps(torch.Tensor.to)
def Tensor_to(self, device=None, *args, **kwargs) -> torch.Tensor:
    """
    Moves the tensor to a specific device, ensuring compatibility with XPU devices.

    This override ensures that tensors are correctly moved to XPU devices when specified.

    Args:
        device (torch.device, optional): The target device. 

    Returns:
        torch.Tensor: The tensor moved to the specified device. 
    """
    if check_device(device):
        device = return_xpu(device)
    return original_Tensor_to(self, device=device, *args, **kwargs)

original_Tensor_cuda = torch.Tensor.cuda

@wraps(torch.Tensor.cuda)
def Tensor_cuda(self, device=None, *args, **kwargs) -> torch.Tensor:
    """
    Moves the tensor to a CUDA or XPU device.

    This override ensures that tensors are correctly moved to XPU devices when specified.

    Args:
        device (torch.device or int, optional): The target device index. If None, the current CUDA
            or XPU device is used.
    Returns:
        torch.Tensor: The tensor moved to the specified device. 
    """
    if check_device(device):
        return original_Tensor_cuda(self, return_xpu(device), *args, **kwargs)
    return original_Tensor_cuda(self, device, *args, **kwargs)

# --- Storage Initialization Workarounds ---

_original_untyped_storage_init = torch.UntypedStorage.__init__

@wraps(torch.UntypedStorage.__init__)
def untyped_storage_init(self: torch.UntypedStorage, *args: tuple, device: Optional[Union[torch.device, int]] = None, **kwargs: dict) -> None:
    """
    Initializes a new storage object, considering XPU device compatibility.

    This function overrides the standard `torch.UntypedStorage.__init__` to ensure
    that new storage objects are created on the correct device, particularly when
    an XPU device is specified. If the specified device is CUDA or XPU, it will be
    converted to an XPU device.

    Args:
        self (torch.UntypedStorage): The storage instance being initialized.
        *args (tuple): Positional arguments for the storage initialization.
        device (torch.device or int, optional): The desired device for the storage object.
        **kwargs (dict): Keyword arguments for the storage initialization.
    """
    if check_device(device):
        device = return_xpu(device)
    super(torch.UntypedStorage, self).__init__(*args, device=device, **kwargs)

_original_untyped_storage_cuda = torch.UntypedStorage.cuda

@wraps(torch.UntypedStorage.cuda)
def untyped_storage_cuda(self: torch.UntypedStorage, device: Optional[Union[torch.device, int]] = None, *args: tuple, **kwargs: dict) -> torch.UntypedStorage:
    """
    Moves the storage to a CUDA or XPU device.

    This function overrides the standard `torch.UntypedStorage.cuda` to ensure that
    the storage is correctly moved to the appropriate device, handling XPU devices
    when specified.

    Args:
        self (torch.UntypedStorage): The storage instance to move.
        device (torch.device or int, optional): The target CUDA or XPU device.
        *args (tuple): Positional arguments for the method.
        **kwargs (dict): Keyword arguments for the method.

    Returns:
        torch.UntypedStorage: The storage object on the new device.
    """
    if check_device(device):
        device = return_xpu(device)
    return super(torch.UntypedStorage, self).cuda(device=device, *args, **kwargs)

# Apply the overrides
torch.UntypedStorage.__init__ = untyped_storage_init
torch.UntypedStorage.cuda = untyped_storage_cuda

# --- Tensor Creation Function Overrides ---

# Helper function to handle device conversion
def _convert_device(device: Optional[Union[torch.device, str, int]]) -> Optional[Union[str, torch.device]]:

    """
    Converts a given device specification to an XPU device specification if needed.

    Args:
        device (Optional[torch.device, str, int]): The device specification to convert.

    Returns:
        Optional[Union[str, torch.device]]: The converted XPU device specification or the original device.
    """
    if check_device(device):
        return return_xpu(device)
    return device




# Modernized function to override tensor creation functions for XPU compatibility
def _override_tensor_creation_fn(original_fn, *args, device=None, **kwargs) -> torch.Tensor:
    """
    Generalized function to override tensor creation functions, ensuring compatibility with XPU devices.

    Args:
        original_fn (callable): The original PyTorch tensor creation function.
        *args: Variable length argument list used to specify the size of the tensor.
        device (torch.device or int, optional): The desired device for the tensor.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        torch.Tensor: The created tensor on the specified device.
    """
    device = _convert_device(device)  # Use the helper function to convert the device if necessary
    return original_fn(*args, device=device, **kwargs)  # Call the original function with the (potentially) updated device

# Store and override tensor creation functions
original_torch_empty = torch.empty
torch.empty = lambda *args, device=None, **kwargs: _override_tensor_creation_fn(original_torch_empty, *args, device=device, **kwargs)

original_torch_randn = torch.randn
torch.randn = lambda *args, device=None, dtype=None, **kwargs: _override_tensor_creation_fn(original_torch_randn, *args, device=device, dtype=dtype, **kwargs)

original_torch_ones = torch.ones
torch.ones = lambda *args, device=None, **kwargs: _override_tensor_creation_fn(original_torch_ones, *args, device=device, **kwargs)

original_torch_zeros = torch.zeros
torch.zeros = lambda *args, device=None, **kwargs: _override_tensor_creation_fn(original_torch_zeros, *args, device=device, **kwargs)

original_torch_linspace = torch.linspace
torch.linspace = lambda *args, device=None, **kwargs: _override_tensor_creation_fn(original_torch_linspace, *args, device=device, **kwargs)

# Override torch.Generator creation
original_torch_Generator = torch.Generator
torch.Generator = lambda device=None: original_torch_Generator(_convert_device(device))

# --- Load Override ---

# Store a reference to the original `torch.load` function for loading objects from files
original_torch_load = torch.load

# Override the function to handle loading tensors onto XPU devices
@wraps(torch.load)
def torch_load(f: Union[str, Path, IO], map_location: Optional[Union[torch.device, str, Callable]] = None, *args, **kwargs) -> Any:
    """
    Loads an object saved with `torch.save` from a file.

    This function overrides the standard `torch.load` to ensure that tensors and other
    objects are correctly loaded onto the specified device, particularly handling XPU devices.

    Args:
        f (file-like object, str, or pathlib.Path): A file-like object, a string, or 
            a Path object containing a file name.
        map_location (torch.device, str, or function, optional): Specifies how to remap 
            storage locations.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Any: The loaded object.
    """
    # Convert to a string if f is a pathlib.Path
    if isinstance(f, Path):
        f = str(f)

    map_location = _map_location_to_xpu(map_location)
    return original_torch_load(f, *args, map_location=map_location, **kwargs)

def _map_location_to_xpu(map_location: Optional[Union[torch.device, str, Callable]]) -> Optional[Union[torch.device, str, Callable]]:
    """
    Converts the given `map_location` to an XPU device if it's specified as CUDA or XPU.

    Args:
        map_location (torch.device, str, or Callable, optional): Specifies how to remap storage locations.

    Returns:
        torch.device, str, or Callable:  The `map_location` argument, potentially remapped to an XPU device.
    """
    if check_device(map_location):  # Check if map_location is a CUDA or XPU device
        return return_xpu(map_location)  # Convert to XPU device specification
    return map_location  # If not a CUDA/XPU device, return the original map_location

# --- Hijack Functions ---
def ipex_hijacks():
    """
    Replaces specific PyTorch functions with their XPU-compatible versions.

    This function should be called to activate the compatibility layer for XPU devices, 
    ensuring that subsequent PyTorch operations are handled correctly on XPUs. 
    """

    logger.info("Applying XPU compatibility patches to PyTorch...")

    _override_tensor_creation_functions()
    _override_tensor_movement_functions()
    _override_storage_initialization_functions()
    _override_backend_functions()
    _override_neural_network_functions()
    _override_numpy_conversion_functions()

    logger.info("PyTorch XPU compatibility layer activated.")

def _override_tensor_creation_functions():
    """Overrides PyTorch tensor creation functions for XPU compatibility."""
    torch.tensor = torch_tensor
    torch.empty = torch_empty
    torch.randn = torch_randn
    torch.ones = torch_ones
    torch.zeros = torch_zeros
    torch.linspace = torch_linspace
    torch.Generator = torch_Generator

def _override_tensor_movement_functions():
    """Overrides functions for moving tensors to XPUs."""
    torch.Tensor.to = Tensor_to
    torch.Tensor.cuda = Tensor_cuda

def _override_storage_initialization_functions():
    """Overrides storage initialization functions for XPU compatibility."""
    torch.UntypedStorage.__init__ = UntypedStorage_init
    torch.UntypedStorage.cuda = UntypedStorage_cuda

def _override_backend_functions():
    """Overrides PyTorch backend functions for XPU compatibility."""
    torch.backends.cuda.sdp_kernel = return_null_context
    torch.UntypedStorage.is_cuda = is_cuda
    torch.cuda.is_available = is_available
    torch.amp.autocast_mode.autocast.__init__ = autocast_init

def _override_neural_network_functions():
    """Overrides PyTorch neural network functions for XPU compatibility."""
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention
    torch.nn.functional.group_norm = functional_group_norm
    torch.nn.functional.layer_norm = functional_layer_norm
    torch.nn.functional.linear = functional_linear
    torch.nn.functional.conv2d = functional_conv2d
    torch.nn.functional.interpolate = interpolate
    torch.nn.functional.pad = functional_pad

    torch.bmm = torch_bmm
    torch.cat = torch_cat

def _override_numpy_conversion_functions():
    """Overrides NumPy array conversion functions if FP64 is not supported."""
    if not device_supports_fp64:
        torch.from_numpy = from_numpy
        torch.as_tensor = as_tensor 