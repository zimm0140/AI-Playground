## code credit goes to https://github.com/vladmandic/automatic/blob/master/modules/intel/ipex/hijacks.py
##

import os
from functools import wraps
from contextlib import nullcontext
import torch
import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import
import numpy as np
from typing import Union, Optional, Tuple

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
    Checks if an XPU device is available using the Intel Extension for PyTorch (IPEX).

    This function overrides `torch.cuda.is_available` to check for XPU availability instead of CUDA.

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
        return "cuda" in device or "xpu" in device
    if isinstance(device, int):
        return True  # Assuming int is used for device index
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

# Override autocast initialization to use bfloat16 for XPU devices
@wraps(torch.amp.autocast_mode.autocast.__init__)
def autocast_init(self, device_type, dtype=None, enabled=True, cache_enabled=None):
    """
    Initializes the autocast context manager.

    If the device type is 'cuda' or 'xpu' and no dtype is specified, the default dtype
    is set to `torch.bfloat16` (bfloat16) for XPU devices.

    This override ensures that autocasting on XPUs uses bfloat16, which is often
    preferred for its memory efficiency on these devices.
    """
    if device_type in ("cuda", "xpu"):  # Use 'in' operator for set membership check
        dtype = dtype or torch.bfloat16  # Use the `or` operator to simplify default dtype assignment
        return original_autocast_init(self, device_type="xpu", dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
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

# Store a reference to the original `torch.from_numpy` function
original_from_numpy = torch.from_numpy

# Override `torch.from_numpy` to convert FP64 (double-precision) to FP32 (single-precision)
@wraps(torch.from_numpy)
def from_numpy(ndarray):
    """
    Creates a tensor from a NumPy ndarray.

    This function overrides the standard `torch.from_numpy` to handle potential data type
    incompatibilities with certain XPU architectures. If the NumPy array's data type
    is `float`, it is converted to `float32` before creating the PyTorch tensor.

    Args:
        ndarray (numpy.ndarray): The NumPy array to convert to a tensor.

    Returns:
        torch.Tensor: The tensor created from the NumPy array.
    """
    if ndarray.dtype == float:
        ndarray = ndarray.astype('float32')  # Convert to float32 if the input is a float
    return original_from_numpy(ndarray)

# Store a reference to the original `torch.as_tensor` function
original_as_tensor = torch.as_tensor

# Override `torch.as_tensor` to handle device and data type conversions
@wraps(torch.as_tensor)
def as_tensor(data, dtype=None, device=None):
    """
    Creates a tensor from the input data.

    This function overrides `torch.as_tensor` to ensure compatibility with XPU devices and to handle
    potential data type issues. If the input `data` is a NumPy array with a `float` data type and
    the target device is not explicitly set to "cpu", the array is converted to `float32`.

    Args:
        data (array_like): Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar, and other types.
        dtype (torch.dtype, optional): The desired data type of returned tensor. Default: if None, infers data type from data.
        device (torch.device, optional): The desired device of returned tensor. Default: if None, uses the current device for the default tensor type.

    Returns:
        torch.Tensor: The tensor created from the input data.
    """
    if check_device(device):
        device = return_xpu(device)
    if isinstance(data, np.ndarray) and data.dtype == float and not (
        (isinstance(device, torch.device) and device.type == "cpu") or (isinstance(device, str) and "cpu" in device)):
        data = data.astype('float32')
    return original_as_tensor(data, dtype=dtype, device=device)

# --- 32-Bit Attention Workarounds ---

# Retrieve the value of the environment variable 'IPEX_FORCE_ATTENTION_SLICE'
# This variable is used to force the use of 32-bit attention operations even if the device supports FP64. 
IPEX_FORCE_ATTENTION_SLICE = os.getenv('IPEX_FORCE_ATTENTION_SLICE')

# Conditionally select the appropriate functions for `torch.bmm` and `scaled_dot_product_attention` 
# based on device support for FP64 and the environment variable.
if device_supports_fp64 and IPEX_FORCE_ATTENTION_SLICE is None:
    # Use the original PyTorch functions if the device supports FP64 and slicing is not enforced.
    original_torch_bmm = torch.bmm
    original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
else:
    # If the device lacks FP64 support or slicing is enforced, attempt to load the optimized 32-bit versions
    try:
        from attention import torch_bmm_32_bit as original_torch_bmm
        from attention import scaled_dot_product_attention_32_bit as original_scaled_dot_product_attention
    except ImportError:  # pylint: disable=broad-exception-caught
        # If loading the 32-bit versions fails, use the original PyTorch functions as a fallback
        logger.warning("32-bit attention functions not found, falling back to original functions.")
        original_torch_bmm = torch.bmm
        original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention

# --- Data Type Errors: ---

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
    if input.dtype != mat2.dtype:  # Check if input tensors have the same data type
        mat2 = mat2.to(input.dtype)  # Convert mat2 to the data type of input if they don't match
    return original_torch_bmm(input, mat2, out=out)  # Call the original torch.bmm function

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
    key = key.to(dtype=query.dtype) if query.dtype != key.dtype else key
    value = value.to(dtype=query.dtype) if query.dtype != value.dtype else value
    if attn_mask is not None and query.dtype != attn_mask.dtype:
        attn_mask = attn_mask.to(dtype=query.dtype)
    return original_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)

# --- A1111 FP16 Workaround ---
original_functional_group_norm = torch.nn.functional.group_norm

# Override `torch.nn.functional.group_norm` to ensure data type consistency for FP16 operations
@wraps(torch.nn.functional.group_norm)
def functional_group_norm(input: torch.Tensor, num_groups: int, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None, eps: float = 1e-05) -> torch.Tensor:
    """
    Applies Group Normalization over a mini-batch of inputs, ensuring data type compatibility.

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
    if weight is not None and input.dtype != weight.data.dtype:
        input = input.to(dtype=weight.data.dtype)
    if bias is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_group_norm(input, num_groups, weight=weight, bias=bias, eps=eps)

# --- A1111 BF16 Workaround ---
original_functional_layer_norm = torch.nn.functional.layer_norm

# Override `torch.nn.functional.layer_norm` to handle data type consistency for BF16 operations
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
    if weight is not None and input.dtype != weight.data.dtype:
        input = input.to(dtype=weight.data.dtype)
    if bias is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_layer_norm(input, normalized_shape, weight=weight, bias=bias, eps=eps)

# --- Training Workaround ---

# Store the original torch.nn.functional.linear for later use
original_functional_linear = torch.nn.functional.linear

# Override the linear function to ensure data type consistency during training
@wraps(torch.nn.functional.linear)
def functional_linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This function overrides the standard `torch.nn.functional.linear` function to ensure that 
    the input tensor and the weight matrix have the same data type. If the types do not match,
    the input tensor is converted to the data type of the weight matrix. This is particularly
    important during training to prevent data type errors. 

    Args:
        input (torch.Tensor): Input tensor of shape (N, *, in_features).
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features).
        bias (torch.Tensor, optional): Bias vector of shape (out_features). Default: None.

    Returns:
        torch.Tensor: Output tensor of shape (N, *, out_features).
    """
    if input.dtype != weight.data.dtype:  # Check if data types match
        input = input.to(dtype=weight.data.dtype)  # Convert the input tensor to the weight's data type if necessary
    if bias is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)  # Convert the bias tensor if necessary
    return original_functional_linear(input, weight, bias=bias)  # Call the original linear function


# --- Convolution Workaround ---

# Store the original `torch.nn.functional.conv2d` function
original_functional_conv2d = torch.nn.functional.conv2d

# Override the 2D convolution function to ensure data type consistency 
@wraps(torch.nn.functional.conv2d)
def functional_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, 
                       stride: Union[int, Tuple] = 1, padding: Union[int, Tuple] = 0, 
                       dilation: Union[int, Tuple] = 1, groups: int = 1) -> torch.Tensor:
    """
    Applies a 2D convolution over an input signal composed of several input planes.

    This function overrides the standard `torch.nn.functional.conv2d` function to ensure 
    that the input tensor, weight tensor (filter), and bias tensor (if provided) have 
    the same data type. If the data types do not match, the input tensor and the bias 
    tensor are converted to the data type of the weight tensor.

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
    if input.dtype != weight.data.dtype:  # Check if input and weight data types match
        input = input.to(dtype=weight.data.dtype)  # Convert the input to the weight's data type if necessary
    if bias is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)  # Convert the bias if necessary
    return original_functional_conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)  # Call the original conv2d function

# --- Embedding Workaround ---

# Store a reference to the original `torch.cat` function
original_torch_cat = torch.cat

# Override the tensor concatenation function to handle potential data type mismatches
@wraps(torch.cat)
def torch_cat(tensor, *args, **kwargs) -> torch.Tensor:
    """
    Concatenates the given sequence of seq tensors in the given dimension.

    This override addresses potential issues with data type mismatches during tensor concatenation
    by converting tensors to the same data type as the second tensor. This is particularly relevant
    when working with bfloat16 (BF16) precision.

    Args:
        tensors (sequence of Tensors): Any Python sequence of tensors of the same type. Non-tensor arguments will
                                        be cast to tensors before concatenation.
        dim (int, optional): The dimension over which the tensors are concatenated.

    Keyword args:
        out (Tensor, optional): The output tensor.

    Returns:
        torch.Tensor: The concatenated tensor.
    """
    # If there are three tensors and their data types don't match, convert the first and third to the type of the second.
    if len(tensor) == 3 and (tensor[0].dtype != tensor[1].dtype or tensor[2].dtype != tensor[1].dtype):
        return original_torch_cat([tensor[0].to(tensor[1].dtype), tensor[1], tensor[2].to(tensor[1].dtype)], *args, **kwargs)
    return original_torch_cat(tensor, *args, **kwargs)  # If data types match or there are not three tensors, use original torch.cat

# --- Padding Workaround ---

# Store a reference to the original `torch.nn.functional.pad` function
original_functional_pad = torch.nn.functional.pad

# Override the padding function to handle potential issues with bfloat16 (BF16)
@wraps(torch.nn.functional.pad)
def functional_pad(input: torch.Tensor, pad: Tuple, mode: str = 'constant', value: Optional[float] = None) -> torch.Tensor:
    """
    Pads tensor.

    This override handles potential issues with padding operations when using bfloat16 data types
    by converting the input tensor to float32 before padding, and then converting the result back
    to bfloat16.

    Args:
        input (torch.Tensor): N-dimensional tensor.
        pad (tuple): m-elements tuple, where m/2 <= input dimensions and m is even.
        mode (string, optional): 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'.
        value (scalar, optional): Fill value for 'constant' padding. Default: 0.

    Returns:
        torch.Tensor: Padded tensor.
    """
    if mode == 'reflect' and input.dtype == torch.bfloat16:  # Check for 'reflect' mode and bfloat16 data type
        return original_functional_pad(input.to(torch.float32), pad, mode=mode, value=value).to(dtype=torch.bfloat16)  # Convert to float32, pad, then convert back to bfloat16
    return original_functional_pad(input, pad, mode=mode, value=value)  # Use original padding function if not in 'reflect' mode or not bfloat16

# --- Tensor Creation Workarounds ---

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
    if check_device(device):
        device = return_xpu(device)

    # Check if dtype needs to be converted to float32 for XPU compatibility
    if not device_supports_fp64:
        if isinstance(device, torch.device) and device.type == "xpu" or isinstance(device, str) and "xpu" in device:
            if dtype == torch.float64:
                dtype = torch.float32
            elif dtype is None and hasattr(data, "dtype") and data.dtype in (torch.float64, float):
                dtype = torch.float32

    return original_torch_tensor(data, *args, dtype=dtype, device=device, **kwargs)

original_Tensor_to = torch.Tensor.to

@wraps(torch.Tensor.to)
def Tensor_to(self, device=None, *args, **kwargs):
    """
    Moves the tensor to a specific device, ensuring compatibility with XPU devices.

    This override ensures that tensors are correctly moved to XPU devices when specified.

    Args:
        device (torch.device, optional): The target device. 

    Returns:
        torch.Tensor: The tensor moved to the specified device. 
    """
    if check_device(device):
        return original_Tensor_to(self, return_xpu(device), *args, **kwargs)
    return original_Tensor_to(self, device, *args, **kwargs)

original_Tensor_cuda = torch.Tensor.cuda

@wraps(torch.Tensor.cuda)
def Tensor_cuda(self, device=None, *args, **kwargs):
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

original_UntypedStorage_init = torch.UntypedStorage.__init__

@wraps(torch.UntypedStorage.__init__)
def UntypedStorage_init(*args, device=None, **kwargs):
    """
    Initializes a new storage object, considering XPU device compatibility.

    This function overrides the standard `torch.UntypedStorage.__init__` to ensure
    that new storage objects are created on the correct device, particularly when
    an XPU device is specified. If the specified device is CUDA or XPU, it will be
    converted to an XPU device. 

    Args:
        *args: Variable length argument list.
        device (torch.device or int, optional): The desired device for the storage object.
        **kwargs: Arbitrary keyword arguments.
    """
    if check_device(device):
        device = return_xpu(device)
    return original_UntypedStorage_init(*args, device=device, **kwargs)

original_UntypedStorage_cuda = torch.UntypedStorage.cuda

@wraps(torch.UntypedStorage.cuda)
def UntypedStorage_cuda(self, device=None, *args, **kwargs):
    """
    Moves the storage to a CUDA or XPU device.

    This function overrides the standard `torch.UntypedStorage.cuda` to ensure that 
    the storage is correctly moved to the appropriate device, handling XPU devices 
    when specified. 

    Args:
        device (torch.device or int, optional): The target CUDA or XPU device. 
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """
    if check_device(device):
        return original_UntypedStorage_cuda(self, return_xpu(device), *args, **kwargs)
    return original_UntypedStorage_cuda(self, device, *args, **kwargs)

# --- Tensor Creation Function Overrides ---

# Store a reference to the original `torch.empty` function
original_torch_empty = torch.empty

# Override the function for creating an empty tensor to support XPUs
@wraps(torch.empty)
def torch_empty(*args, device=None, **kwargs) -> torch.Tensor:
    """
    Returns a tensor filled with uninitialized data.

    This function overrides `torch.empty` to ensure that empty tensors are created on
    the correct device, particularly when an XPU device is specified.

    Args:
        *args:  Variable length argument list used to specify the size of the tensor.
        device (torch.device or int, optional): The desired device for the tensor.
        **kwargs: Arbitrary keyword arguments. 

    Returns:
        torch.Tensor:  A tensor filled with uninitialized data on the specified device.
    """
    if check_device(device):
        return original_torch_empty(*args, device=return_xpu(device), **kwargs)
    return original_torch_empty(*args, device=device, **kwargs)

# Store the original `torch.randn` function
original_torch_randn = torch.randn

# Override the function for creating a tensor with random values from a normal distribution
@wraps(torch.randn)
def torch_randn(*args, device=None, dtype=None, **kwargs) -> torch.Tensor:
    """
    Returns a tensor filled with random numbers from a standard normal distribution.

    This override ensures that the tensor is created on the appropriate device, handling
    XPU devices correctly.

    Args:
        *args:  Variable length argument list used to specify the size of the tensor.
        device (torch.device or int, optional): The desired device for the tensor.
        dtype (torch.dtype, optional):  The desired data type of the tensor.
        **kwargs:  Arbitrary keyword arguments.

    Returns:
        torch.Tensor: The created tensor filled with random numbers from a normal distribution.
    """
    if dtype == bytes:  # Handle the case where dtype is 'bytes', setting it to None to avoid errors
        dtype = None
    if check_device(device):  # Check if the device is a CUDA or XPU device
        return original_torch_randn(*args, device=return_xpu(device), **kwargs)  # Create the tensor on the appropriate XPU device
    return original_torch_randn(*args, device=device, **kwargs)  # If not a CUDA/XPU device, use the original function

# Store the original `torch.ones` function
original_torch_ones = torch.ones

# Override the function for creating a tensor filled with ones to support XPUs
@wraps(torch.ones)
def torch_ones(*args, device=None, **kwargs) -> torch.Tensor:
    """
    Returns a tensor filled with the scalar value 1.

    This function overrides `torch.ones` to ensure that the tensor is created on the 
    correct device, specifically handling XPU devices. 

    Args:
        *args:  Variable length argument list used to specify the size of the tensor.
        device (torch.device or int, optional): The desired device for the tensor.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        torch.Tensor: A tensor filled with the scalar value 1 on the specified device.
    """
    if check_device(device):
        return original_torch_ones(*args, device=return_xpu(device), **kwargs) 
    return original_torch_ones(*args, device=device, **kwargs)  

# Store the original `torch.zeros` function
original_torch_zeros = torch.zeros

# Override the function for creating a tensor filled with zeros to support XPUs
@wraps(torch.zeros)
def torch_zeros(*args, device=None, **kwargs) -> torch.Tensor:
    """
    Returns a tensor filled with the scalar value 0.

    This function overrides `torch.zeros` to ensure that the tensor is created on the 
    correct device, specifically handling XPU devices. 

    Args:
        *args:  Variable length argument list used to specify the size of the tensor.
        device (torch.device or int, optional): The desired device for the tensor.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        torch.Tensor: A tensor filled with the scalar value 0 on the specified device.
    """
    if check_device(device):
        return original_torch_zeros(*args, device=return_xpu(device), **kwargs)
    return original_torch_zeros(*args, device=device, **kwargs)

# Store the original `torch.linspace` function
original_torch_linspace = torch.linspace

# Override the function for creating a tensor of evenly spaced values to support XPUs
@wraps(torch.linspace)
def torch_linspace(*args, device=None, **kwargs) -> torch.Tensor:
    """
    Returns a one-dimensional tensor of evenly spaced values.

    This function overrides `torch.linspace` to ensure that the tensor is created on the 
    correct device, specifically handling XPU devices.

    Args:
        *args: Variable length argument list to specify the start, end, steps, etc. for the linspace.
        device (torch.device or int, optional): The desired device for the tensor.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        torch.Tensor: A one-dimensional tensor of evenly spaced values on the specified device. 
    """
    if check_device(device):  # Check if the device is for CUDA or XPU
        return original_torch_linspace(*args, device=return_xpu(device), **kwargs)  # Create the linspace tensor on the corresponding XPU
    return original_torch_linspace(*args, device=device, **kwargs)  # Otherwise, create the tensor on the specified device

# Override `torch.Generator` to ensure it is created on the appropriate device (XPU or CUDA)
original_torch_Generator = torch.Generator

@wraps(torch.Generator)
def torch_Generator(device=None):
    """
    Creates a pseudo-random number generator object.

    This function overrides `torch.Generator` to ensure that the generator is created on
    the correct device, handling both CUDA and XPU devices.

    Args:
        device (torch.device or int, optional): The desired device for the generator.

    Returns:
        torch.Generator: A pseudo-random number generator object on the specified device. 
    """
    if check_device(device):  # Check if the device is for CUDA or XPU
        return original_torch_Generator(return_xpu(device)) # Create the generator on the corresponding XPU device
    return original_torch_Generator(device)  # Otherwise, create the generator on the specified device

# --- Load Override ---

# Store a reference to the original `torch.load` function for loading objects from files
original_torch_load = torch.load

# Override the function to handle loading tensors onto XPU devices
@wraps(torch.load)
def torch_load(f, map_location=None, *args, **kwargs):
    """
    Loads an object saved with `torch.save` from a file.

    This function overrides the standard `torch.load` to ensure that tensors and other
    objects are correctly loaded onto the specified device, particularly handling XPU devices. 

    If the specified `map_location` is a CUDA or XPU device, the function will remap the 
    storage location to the corresponding XPU device using `return_xpu(map_location)`.

    Args:
        f (file-like object, str, or path-like object): A file-like object, a string, or 
            a path-like object containing a file name.
        map_location (torch.device, str, or function, optional):  Specifies how to remap 
            storage locations.
        *args:  Variable length argument list.
        **kwargs:  Arbitrary keyword arguments.

    Returns:
        Any: The loaded object. 
    """
    if check_device(map_location):
        return original_torch_load(f, *args, map_location=return_xpu(map_location), **kwargs)
    return original_torch_load(f, *args, map_location=map_location, **kwargs)


# --- Hijack Functions ---
def ipex_hijacks():
    """
    Replaces specific PyTorch functions with their XPU-compatible versions.

    This function should be called to activate the compatibility layer for XPU devices, ensuring that
    subsequent PyTorch operations are correctly routed to XPUs and handle data types and other
    potential compatibility issues.
    """

    # Override core tensor creation and manipulation functions
    torch.tensor = torch_tensor
    torch.Tensor.to = Tensor_to
    torch.Tensor.cuda = Tensor_cuda
    torch.UntypedStorage.__init__ = UntypedStorage_init
    torch.UntypedStorage.cuda = UntypedStorage_cuda
    torch.empty = torch_empty
    torch.randn = torch_randn
    torch.ones = torch_ones
    torch.zeros = torch_zeros
    torch.linspace = torch_linspace
    torch.Generator = torch_Generator
    torch.load = torch_load

    # Override specific PyTorch backend functions
    torch.backends.cuda.sdp_kernel = return_null_context
    torch.UntypedStorage.is_cuda = is_cuda
    torch.cuda.is_available = is_available
    torch.amp.autocast_mode.autocast.__init__ = autocast_init

    # Override PyTorch functions for neural network operations
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention
    torch.nn.functional.group_norm = functional_group_norm
    torch.nn.functional.layer_norm = functional_layer_norm
    torch.nn.functional.linear = functional_linear
    torch.nn.functional.conv2d = functional_conv2d
    torch.nn.functional.interpolate = interpolate
    torch.nn.functional.pad = functional_pad

    torch.bmm = torch_bmm
    torch.cat = torch_cat

    # Override NumPy array conversion functions if FP64 is not supported on the device
    if not device_supports_fp64:
        torch.from_numpy = from_numpy
        torch.as_tensor = as_tensor