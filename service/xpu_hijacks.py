## code credit goes to https://github.com/vladmandic/automatic/blob/master/modules/intel/ipex/hijacks.py
##

import os
from functools import wraps
from contextlib import nullcontext
import torch
import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import
import numpy as np

"""
PyTorch Compatibility Layer for Intel XPUs

This module provides a compatibility layer to ensure that PyTorch code designed for CUDA devices 
can run smoothly on Intel XPU devices. It accomplishes this by hijacking and overriding various
PyTorch functions, properties, and attributes, adapting them to work with XPU-specific features
and limitations.

Key features of this module include:

- Redirecting CUDA Operations to XPU:  It globally redirects PyTorch's CUDA functions and 
  attributes to use Intel XPUs. This enables compatibility with existing CUDA-based PyTorch code.
- Data Type Handling:  It handles data type conversions and ensures consistency in data types across 
  different operations, particularly when working with XPUs that might have limitations on supported data types.
- Workarounds for XPU Limitations:  It implements specific workarounds for known issues with certain
  XPU architectures (e.g., Alchemist GPUs), such as lack of FP64 support or limitations in specific 
  operations.
- Performance Optimization:  It includes optimizations for specific operations (e.g., attention mechanisms)
  to improve performance on XPUs, potentially using techniques like slicing or CPU offloading. 
- Preservation of Function Metadata: It uses the `@wraps` decorator to preserve the documentation
  and metadata of the original PyTorch functions being overridden, ensuring clarity and consistency.

Usage: 
    To enable these compatibility patches, simply import this module and call the `ipex_hijacks()` function.
"""

# Check if the current XPU device supports 64-bit floating point precision (FP64)
device_supports_fp64 = torch.xpu.has_fp64_dtype()

# Globally redirect PyTorch CUDA operations to use XPU operations
torch.cuda = torch.xpu

def return_null_context(*args, **kwargs):  # pylint: disable=unused-argument
    """
    Returns a null context manager.

    This is a utility function to disable certain context-based operations, such as CUDA's SDP kernel, 
    when they are not needed or supported on XPUs.
    """
    return nullcontext()

@wraps(torch.cuda.is_available)
def is_available():
    """
    Checks if an XPU device is available using the Intel Extension for PyTorch (IPEX).

    This function overrides `torch.cuda.is_available` to check for XPU availability instead of CUDA.

    Returns:
        bool: True if an XPU device is available, False otherwise.
    """
    return ipex.has_xpu()

@property
def is_cuda(self):
    """
    Checks if a tensor's device is either an XPU or a CUDA device.

    This property is added to PyTorch's `Tensor` object to provide a unified way 
    to check if a tensor is on a device suitable for acceleration (either CUDA or XPU).

    Returns:
        bool: True if the tensor's device is 'xpu' or 'cuda', False otherwise.
    """
    return self.device.type == 'xpu' or self.device.type == 'cuda'

def check_device(device):
    """
    Checks if the provided device specification is for a CUDA or XPU device.

    Args:
        device (torch.device or str or int): The device specification to check.

    Returns:
        bool: True if the device specification is for 'cuda' or 'xpu', False otherwise.
    """
    return bool((isinstance(device, torch.device) and device.type == "cuda") or 
                (isinstance(device, str) and "cuda" in device) or isinstance(device, int))

def return_xpu(device):
    """
    Converts a device specification to a corresponding XPU device string.

    This function handles different types of input for the `device` argument and 
    returns a standardized XPU device string.

    Args:
        device (torch.device or str or int): The device specification to convert.

    Returns:
        str: The corresponding XPU device string (e.g., "xpu:0").
    """
    if isinstance(device, str) and ":" in device:
        return f"xpu:{device.split(':')[-1]}"
    elif isinstance(device, int):
        return f"xpu:{device}"
    elif isinstance(device, torch.device):
        return torch.device("xpu")
    else:
        return "xpu"


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
    if device_type == "cuda" or device_type == "xpu":
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
    else:
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
        return original_from_numpy(ndarray.astype('float32'))
    else:
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
        return original_as_tensor(data, dtype=torch.float32, device=device)
    else:
        return original_as_tensor(data, dtype=dtype, device=device)


# --- 32-Bit Attention Workarounds ---

# Conditionally select the appropriate functions for bmm and scaled dot-product attention
if device_supports_fp64 and os.environ.get('IPEX_FORCE_ATTENTION_SLICE', None) is None:
    # Use the original PyTorch functions if the device supports FP64 and slicing is not enforced.
    original_torch_bmm = torch.bmm
    original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
else:
    # If the device lacks FP64 support or slicing is enforced, attempt to load the optimized 32-bit versions
    try:
        from attention import torch_bmm_32_bit as original_torch_bmm
        from attention import scaled_dot_product_attention_32_bit as original_scaled_dot_product_attention
    except Exception:  # pylint: disable=broad-exception-caught
        # If loading the 32-bit versions fails, use the original PyTorch functions as a fallback
        original_torch_bmm = torch.bmm
        original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention

# --- Data Type Errors: ---

# Override `torch.bmm` to ensure matching data types for inputs
@wraps(torch.bmm)
def torch_bmm(input, mat2, *, out=None):
    """
    Performs a batch matrix-matrix product of matrices stored in input and mat2.

    This function overrides the standard `torch.bmm` to ensure that both input tensors
    have the same data type before performing the operation. It converts `mat2` to the
    data type of `input` if necessary.

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
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    """
    Computes scaled dot product attention on input tensors.

    This function ensures that the query, key, value, and attention mask (if provided) tensors all have 
    the same data type. It converts the tensors to the data type of the `query` tensor if necessary.

    Args:
        query (torch.Tensor): Query tensor.
        key (torch.Tensor): Key tensor.
        value (torch.Tensor): Value tensor.
        attn_mask (torch.Tensor, optional): Optional attention mask tensor. Default: None.
        dropout_p (float, optional): Dropout probability. Default: 0.0
        is_causal (bool, optional): If True, applies a causal mask to the attention scores. Default: False

    Returns:
        torch.Tensor: The output tensor resulting from the scaled dot product attention operation.
    """
    if query.dtype != key.dtype:
        key = key.to(dtype=query.dtype)
    if query.dtype != value.dtype:
        value = value.to(dtype=query.dtype)
    if attn_mask is not None and query.dtype != attn_mask.dtype:
        attn_mask = attn_mask.to(dtype=query.dtype)
    return original_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

# --- A1111 FP16 Workaround ---
original_functional_group_norm = torch.nn.functional.group_norm

# Override `torch.nn.functional.group_norm` to ensure data type consistency for FP16 operations
@wraps(torch.nn.functional.group_norm)
def functional_group_norm(input, num_groups, weight=None, bias=None, eps=1e-05):
    """
    Applies Group Normalization over a mini-batch of inputs.

    This override ensures data type compatibility between input, weight, and bias, potentially converting
    the input tensor to the data type of the weight tensor. 

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
    if bias is not None and weight is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_group_norm(input, num_groups, weight=weight, bias=bias, eps=eps)

# --- A1111 BF16 Workaround ---
original_functional_layer_norm = torch.nn.functional.layer_norm

# Override `torch.nn.functional.layer_norm` to handle data type consistency for BF16 operations
@wraps(torch.nn.functional.layer_norm)
def functional_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    """
    Applies Layer Normalization over a mini-batch of inputs.

    This override ensures data type consistency between input, weight, and bias, converting the input
    to the data type of the weight if necessary. 

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
    if bias is not None and weight is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_layer_norm(input, normalized_shape, weight=weight, bias=bias, eps=eps)

# --- Training Workaround ---
original_functional_linear = torch.nn.functional.linear

# Override `torch.nn.functional.linear` to ensure data type consistency during training
@wraps(torch.nn.functional.linear)
def functional_linear(input, weight, bias=None):
    """
    Applies a linear transformation to the incoming data.

    This override ensures data type compatibility between the input, weight, and bias 
    tensors during training, converting tensors as necessary.

    Args:
        input (torch.Tensor): Input tensor of shape (N, *, in_features).
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features).
        bias (torch.Tensor, optional): Bias vector of shape (out_features). Default: None.

    Returns:
        torch.Tensor: Output tensor of shape (N, *, out_features).
    """
    if input.dtype != weight.data.dtype:
        input = input.to(dtype=weight.data.dtype)
    if bias is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_linear(input, weight, bias=bias)

# Store a reference to the original `torch.nn.functional.conv2d`
original_functional_conv2d = torch.nn.functional.conv2d

# Override `torch.nn.functional.conv2d` for data type compatibility during training
@wraps(torch.nn.functional.conv2d)
def functional_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Applies a 2D convolution over an input signal.

    This override ensures data type compatibility between the input tensor, weights, and bias
    during training, converting tensors as necessary.

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
    if input.dtype != weight.data.dtype:
        input = input.to(dtype=weight.data.dtype)
    if bias is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

# --- A1111 Embedding BF16 Workaround ---
original_torch_cat = torch.cat

# Override `torch.cat` to handle potential data type mismatches during tensor concatenation
@wraps(torch.cat)
def torch_cat(tensor, *args, **kwargs):
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
    if len(tensor) == 3 and (tensor[0].dtype != tensor[1].dtype or tensor[2].dtype != tensor[1].dtype):
        return original_torch_cat([tensor[0].to(tensor[1].dtype), tensor[1], tensor[2].to(tensor[1].dtype)], *args, **kwargs)
    else:
        return original_torch_cat(tensor, *args, **kwargs)

# --- SwinIR BF16: ---
original_functional_pad = torch.nn.functional.pad

# Override `torch.nn.functional.pad` to ensure compatibility with bfloat16 (BF16) data types
@wraps(torch.nn.functional.pad)
def functional_pad(input, pad, mode='constant', value=None):
    """
    Pads tensor.

    This override handles potential issues with padding operations when using bfloat16 data types
    by converting the input tensor to float32 before padding, and then converting the result back
    to bfloat16.

    Args:
        input (torch.Tensor): N-dimensional tensor
        pad (tuple): m-elements tuple, where m/2 <= input dimensions and m is even.
        mode (string, optional): 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        value (scalar, optional): Fill value for 'constant' padding. Default: 0

    Returns:
        torch.Tensor: Padded tensor.
    """
    if mode == 'reflect' and input.dtype == torch.bfloat16:
        return original_functional_pad(input.to(torch.float32), pad, mode=mode, value=value).to(dtype=torch.bfloat16)
    else:
        return original_functional_pad(input, pad, mode=mode, value=value)


original_torch_tensor = torch.tensor

# Override `torch.tensor` to handle device selection and potential data type conversion to float32 on XPUs
@wraps(torch.tensor)
def torch_tensor(data, *args, dtype=None, device=None, **kwargs):
    """
    Constructs a tensor from data.

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
    if not device_supports_fp64:
        if (isinstance(device, torch.device) and device.type == "xpu") or (isinstance(device, str) and "xpu" in device):
            if dtype == torch.float64:
                dtype = torch.float32
            elif dtype is None and (hasattr(data, "dtype") and (data.dtype == torch.float64 or data.dtype == float)):
                dtype = torch.float32
    return original_torch_tensor(data, *args, dtype=dtype, device=device, **kwargs)

# Override `torch.Tensor.to` to handle device conversion, ensuring that tensors are moved to XPU if needed
original_Tensor_to = torch.Tensor.to
@wraps(torch.Tensor.to)
def Tensor_to(self, device=None, *args, **kwargs):
    """
    Moves the tensor to a specific device.

    Args:
        device (torch.device, optional): The target device. 

    Returns:
        torch.Tensor: The tensor moved to the specified device. 
    """
    if check_device(device):
        return original_Tensor_to(self, return_xpu(device), *args, **kwargs)
    else:
        return original_Tensor_to(self, device, *args, **kwargs)

# Override `torch.Tensor.cuda` to redirect the tensor to the XPU device
original_Tensor_cuda = torch.Tensor.cuda
@wraps(torch.Tensor.cuda)
def Tensor_cuda(self, device=None, *args, **kwargs):
    """
    Moves the tensor to a CUDA or XPU device.

    Args:
        device (torch.device or int, optional): The target device index. If None, the current CUDA
            or XPU device is used.
    Returns:
        torch.Tensor: The tensor moved to the specified device. 
    """
    if check_device(device):
        return original_Tensor_cuda(self, return_xpu(device), *args, **kwargs)
    else:
        return original_Tensor_cuda(self, device, *args, **kwargs)

# Override `torch.UntypedStorage.__init__` to handle device placement during storage initialization
original_UntypedStorage_init = torch.UntypedStorage.__init__
@wraps(torch.UntypedStorage.__init__)
def UntypedStorage_init(*args, device=None, **kwargs):
    """
    Initializes a new storage object.
    """
    if check_device(device):
        return original_UntypedStorage_init(*args, device=return_xpu(device), **kwargs)
    else:
        return original_UntypedStorage_init(*args, device=device, **kwargs)

# Override `torch.UntypedStorage.cuda` to move storage to XPU if needed
original_UntypedStorage_cuda = torch.UntypedStorage.cuda
@wraps(torch.UntypedStorage.cuda)
def UntypedStorage_cuda(self, device=None, *args, **kwargs):
    """
    Moves the storage to a CUDA or XPU device.
    """
    if check_device(device):
        return original_UntypedStorage_cuda(self, return_xpu(device), *args, **kwargs)
    else:
        return original_UntypedStorage_cuda(self, device, *args, **kwargs)

# Override various tensor creation functions to ensure device and type compatibility with XPU
original_torch_empty = torch.empty
@wraps(torch.empty)
def torch_empty(*args, device=None, **kwargs):
    """
    Returns a tensor filled with uninitialized data.
    """
    if check_device(device):
        return original_torch_empty(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_empty(*args, device=device, **kwargs)

original_torch_randn = torch.randn
@wraps(torch.randn)
def torch_randn(*args, device=None, dtype=None, **kwargs):
    """
    Returns a tensor filled with random numbers from a normal distribution.
    """
    if dtype == bytes:
        dtype = None
    if check_device(device):
        return original_torch_randn(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_randn(*args, device=device, **kwargs)

original_torch_ones = torch.ones
@wraps(torch.ones)
def torch_ones(*args, device=None, **kwargs):
    """
    Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size.
    """
    if check_device(device):
        return original_torch_ones(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_ones(*args, device=device, **kwargs)

original_torch_zeros = torch.zeros
@wraps(torch.zeros)
def torch_zeros(*args, device=None, **kwargs):
    """
    Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.
    """
    if check_device(device):
        return original_torch_zeros(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_zeros(*args, device=device, **kwargs)

original_torch_linspace = torch.linspace
@wraps(torch.linspace)
def torch_linspace(*args, device=None, **kwargs):
    """
    Returns a one-dimensional tensor of steps equally spaced points between start and end.
    """
    if check_device(device):
        return original_torch_linspace(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_linspace(*args, device=device, **kwargs)

# Override `torch.Generator` for compatibility with XPU
original_torch_Generator = torch.Generator
@wraps(torch.Generator)
def torch_Generator(device=None):
    """
    Creates a pseudo-random number generator object.
    """
    if check_device(device):
        return original_torch_Generator(return_xpu(device))
    else:
        return original_torch_Generator(device)

# Override `torch.load` to handle loading tensors onto XPU when specified
original_torch_load = torch.load
@wraps(torch.load)
def torch_load(f, map_location=None, *args, **kwargs):
    """
    Loads an object saved with `torch.save` from a file.
    """
    if check_device(map_location):
        return original_torch_load(f, *args, map_location=return_xpu(map_location), **kwargs)
    else:
        return original_torch_load(f, *args, map_location=map_location, **kwargs)

# Hijack Functions:
def ipex_hijacks():
    """
    Replaces specific PyTorch functions with their XPU-compatible versions.

    This function should be called to activate the compatibility layer for XPU devices.
    """
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

    torch.backends.cuda.sdp_kernel = return_null_context
    torch.UntypedStorage.is_cuda = is_cuda
    torch.cuda.is_available = is_available
    torch.amp.autocast_mode.autocast.__init__ = autocast_init

    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention
    torch.nn.functional.group_norm = functional_group_norm
    torch.nn.functional.layer_norm = functional_layer_norm
    torch.nn.functional.linear = functional_linear
    torch.nn.functional.conv2d = functional_conv2d
    torch.nn.functional.interpolate = interpolate
    torch.nn.functional.pad = functional_pad

    torch.bmm = torch_bmm
    torch.cat = torch_cat
    if not device_supports_fp64:
        torch.from_numpy = from_numpy
        torch.as_tensor = as_tensor