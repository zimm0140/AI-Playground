"""
PyTorch XPU Hijacks Module
--------------------------
This module contains function hijacks to redirect PyTorch CUDA operations to Intel XPU devices.
It allows code written for CUDA to work with Intel GPUs by replacing or modifying PyTorch functions.

Code credit: https://github.com/vladmandic/automatic/blob/master/modules/intel/ipex/hijacks.py
"""

import os
from functools import wraps
from contextlib import nullcontext
import torch
import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import
import numpy as np


# Check if the device supports 64-bit floating point operations
device_supports_fp64 = torch.xpu.has_fp64_dtype()

# pylint: disable=protected-access, missing-function-docstring, line-too-long, unnecessary-lambda, no-else-return

# Redirect torch.cuda to torch.xpu to make CUDA code work with Intel XPU
torch.cuda = torch.xpu


def return_null_context(*args, **kwargs):  # pylint: disable=unused-argument
    """
    Return a null context manager regardless of input arguments.
    This is used to replace functions that would normally return context managers.
    """
    return nullcontext()


@wraps(torch.cuda.is_available)
def is_available():
    """
    Hijacked version of torch.cuda.is_available that checks for XPU availability instead.
    
    Returns:
        bool: True if Intel XPU is available, False otherwise.
    """
    return ipex.has_xpu()


@property
def is_cuda(self):
    """
    Property that returns True if the device is XPU or CUDA.
    This allows code checking for CUDA devices to also recognize XPU devices.
    
    Returns:
        bool: True if device is XPU or CUDA, False otherwise.
    """
    return self.device.type == "xpu" or self.device.type == "cuda"


def check_device(device):
    """
    Check if a device is a CUDA device.
    
    Args:
        device: Device to check, can be a torch.device, string, or integer.
        
    Returns:
        bool: True if the device is a CUDA device, False otherwise.
    """
    return bool(
        (isinstance(device, torch.device) and device.type == "cuda")
        or (isinstance(device, str) and "cuda" in device)
        or isinstance(device, int)
    )


def return_xpu(device):
    """
    Convert a CUDA device specification to an XPU device specification.
    
    Args:
        device: A device specification (string, int, or torch.device).
        
    Returns:
        str or torch.device: The equivalent XPU device.
    """
    return (
        f"xpu:{device.split(':')[-1]}"
        if isinstance(device, str) and ":" in device
        else f"xpu:{device}"
        if isinstance(device, int)
        else torch.device("xpu")
        if isinstance(device, torch.device)
        else "xpu"
    )


# Store the original autocast initialization
original_autocast_init = torch.amp.autocast_mode.autocast.__init__


@wraps(torch.amp.autocast_mode.autocast.__init__)
def autocast_init(self, device_type, dtype=None, enabled=True, cache_enabled=None):
    """
    Hijacked version of torch.amp.autocast_mode.autocast.__init__ that uses bfloat16 for XPU/CUDA.
    
    Args:
        device_type: The device type, e.g., 'cuda', 'xpu', 'cpu'.
        dtype: The data type to use for the autocast. Defaults to bfloat16 for XPU/CUDA.
        enabled: Whether autocast is enabled.
        cache_enabled: Whether caching is enabled.
    """
    if device_type == "cuda" or device_type == "xpu":
        if dtype is None:
            dtype = torch.bfloat16
        return original_autocast_init(
            self,
            device_type="xpu",
            dtype=dtype,
            enabled=enabled,
            cache_enabled=cache_enabled,
        )
    else:
        return original_autocast_init(
            self,
            device_type=device_type,
            dtype=dtype,
            enabled=enabled,
            cache_enabled=cache_enabled,
        )


# Store the original interpolate function
original_interpolate = torch.nn.functional.interpolate


@wraps(torch.nn.functional.interpolate)
def interpolate(
    tensor,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
    antialias=False,
):  # pylint: disable=too-many-arguments
    """
    Hijacked version of torch.nn.functional.interpolate that offloads to CPU for certain operations.
    
    For operations that may not be well-supported on XPU (antialias, align_corners, bicubic mode),
    this function temporarily moves the tensor to CPU, performs the operation, and moves it back.
    
    Args:
        tensor: Input tensor.
        size: Output size.
        scale_factor: Scale factor.
        mode: Interpolation mode.
        align_corners: Whether to align corners.
        recompute_scale_factor: Whether to recompute scale factor.
        antialias: Whether to use antialiasing.
        
    Returns:
        torch.Tensor: The interpolated tensor.
    """
    if antialias or align_corners is not None or mode == "bicubic":
        return_device = tensor.device
        return_dtype = tensor.dtype
        return original_interpolate(
            tensor.to("cpu", dtype=torch.float32),
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        ).to(return_device, dtype=return_dtype)
    else:
        return original_interpolate(
            tensor,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )


# Store the original from_numpy function
original_from_numpy = torch.from_numpy


@wraps(torch.from_numpy)
def from_numpy(ndarray):
    """
    Hijacked version of torch.from_numpy that converts float64 NumPy arrays to float32.
    
    This is needed because Alchemist GPUs don't support 64-bit operations.
    
    Args:
        ndarray: NumPy array to convert to a torch tensor.
        
    Returns:
        torch.Tensor: The converted tensor.
    """
    if ndarray.dtype == float:
        return original_from_numpy(ndarray.astype("float32"))
    else:
        return original_from_numpy(ndarray)


# Store the original as_tensor function
original_as_tensor = torch.as_tensor


@wraps(torch.as_tensor)
def as_tensor(data, dtype=None, device=None):
    """
    Hijacked version of torch.as_tensor that handles float64 conversion for XPU devices.
    
    Args:
        data: Data to convert to a tensor.
        dtype: Data type of the returned tensor.
        device: Device to place the tensor on.
        
    Returns:
        torch.Tensor: The converted tensor.
    """
    if check_device(device):
        device = return_xpu(device)
    if (
        isinstance(data, np.ndarray)
        and data.dtype == float
        and not (
            (isinstance(device, torch.device) and device.type == "cpu")
            or (isinstance(device, str) and "cpu" in device)
        )
    ):
        return original_as_tensor(data, dtype=torch.float32, device=device)
    else:
        return original_as_tensor(data, dtype=dtype, device=device)


# Handle 32-bit attention workarounds for devices that don't support float64
if device_supports_fp64 and os.environ.get("IPEX_FORCE_ATTENTION_SLICE", None) is None:
    original_torch_bmm = torch.bmm
    original_scaled_dot_product_attention = (
        torch.nn.functional.scaled_dot_product_attention
    )
else:
    # 32 bit attention workarounds for Alchemist:
    try:
        from attention import torch_bmm_32_bit as original_torch_bmm
        from attention import (
            scaled_dot_product_attention_32_bit as original_scaled_dot_product_attention,
        )
    except Exception:  # pylint: disable=broad-exception-caught
        original_torch_bmm = torch.bmm
        original_scaled_dot_product_attention = (
            torch.nn.functional.scaled_dot_product_attention
        )


@wraps(torch.bmm)
def torch_bmm(input, mat2, *, out=None):
    """
    Hijacked version of torch.bmm that handles data type mismatches.
    
    Ensures that both input matrices have the same data type before matrix multiplication.
    
    Args:
        input: First batch of matrices.
        mat2: Second batch of matrices.
        out: Output tensor.
        
    Returns:
        torch.Tensor: The result of batch matrix multiplication.
    """
    if input.dtype != mat2.dtype:
        mat2 = mat2.to(input.dtype)
    return original_torch_bmm(input, mat2, out=out)


@wraps(torch.nn.functional.scaled_dot_product_attention)
def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
):
    """
    Hijacked version of torch.nn.functional.scaled_dot_product_attention that handles data type mismatches.
    
    Ensures that query, key, value, and attention mask all have the same data type.
    
    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        attn_mask: Attention mask tensor.
        dropout_p: Dropout probability.
        is_causal: Whether to use causal attention.
        
    Returns:
        torch.Tensor: The result of scaled dot-product attention.
    """
    if query.dtype != key.dtype:
        key = key.to(dtype=query.dtype)
    if query.dtype != value.dtype:
        value = value.to(dtype=query.dtype)
    if attn_mask is not None and query.dtype != attn_mask.dtype:
        attn_mask = attn_mask.to(dtype=query.dtype)
    return original_scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
    )


# Store the original group_norm function
original_functional_group_norm = torch.nn.functional.group_norm


@wraps(torch.nn.functional.group_norm)
def functional_group_norm(input, num_groups, weight=None, bias=None, eps=1e-05):
    """
    Hijacked version of torch.nn.functional.group_norm that handles data type mismatches.
    
    Ensures that input and weight/bias have the same data type.
    
    Args:
        input: Input tensor.
        num_groups: Number of groups.
        weight: Weight tensor.
        bias: Bias tensor.
        eps: Epsilon value for numerical stability.
        
    Returns:
        torch.Tensor: The result of group normalization.
    """
    if weight is not None and input.dtype != weight.data.dtype:
        input = input.to(dtype=weight.data.dtype)
    if bias is not None and weight is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_group_norm(
        input, num_groups, weight=weight, bias=bias, eps=eps
    )


# Store the original layer_norm function
original_functional_layer_norm = torch.nn.functional.layer_norm


@wraps(torch.nn.functional.layer_norm)
def functional_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    """
    Hijacked version of torch.nn.functional.layer_norm that handles data type mismatches.
    
    Ensures that input and weight/bias have the same data type.
    
    Args:
        input: Input tensor.
        normalized_shape: Shape of the input tensor.
        weight: Weight tensor.
        bias: Bias tensor.
        eps: Epsilon value for numerical stability.
        
    Returns:
        torch.Tensor: The result of layer normalization.
    """
    if weight is not None and input.dtype != weight.data.dtype:
        input = input.to(dtype=weight.data.dtype)
    if bias is not None and weight is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_layer_norm(
        input, normalized_shape, weight=weight, bias=bias, eps=eps
    )


# Store the original linear function
original_functional_linear = torch.nn.functional.linear


@wraps(torch.nn.functional.linear)
def functional_linear(input, weight, bias=None):
    """
    Hijacked version of torch.nn.functional.linear that handles data type mismatches.
    
    Ensures that input and weight/bias have the same data type.
    
    Args:
        input: Input tensor.
        weight: Weight tensor.
        bias: Bias tensor.
        
    Returns:
        torch.Tensor: The result of the linear transformation.
    """
    if input.dtype != weight.data.dtype:
        input = input.to(dtype=weight.data.dtype)
    if bias is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_linear(input, weight, bias=bias)


# Store the original conv2d function
original_functional_conv2d = torch.nn.functional.conv2d


@wraps(torch.nn.functional.conv2d)
def functional_conv2d(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
):
    """
    Hijacked version of torch.nn.functional.conv2d that handles data type mismatches.
    
    Ensures that input and weight/bias have the same data type.
    
    Args:
        input: Input tensor.
        weight: Weight tensor.
        bias: Bias tensor.
        stride: Stride of the convolution.
        padding: Padding added to all sides of the input.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections from input channels to output channels.
        
    Returns:
        torch.Tensor: The result of the 2D convolution.
    """
    if input.dtype != weight.data.dtype:
        input = input.to(dtype=weight.data.dtype)
    if bias is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_conv2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


# Store the original cat function
original_torch_cat = torch.cat


@wraps(torch.cat)
def torch_cat(tensor, *args, **kwargs):
    """
    Hijacked version of torch.cat that handles data type mismatches in tensor concatenation.
    
    Specifically handles the case of three tensors with mismatched data types.
    
    Args:
        tensor: Sequence of tensors to concatenate.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.
        
    Returns:
        torch.Tensor: The concatenated tensor.
    """
    if len(tensor) == 3 and (
        tensor[0].dtype != tensor[1].dtype or tensor[2].dtype != tensor[1].dtype
    ):
        return original_torch_cat(
            [tensor[0].to(tensor[1].dtype), tensor[1], tensor[2].to(tensor[1].dtype)],
            *args,
            **kwargs,
        )
    else:
        return original_torch_cat(tensor, *args, **kwargs)


# Store the original pad function
original_functional_pad = torch.nn.functional.pad


@wraps(torch.nn.functional.pad)
def functional_pad(input, pad, mode="constant", value=None):
    """
    Hijacked version of torch.nn.functional.pad for SwinIR BF16 compatibility.
    
    For 'reflect' mode with bfloat16 input, temporarily converts to float32, pads, then converts back.
    
    Args:
        input: Input tensor.
        pad: Padding size.
        mode: Padding mode.
        value: Fill value for 'constant' padding.
        
    Returns:
        torch.Tensor: The padded tensor.
    """
    if mode == "reflect" and input.dtype == torch.bfloat16:
        return original_functional_pad(
            input.to(torch.float32), pad, mode=mode, value=value
        ).to(dtype=torch.bfloat16)
    else:
        return original_functional_pad(input, pad, mode=mode, value=value)


# Store the original tensor function
original_torch_tensor = torch.tensor


@wraps(torch.tensor)
def torch_tensor(data, *args, dtype=None, device=None, **kwargs):
    """
    Hijacked version of torch.tensor that handles CUDA to XPU device conversion and float64 to float32 conversion.
    
    Args:
        data: Data to create a tensor from.
        *args: Additional arguments.
        dtype: Data type of the tensor.
        device: Device to place the tensor on.
        **kwargs: Additional keyword arguments.
        
    Returns:
        torch.Tensor: The created tensor.
    """
    if check_device(device):
        device = return_xpu(device)
    if not device_supports_fp64:
        if (isinstance(device, torch.device) and device.type == "xpu") or (
            isinstance(device, str) and "xpu" in device
        ):
            if dtype == torch.float64:
                dtype = torch.float32
            elif dtype is None and (
                hasattr(data, "dtype")
                and (data.dtype == torch.float64 or data.dtype == float)
            ):
                dtype = torch.float32
    return original_torch_tensor(data, *args, dtype=dtype, device=device, **kwargs)


# Store the original Tensor.to method
original_Tensor_to = torch.Tensor.to


@wraps(torch.Tensor.to)
def Tensor_to(self, device=None, *args, **kwargs):
    """
    Hijacked version of torch.Tensor.to that converts CUDA device specifications to XPU.
    
    Args:
        self: The tensor to move.
        device: Target device.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.
        
    Returns:
        torch.Tensor: The tensor on the target device.
    """
    if check_device(device):
        return original_Tensor_to(self, return_xpu(device), *args, **kwargs)
    else:
        return original_Tensor_to(self, device, *args, **kwargs)


# Store the original Tensor.cuda method
original_Tensor_cuda = torch.Tensor.cuda


@wraps(torch.Tensor.cuda)
def Tensor_cuda(self, device=None, *args, **kwargs):
    """
    Hijacked version of torch.Tensor.cuda that converts to XPU instead of CUDA.
    
    Args:
        self: The tensor to move.
        device: Target device.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.
        
    Returns:
        torch.Tensor: The tensor on the XPU device.
    """
    if check_device(device):
        return original_Tensor_cuda(self, return_xpu(device), *args, **kwargs)
    else:
        return original_Tensor_cuda(self, device, *args, **kwargs)


# Store the original UntypedStorage.__init__ method
original_UntypedStorage_init = torch.UntypedStorage.__init__


@wraps(torch.UntypedStorage.__init__)
def UntypedStorage_init(*args, device=None, **kwargs):
    """
    Hijacked version of torch.UntypedStorage.__init__ that converts CUDA device specifications to XPU.
    
    Args:
        *args: Arguments.
        device: Target device.
        **kwargs: Keyword arguments.
        
    Returns:
        The result of the original UntypedStorage.__init__.
    """
    if check_device(device):
        return original_UntypedStorage_init(*args, device=return_xpu(device), **kwargs)
    else:
        return original_UntypedStorage_init(*args, device=device, **kwargs)


# Store the original UntypedStorage.cuda method
original_UntypedStorage_cuda = torch.UntypedStorage.cuda


@wraps(torch.UntypedStorage.cuda)
def UntypedStorage_cuda(self, device=None, *args, **kwargs):
    """
    Hijacked version of torch.UntypedStorage.cuda that converts to XPU instead of CUDA.
    
    Args:
        self: The storage to move.
        device: Target device.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.
        
    Returns:
        torch.UntypedStorage: The storage on the XPU device.
    """
    if check_device(device):
        return original_UntypedStorage_cuda(self, return_xpu(device), *args, **kwargs)
    else:
        return original_UntypedStorage_cuda(self, device, *args, **kwargs)


# Store the original empty function
original_torch_empty = torch.empty


@wraps(torch.empty)
def torch_empty(*args, device=None, **kwargs):
    """
    Hijacked version of torch.empty that converts CUDA device specifications to XPU.
    
    Args:
        *args: Arguments for the empty tensor.
        device: Target device.
        **kwargs: Additional keyword arguments.
        
    Returns:
        torch.Tensor: An uninitialized tensor on the target device.
    """
    if check_device(device):
        return original_torch_empty(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_empty(*args, device=device, **kwargs)


# Store the original randn function
original_torch_randn = torch.randn


@wraps(torch.randn)
def torch_randn(*args, device=None, dtype=None, **kwargs):
    """
    Hijacked version of torch.randn that converts CUDA device specifications to XPU.
    
    Args:
        *args: Arguments for the random tensor.
        device: Target device.
        dtype: Data type of the tensor.
        **kwargs: Additional keyword arguments.
        
    Returns:
        torch.Tensor: A tensor of random numbers from a normal distribution.
    """
    if dtype == bytes:  # noqa: E721
        dtype = None
    if check_device(device):
        return original_torch_randn(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_randn(*args, device=device, **kwargs)


# Store the original ones function
original_torch_ones = torch.ones


@wraps(torch.ones)
def torch_ones(*args, device=None, **kwargs):
    """
    Hijacked version of torch.ones that converts CUDA device specifications to XPU.
    
    Args:
        *args: Arguments for the tensor.
        device: Target device.
        **kwargs: Additional keyword arguments.
        
    Returns:
        torch.Tensor: A tensor filled with ones.
    """
    if check_device(device):
        return original_torch_ones(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_ones(*args, device=device, **kwargs)


# Store the original zeros function
original_torch_zeros = torch.zeros


@wraps(torch.zeros)
def torch_zeros(*args, device=None, **kwargs):
    """
    Hijacked version of torch.zeros that converts CUDA device specifications to XPU.
    
    Args:
        *args: Arguments for the tensor.
        device: Target device.
        **kwargs: Additional keyword arguments.
        
    Returns:
        torch.Tensor: A tensor filled with zeros.
    """
    if check_device(device):
        return original_torch_zeros(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_zeros(*args, device=device, **kwargs)


# Store the original linspace function
original_torch_linspace = torch.linspace


@wraps(torch.linspace)
def torch_linspace(*args, device=None, **kwargs):
    """
    Hijacked version of torch.linspace that converts CUDA device specifications to XPU.
    
    Args:
        *args: Arguments for the tensor.
        device: Target device.
        **kwargs: Additional keyword arguments.
        
    Returns:
        torch.Tensor: A tensor of linearly spaced values.
    """
    if check_device(device):
        return original_torch_linspace(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_linspace(*args, device=device, **kwargs)


# Store the original Generator function
original_torch_Generator = torch.Generator


@wraps(torch.Generator)
def torch_Generator(device=None):
    """
    Hijacked version of torch.Generator that converts CUDA device specifications to XPU.
    
    Args:
        device: Target device.
        
    Returns:
        torch.Generator: A random number generator on the target device.
    """
    if check_device(device):
        return original_torch_Generator(return_xpu(device))
    else:
        return original_torch_Generator(device)


# Store the original load function
original_torch_load = torch.load


@wraps(torch.load)
def torch_load(f, map_location=None, *args, **kwargs):
    """
    Hijacked version of torch.load that converts CUDA device specifications to XPU.
    
    Args:
        f: File-like object or string containing a file name.
        map_location: Location to which the storage is mapped.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.
        
    Returns:
        The object loaded from the file.
    """
    if check_device(map_location):
        return original_torch_load(
            f, *args, map_location=return_xpu(map_location), **kwargs
        )
    else:
        return original_torch_load(f, *args, map_location=map_location, **kwargs)


def ipex_hijacks():
    """
    Apply all the XPU hijacks to the torch module.
    
    This function replaces various PyTorch functions with the hijacked versions
    defined in this module to make code written for CUDA work with Intel XPU devices.
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
