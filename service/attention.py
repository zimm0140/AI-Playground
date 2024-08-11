"""
This module provides optimized attention operations for deep learning models on Intel XPUs. 

It specifically addresses potential memory limitations by dynamically slicing large 
attention operations (like `torch.bmm` and scaled dot-product attention) into smaller, 
more manageable chunks. This enables the processing of larger models or inputs on 
memory-constrained XPU devices.

Note: This module is designed for advanced users working with Intel XPUs and may 
not have the intended effect on other hardware. Performance improvements may vary 
depending on the model and hardware configuration.

Example Usage:
    >>> import torch
    >>> from attention import torch_bmm_32_bit, scaled_dot_product_attention_32_bit

    >>> # Example with torch_bmm_32_bit
    >>> input_tensor = torch.rand(64, 128, 512, device='xpu')
    >>> mat2_tensor = torch.rand(64, 512, 128, device='xpu')
    >>> output = torch_bmm_32_bit(input_tensor, mat2_tensor) 
    >>> print(output.shape)  # Output: torch.Size([64, 128, 128])

    >>> # Example with scaled_dot_product_attention_32_bit
    >>> query = torch.rand(64, 128, 512, device='xpu')
    >>> key = torch.rand(64, 128, 512, device='xpu')
    >>> value = torch.rand(64, 128, 512, device='xpu')
    >>> output = scaled_dot_product_attention_32_bit(query, key, value)
    >>> print(output.shape)  # Output: torch.Size([64, 128, 512])
"""

import torch
from functools import cache
from pydantic import BaseSettings, Field, validator
from dotenv import load_dotenv
from typing import Tuple, Optional

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """
    Configuration settings for attention slicing, loaded from environment variables. 

    Attributes:
        sdpa_slice_trigger_rate (float): Threshold (in MB) that triggers slicing for 
            Scaled Dot-Product Attention (SDPA). Defaults to 6 MB.
        attention_slice_rate (float): Threshold (in MB) that triggers slicing for general 
            attention operations (including `torch.bmm`). Defaults to 4 MB.

    Example:
        >>> settings = Settings()
        >>> print(settings.sdpa_slice_trigger_rate)
        6.0
    """
    sdpa_slice_trigger_rate: float = Field(default=6, env="IPEX_SDPA_SLICE_TRIGGER_RATE")
    attention_slice_rate: float = Field(default=4, env="IPEX_ATTENTION_SLICE_RATE")

    @validator('sdpa_slice_trigger_rate', 'attention_slice_rate')
    def validate_positive(cls, v: float) -> float:
        """
        Ensures slicing thresholds are positive values.

        Args:
            v (float): The value to validate.

        Returns:
            float: The validated positive value.

        Raises:
            ValueError: If the provided value is not positive.
        """
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

# Instantiate the settings from environment variables.
settings = Settings()

@cache
def find_slice_size(slice_size: int, slice_block_size: float) -> int:
    """
    Determines a suitable slice size based on memory constraints.

    Reduces `slice_size` iteratively until it fits within the `attention_slice_rate` limit.

    Args:
        slice_size (int): Initial slice size.
        slice_block_size (float): Memory size of a block (in MB).

    Returns:
        int: The adjusted slice size (minimum 1). 
    """
    while (slice_size * slice_block_size) > settings.attention_slice_rate:
        slice_size //= 2
        if slice_size <= 1:
            slice_size = 1
            break
    return slice_size

@cache
def find_sdpa_slice_sizes(query_shape: Tuple[int, ...], query_element_size: int) -> Tuple[bool, bool, bool, int, int, int]:
    """
    Determines slice sizes for Scaled Dot-Product Attention (SDPA) operations.

    Calculates if slicing is needed for SDPA and determines slice sizes for each dimension to 
    ensure it fits within memory limits.

    Args:
        query_shape (Tuple[int, ...]): Shape of the query tensor (3D or 4D).
        query_element_size (int): Size of each element in bytes.

    Returns:
        Tuple[bool, bool, bool, int, int, int]: 
            - Whether to slice along each dimension (batch, token, third).
            - Slice size for each dimension.
    """
    if len(query_shape) == 3:
        batch_size_attention, query_tokens, shape_three = query_shape
        shape_four = 1
    else:
        batch_size_attention, query_tokens, shape_three, shape_four = query_shape

    slice_block_size = query_tokens * shape_three * shape_four / 1024 / 1024 * query_element_size
    block_size = batch_size_attention * slice_block_size

    split_slice_size = batch_size_attention
    split_2_slice_size = query_tokens
    split_3_slice_size = shape_three

    do_split = False
    do_split_2 = False
    do_split_3 = False

    if block_size > settings.sdpa_slice_trigger_rate:
        do_split = True
        split_slice_size = find_slice_size(split_slice_size, slice_block_size)
        if split_slice_size * slice_block_size > settings.attention_slice_rate:
            slice_2_block_size = split_slice_size * shape_three * shape_four / 1024 / 1024 * query_element_size
            do_split_2 = True
            split_2_slice_size = find_slice_size(split_2_slice_size, slice_2_block_size)
            if split_2_slice_size * slice_2_block_size > settings.attention_slice_rate:
                slice_3_block_size = split_slice_size * split_2_slice_size * shape_four / 1024 / 1024 * query_element_size
                do_split_3 = True
                split_3_slice_size = find_slice_size(split_3_slice_size, slice_3_block_size)

    return do_split, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size

@cache
def find_bmm_slice_sizes(input_shape: Tuple[int, int, int], input_element_size: int, mat2_shape: Tuple[int, int, int]) -> Tuple[bool, bool, bool, int, int, int]:
    """
    Determines slice sizes for `torch.bmm` operations based on input tensor shapes.

    Analogous to `find_sdpa_slice_sizes`, but for `torch.bmm`.

    Args:
        input_shape (Tuple[int, int, int]): Shape of the first input tensor.
        input_element_size (int): Size of each element in bytes.
        mat2_shape (Tuple[int, int, int]): Shape of the second matrix.

    Returns:
        Tuple[bool, bool, bool, int, int, int]: 
            - Whether to slice along each dimension (batch, token, third).
            - Slice size for each dimension. 
    """
    batch_size_attention, input_tokens, mat2_atten_shape = input_shape[0], input_shape[1], mat2_shape[2]
    slice_block_size = input_tokens * mat2_atten_shape / 1024 / 1024 * input_element_size
    block_size = batch_size_attention * slice_block_size

    split_slice_size = batch_size_attention
    split_2_slice_size = input_tokens
    split_3_slice_size = mat2_atten_shape

    do_split = False
    do_split_2 = False
    do_split_3 = False

    if block_size > settings.attention_slice_rate:
        do_split = True
        split_slice_size = find_slice_size(split_slice_size, slice_block_size)
        if split_slice_size * slice_block_size > settings.attention_slice_rate:
            slice_2_block_size = split_slice_size * mat2_atten_shape / 1024 / 1024 * input_element_size
            do_split_2 = True
            split_2_slice_size = find_slice_size(split_2_slice_size, slice_2_block_size)
            if split_2_slice_size * slice_2_block_size > settings.attention_slice_rate:
                slice_3_block_size = split_slice_size * split_2_slice_size / 1024 / 1024 * input_element_size
                do_split_3 = True
                split_3_slice_size = find_slice_size(split_3_slice_size, slice_3_block_size)

    return do_split, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size

# Store the original torch.bmm function
original_torch_bmm = torch.bmm

def torch_bmm_32_bit(input: torch.Tensor, mat2: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    XPU-optimized `torch.bmm` function with 32-bit execution and support for slicing.

    This function overrides the standard `torch.bmm` function for tensors on XPUs. It uses a 
    slicing mechanism to handle large tensors that might exceed memory limits. If the input 
    tensor is not on an XPU or slicing is not needed, the original `torch.bmm` is called. 

    Args:
        input (torch.Tensor): First tensor for batch matrix multiplication.
        mat2 (torch.Tensor): Second tensor for multiplication.
        out (Optional[torch.Tensor], optional): Output tensor, if specified.

    Returns:
        torch.Tensor: The result of the batched matrix multiplication.

    Example:
        >>> output = torch_bmm_32_bit(input_tensor, matrix_tensor)
        >>> output.shape
        torch.Size([64, 128, 128])
    """
    if input.device.type != "xpu":
        return original_torch_bmm(input, mat2, out=out)
    
    do_split, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size = find_bmm_slice_sizes(input.shape, input.element_size(), mat2.shape)

    if do_split:
        batch_size_attention, input_tokens, mat2_atten_shape = input.shape[0], input.shape[1], mat2.shape[2]
        hidden_states = torch.zeros(input.shape[0], input.shape[1], mat2.shape[2], device=input.device, dtype=input.dtype)
        for i in range(batch_size_attention // split_slice_size):
            start_idx = i * split_slice_size
            end_idx = (i + 1) * split_slice_size
            if do_split_2:
                for j in range(input_tokens // split_2_slice_size):
                    start_idx_2 = j * split_2_slice_size
                    end_idx_2 = (j + 1) * split_2_slice_size
                    if do_split_3:
                        for k in range(mat2_atten_shape // split_3_slice_size):
                            start_idx_3 = k * split_3_slice_size
                            end_idx_3 = (k + 1) * split_3_slice_size
                            hidden_states[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3] = original_torch_bmm(
                                input[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3],
                                mat2[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3],
                                out=out
                            )
                    else:
                        hidden_states[start_idx:end_idx, start_idx_2:end_idx_2] = original_torch_bmm(
                            input[start_idx:end_idx, start_idx_2:end.idx_2],
                            mat2[start_idx:end.idx, start_idx_2:end.idx_2],
                            out=out
                        )
            else:
                hidden_states[start_idx:end_idx] = original_torch_bmm(
                    input[start_idx:end.idx],
                    mat2[start.idx:end.idx],
                    out=out
                )
        torch.xpu.synchronize(input.device)
    else:
        return original_torch_bmm(input, mat2, out=out)
    return hidden_states

# Store the original scaled_dot_product_attention function
original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention

def scaled_dot_product_attention_32_bit(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, dropout_p: float = 0.0, is_causal: bool = False, **kwargs) -> torch.Tensor:
    """
    XPU-optimized scaled dot-product attention function for 32-bit execution, 
    with support for slicing.

    This function overrides the standard `scaled_dot_product_attention` when 
    operating on XPUs. It implements a slicing mechanism for large tensors to 
    prevent memory overflow. If tensors are not on an XPU or slicing is 
    not needed, the original function is used.

    Args:
        query (torch.Tensor): Query tensor.
        key (torch.Tensor): Key tensor.
        value (torch.Tensor): Value tensor.
        attn_mask (Optional[torch.Tensor], optional): Attention mask.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.
        is_causal (bool, optional): Whether to apply causal masking. Defaults to False.
        **kwargs: Additional arguments for `scaled_dot_product_attention`.

    Returns:
        torch.Tensor: Result of the scaled dot-product attention. 
    """
    if query.device.type != "xpu":
        return original_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)
    
    do_split, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size = find_sdpa_slice_sizes(query.shape, query.element_size())

    # Slice SDPA
    if do_split:
        batch_size_attention, query_tokens, shape_three = query.shape[0], query.shape[1], query.shape[2]
        hidden_states = torch.zeros(query.shape, device=query.device, dtype=query.dtype)
        for i in range(batch_size_attention // split_slice_size):
            start_idx = i * split_slice_size
            end_idx = (i + 1) * split_slice_size
            if do_split_2:
                for j in range(query_tokens // split_2_slice_size):
                    start_idx_2 = j * split_2_slice_size
                    end_idx_2 = (j + 1) * split_2_slice_size
                    if do_split_3:
                        for k in range(shape_three // split_3_slice_size):
                            start_idx_3 = k * split_3_slice_size
                            end_idx_3 = (k + 1) * split_3_slice_size
                            hidden_states[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3] = original_scaled_dot_product_attention(
                                query[start_idx:end.idx, start_idx_2:end.idx_2, start_idx_3:end.idx_3],
                                key[start.idx:end.idx, start.idx_2:end.idx_2, start.idx_3:end.idx_3],
                                value[start.idx:end.idx, start.idx_2:end.idx_2, start.idx_3:end.idx_3],
                                attn_mask=attn_mask[start.idx:end.idx, start.idx_2:end.idx_2, start.idx_3:end.idx_3] if attn_mask is not None else attn_mask,
                                dropout_p=dropout_p, is_causal=is_causal, **kwargs
                            )
                    else:
                        hidden_states[start_idx:end_idx, start_idx_2:end.idx_2] = original_scaled_dot_product_attention(
                            query[start.idx:end.idx, start.idx_2:end.idx_2],
                            key[start.idx:end.idx, start.idx_2:end.idx_2],
                            value[start.idx:end.idx, start.idx_2:end.idx_2],
                            attn_mask=attn_mask[start.idx:end.idx, start.idx_2:end.idx_2] if attn_mask is not None else attn_mask,
                            dropout_p=dropout_p, is_causal=is_causal, **kwargs
                        )
            else:
                hidden_states[start.idx:end.idx] = original_scaled_dot_product_attention(
                    query[start.idx:end.idx],
                    key[start.idx:end.idx],
                    value[start.idx:end.idx],
                    attn_mask=attn_mask[start.idx:end.idx] if attn_mask is not None else attn_mask,
                    dropout_p=dropout_p, is_causal=is_causal, **kwargs
                )
        torch.xpu.synchronize(query.device)
    else:
        return original_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)
    return hidden_states