"""
Optimized Attention Operations for Intel XPUs

This module provides optimized implementations of `torch.bmm` and
`torch.nn.functional.scaled_dot_product_attention` for Intel XPUs. It employs a
slicing technique to handle large tensors, reducing memory usage and potentially
improving performance.

Important Notes:
- This module is for advanced users working with Intel XPUs. It may not
  have the intended effect on other hardware.
- Performance gains vary based on model and hardware configuration.
- Ensure the environment variables `IPEX_SDPA_SLICE_TRIGGER_RATE` and
  `IPEX_ATTENTION_SLICE_RATE` are set to positive values (in MB).

Main Components:

- `Settings`: Configures slicing thresholds.
- `torch_bmm_32_bit`: XPU-optimized `torch.bmm` with slicing.
- `scaled_dot_product_attention_32_bit`: XPU-optimized scaled dot-product
  attention with slicing.

Example Usage:
    >>> import torch
    >>> from attention import torch_bmm_32_bit, scaled_dot_product_attention_32_bit

    # Example with torch_bmm_32_bit
    >>> input_tensor = torch.rand(64, 128, 512, device='xpu')
    >>> mat2_tensor = torch.rand(64, 512, 128, device='xpu')
    >>> output = torch_bmm_32_bit(input_tensor, mat2_tensor)
    >>> print(output.shape)  # Output: torch.Size([64, 128, 128])

    # Example with scaled_dot_product_attention_32_bit
    >>> query = torch.rand(64, 128, 512, device='xpu')
    >>> key = torch.rand(64, 128, 512, device='xpu')
    >>> value = torch.rand(64, 128, 512, device='xpu')
    >>> output = scaled_dot_product_attention_32_bit(query, key, value)
    >>> print(output.shape)  # Output: torch.Size([64, 128, 512])
"""

import logging
import warnings
import torch
from functools import cache
from typing import Tuple, Optional
from pydantic import BaseSettings, Field, validator

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Define Settings class using Pydantic
class Settings(BaseSettings):
    """
    Configuration settings for attention slicing, loaded from environment variables.

    Attributes:
        sdpa_slice_trigger_rate (float): Threshold (MB) for slicing SDPA. Defaults to 6.
        attention_slice_rate (float): Threshold (MB) for slicing general attention. Defaults to 4.

    Raises:
        ValueError: If environment variables are not set to positive values.
    """
    sdpa_slice_trigger_rate: float = Field(default=6.0, env="IPEX_SDPA_SLICE_TRIGGER_RATE")
    attention_slice_rate: float = Field(default=4.0, env="IPEX_ATTENTION_SLICE_RATE")

    @validator('sdpa_slice_trigger_rate', 'attention_slice_rate')
    def validate_positive(cls, v: float) -> float:
        """Ensures slicing thresholds are positive."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

# Instantiate settings
settings = Settings()

def validate_tensor_shape(tensor: torch.Tensor, expected_dim: int) -> None:
    """Helper function to validate tensor dimensions."""
    if tensor.dim() != expected_dim:
        raise ValueError(f"Expected a {expected_dim}-dimensional tensor, but got a tensor with shape {tensor.shape}")

def check_xpu_available():
    """Checks if XPU is available."""
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        raise RuntimeError("XPU is not available. This module requires PyTorch with XPU support.")

# Check for XPU availability at the module level
check_xpu_available()

@cache
def find_slice_size(slice_size: int, block_size: float) -> int:
    """
    Calculates the optimal slice size based on memory constraints.

    Args:
        slice_size (int): Initial slice size.
        block_size (float): Memory size of a block (in MB).

    Returns:
        int: Adjusted slice size (minimum 1).
    """
    while (slice_size * block_size) > settings.attention_slice_rate:
        slice_size //= 2
        if slice_size <= 1:
            return 1
    return slice_size

@cache
def find_sdpa_slice_sizes(query_shape: Tuple[int, ...], query_element_size: int) -> Tuple[bool, bool, bool, int, int, int]:
    """
    Determines slice sizes for Scaled Dot-Product Attention (SDPA) operations.

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
    Determines slice sizes for `torch.bmm` operations.

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

    Overrides the standard `torch.bmm` for tensors on XPUs. Uses slicing to handle large 
    tensors, preventing memory overflow. If not on XPU, the original function is called.

    Args:
        input (torch.Tensor): First tensor for multiplication.
        mat2 (torch.Tensor): Second tensor.
        out (Optional[torch.Tensor], optional): Output tensor. 

    Returns:
        torch.Tensor: Result of the multiplication.

    Example:
        >>> output = torch_bmm_32_bit(input_tensor, matrix_tensor)
        >>> output.shape
        torch.Size([64, 128, 128])
    """
    validate_tensor_shape(input, 3)
    validate_tensor_shape(mat2, 3)

    if input.device.type != "xpu":
        warnings.warn("Input tensor is not on an XPU. Falling back to the original torch.bmm. "
                      "Performance may be affected.")
        return original_torch_bmm(input, mat2, out=out)

    do_split, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size = find_bmm_slice_sizes(input.shape, input.element_size(), mat2.shape)

    if do_split:
        logger.info("Slicing required for torch_bmm_32_bit. Starting sliced operation.")
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
                            # Slice tensors and perform bmm.
                            hidden_states[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3] = original_torch_bmm(
                                input[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3],
                                mat2[start_idx:end_idx, :, start_idx_3:end_idx_3],
                            ) 
                    else:
                        # Slice tensors and perform bmm.
                        hidden_states[start_idx:end_idx, start_idx_2:end_idx_2] = original_torch_bmm(
                            input[start_idx:end_idx, start_idx_2:end_idx_2],
                            mat2[start_idx:end_idx, :, :],
                        )
            else:
                # Slice tensors and perform bmm.
                hidden_states[start_idx:end_idx] = original_torch_bmm(
                    input[start_idx:end_idx],
                    mat2[start_idx:end_idx], 
                )
        if out is not None:
            out.copy_(hidden_states)
            return out
        torch.xpu.synchronize(input.device)
        return hidden_states
    # No slicing needed
    logger.info("No slicing required for torch_bmm_32_bit. Using original bmm.")
    return original_torch_bmm(input, mat2, out=out)

# Store the original scaled_dot_product_attention function
original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention

def scaled_dot_product_attention_32_bit(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, dropout_p: float = 0.0, is_causal: bool = False, **kwargs) -> torch.Tensor:
    """
    XPU-optimized scaled dot-product attention with 32-bit execution and support for slicing.

    Overrides the standard `scaled_dot_product_attention` to handle large tensors 
    efficiently on XPUs using slicing to reduce memory usage. If not on an XPU, 
    the original function is used.

    Args:
        query (torch.Tensor): Query tensor.
        key (torch.Tensor): Key tensor.
        value (torch.Tensor): Value tensor.
        attn_mask (Optional[torch.Tensor], optional): Attention mask. 
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.
        is_causal (bool, optional): Whether to apply causal masking. Defaults to False.
        **kwargs: Additional arguments for `scaled_dot_product_attention`.

    Returns:
        torch.Tensor: Result of the attention operation.
    """

    validate_tensor_shape(query, 3)
    validate_tensor_shape(key, 3)
    validate_tensor_shape(value, 3)

    if query.device.type != "xpu":
        warnings.warn("Input tensor is not on an XPU. Falling back to the original " 
                      "scaled_dot_product_attention. Performance may be affected.")
        return original_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, 
                                                      dropout_p=dropout_p, is_causal=is_causal, 
                                                      **kwargs)
    
    (do_split_query, do_split_2_query, do_split_3_query, split_slice_size_query, 
     split_2_slice_size_query, split_3_slice_size_query) = find_sdpa_slice_sizes(query.shape, query.element_size())

    (do_split_key, do_split_2_key, do_split_3_key, split_slice_size_key, 
     split_2_slice_size_key, split_3_slice_size_key) = find_sdpa_slice_sizes(key.shape, key.element_size())

    (do_split_value, do_split_2_value, do_split_3_value, split_slice_size_value,
     split_2_slice_size_value, split_3_slice_size_value) = find_sdpa_slice_sizes(value.shape, value.element_size())

    # Slice SDPA if needed
    if do_split_query or do_split_key or do_split_value:
        logger.info("Slicing required for scaled_dot_product_attention_32_bit. Starting sliced operation.")
        batch_size_attention, query_tokens, shape_three = query.shape[0], query.shape[1], query.shape[2]
        hidden_states = torch.zeros(query.shape, device=query.device, dtype=query.dtype)

        # Calculate slicing for attention mask if provided
        if attn_mask is not None:
            (do_split_mask, do_split_2_mask, _, split_slice_size_mask, 
             split_2_slice_size_mask, _) = find_sdpa_slice_sizes(attn_mask.shape, attn_mask.element_size())

        for i in range(batch_size_attention // split_slice_size_query):
            start_idx_query = i * split_slice_size_query
            end_idx_query = (i + 1) * split_slice_size_query

            start_idx_key = i * split_slice_size_key
            end_idx_key = (i + 1) * split_slice_size_key

            start_idx_value = i * split_slice_size_value
            end_idx_value = (i + 1) * split_slice_size_value

            if do_split_2_query or do_split_2_key or do_split_2_value:
                for j in range(query_tokens // split_2_slice_size_query):
                    start_idx_2_query = j * split_2_slice_size_query
                    end_idx_2_query = (j + 1) * split_2_slice_size_query

                    start_idx_2_key = j * split_2_slice_size_key
                    end_idx_2_key = (j + 1) * split_2_slice_size_key

                    start_idx_2_value = j * split_2_slice_size_value
                    end_idx_2_value = (j + 1) * split_2_slice_size_value

                    if do_split_3_query or do_split_3_key or do_split_3_value:
                        for k in range(shape_three // split_3_slice_size_query):
                            start_idx_3_query = k * split_3_slice_size_query
                            end_idx_3_query = (k + 1) * split_3_slice_size_query

                            start_idx_3_key = k * split_3_slice_size_key
                            end_idx_3_key = (k + 1) * split_3_slice_size_key

                            start_idx_3_value = k * split_3_slice_size_value
                            end_idx_3_value = (k + 1) * split_3_slice_size_value

                            # Slice attn_mask using its own slicing parameters
                            if attn_mask is not None:
                                start_idx_mask = i * split_slice_size_mask
                                end_idx_mask = (i + 1) * split_slice_size_mask
                                start_idx_2_mask = j * split_2_slice_size_mask
                                end_idx_2_mask = (j + 1) * split_2_slice_size_mask
                                sliced_attn_mask = attn_mask[start_idx_mask:end_idx_mask, start_idx_2_mask:end_idx_2_mask, :] 
                            else:
                                sliced_attn_mask = None

                            # Perform scaled dot-product attention on sliced tensors
                            hidden_states[start_idx_query:end_idx_query, start_idx_2_query:end_idx_2_query, start_idx_3_query:end_idx_3_query] = original_scaled_dot_product_attention(
                                query[start_idx_query:end_idx_query, start_idx_2_query:end_idx_2_query, start_idx_3_query:end_idx_3_query],
                                key[start_idx_key:end_idx_key, start_idx_2_key:end_idx_2_key, start_idx_3_key:end_idx_3_key], 
                                value[start_idx_value:end_idx_value, start_idx_2_value:end_idx_2_value, start_idx_3_value:end_idx_3_value],
                                attn_mask=sliced_attn_mask, # Use the sliced attention mask
                                dropout_p=dropout_p, is_causal=is_causal, **kwargs
                            )
                    else:
                        # Perform scaled dot-product attention on the first two dimensions
                        if attn_mask is not None:
                            start_idx_mask = i * split_slice_size_mask
                            end_idx_mask = (i + 1) * split_slice_size_mask
                            start_idx_2_mask = j * split_2_slice_size_mask
                            end_idx_2_mask = (j + 1) * split_2_slice_size_mask
                            sliced_attn_mask = attn_mask[start_idx_mask:end_idx_mask, start_idx_2_mask:end_idx_2_mask]
                        else:
                            sliced_attn_mask = None

                        hidden_states[start_idx_query:end_idx_query, start_idx_2_query:end_idx_2_query] = original_scaled_dot_product_attention(
                            query[start_idx_query:end_idx_query, start_idx_2_query:end_idx_2_query],
                            key[start_idx_key:end_idx_key, start_idx_2_key:end_idx_2_key], 
                            value[start_idx_value:end_idx_value, start_idx_2_value:end_idx_2_value],
                            attn_mask=sliced_attn_mask, # Use the sliced attention mask
                            dropout_p=dropout_p, is_causal=is_causal, **kwargs
                        )
            else:
                # Perform scaled dot-product attention on the first dimension
                if attn_mask is not None:
                    start_idx_mask = i * split_slice_size_mask
                    end_idx_mask = (i + 1) * split_slice_size_mask
                    sliced_attn_mask = attn_mask[start_idx_mask:end_idx_mask] 
                else:
                    sliced_attn_mask = None

                hidden_states[start_idx_query:end_idx_query] = original_scaled_dot_product_attention(
                    query[start_idx_query:end_idx_query],
                    key[start_idx_key:end_idx_key],
                    value[start_idx_value:end_idx_value],
                    attn_mask=sliced_attn_mask, # Use the sliced attention mask
                    dropout_p=dropout_p, is_causal=is_causal, **kwargs
                )
        if out is not None:
            out.copy_(hidden_states)
            return out
        torch.xpu.synchronize(query.device)  # Synchronize after sliced operations
        return hidden_states
    # No slicing needed
    logger.info("No slicing required for scaled_dot_product_attention_32_bit. Using original function.")
    return original_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)