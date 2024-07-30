import pytest
from PIL import Image
import numpy as np
from service.inpaint_utils import (
    get_image_ndarray,
    detect_mask_valid_edge,
    pre_input_and_mask,
    calc_out_size,
    make_multiple_of_8,
    resize_by_max,
    slice_image,
    UnsupportedFormat,
    MatteMatting
)

def test_get_image_ndarray_pil():
    img = Image.new("RGB", (100, 100))
    result = get_image_ndarray(img)
    assert isinstance(result, np.ndarray)
    assert result.shape == (100, 100, 3)

def test_get_image_ndarray_numpy():
    arr = np.ones((100, 100, 3), dtype=np.uint8)
    result = get_image_ndarray(arr)
    assert np.array_equal(result, arr)

def test_detect_mask_valid_edge():
    mask = Image.new("L", (100, 100), color=0)
    mask.paste(255, [10, 10, 90, 90])
    result = detect_mask_valid_edge(mask)
    assert result == (10, 10, 90, 90)  # Aligned with function's exclusive boundaries

def test_pre_input_and_mask_exact_crop():
    image = Image.new("RGB", (100, 100))
    mask = Image.new("L", (100, 100))
    mask.paste(255, [10, 10, 90, 90])
    img_crop, mask_crop, slice_box = pre_input_and_mask(image, mask)
    assert img_crop.size == (100, 100)
    assert mask_crop.size == (100, 100)
    assert slice_box == (0, 0, 100, 100)

def test_pre_input_and_mask_adjusted_crop():
    image = Image.new("RGB", (100, 100))
    mask = Image.new("L", (100, 100))
    mask.paste(255, [10, 10, 90, 90])
    img_crop, mask_crop, slice_box = pre_input_and_mask(image, mask, slice_increment=128)
    expected_size = (100, 100)  # Image dimensions constrain the slice size
    assert img_crop.size == expected_size
    assert mask_crop.size == expected_size
    assert slice_box == (0, 0, 100, 100)


def test_calc_out_size():
    width, height = 800, 600
    new_width, new_height, ratio = calc_out_size(width, height)
    assert new_width == 768
    assert new_height == 576
    assert ratio == 1.0416666666666667

def test_make_multiple_of_8():
    assert make_multiple_of_8(5) == 8
    assert make_multiple_of_8(10) == 16
    assert make_multiple_of_8(16) == 16

def test_resize_by_max():
    image = Image.new("RGB", (1000, 500))
    resized_img, ratio = resize_by_max(image, 768)
    assert resized_img.size == (768, 384)
    assert ratio == 1.3020833333333333

def test_slice_image():
    image = np.ones((300, 200, 3), dtype=np.uint8) * 255
    slices = slice_image(image)
    assert len(slices) == 6
    assert slices[0].shape == (100, 100, 3)

def test_unsupported_format_exception():
    with pytest.raises(UnsupportedFormat):
        raise UnsupportedFormat('invalid_type')