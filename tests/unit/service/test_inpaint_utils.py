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
    MatteMatting,
)

def test_get_image_ndarray_pil():
    """
    Test conversion of a PIL Image to a numpy array.

    This function creates a new RGB PIL Image, converts it to a numpy array using 
    `get_image_ndarray`, and checks if the result is a numpy array with the expected shape.
    """
    img = Image.new("RGB", (100, 100))
    result = get_image_ndarray(img)
    assert isinstance(result, np.ndarray)
    assert result.shape == (100, 100, 3)

def test_get_image_ndarray_numpy():
    """
    Test that a numpy array input returns the same numpy array.

    This function creates a numpy array, passes it to `get_image_ndarray`, and checks 
    if the returned array is identical to the original.
    """
    arr = np.ones((100, 100, 3), dtype=np.uint8)
    result = get_image_ndarray(arr)
    assert np.array_equal(result, arr)

def test_get_image_ndarray_invalid_type():
    """
    Test that an invalid input type raises a TypeError.

    This function passes an invalid type (string) to `get_image_ndarray` and verifies 
    that it raises a TypeError.
    """
    with pytest.raises(TypeError):
        get_image_ndarray("invalid_type")

def test_detect_mask_valid_edge():
    """
    Test detection of the bounding box in a mask image.

    This function creates a mask image with a specific bounding box, passes it to 
    `detect_mask_valid_edge`, and checks if the returned bounding box is correct.
    """
    mask = Image.new("L", (100, 100), color=0)
    mask.paste(255, [10, 10, 91, 91])  # Use exclusive bound for the mask
    result = detect_mask_valid_edge(mask)
    assert result == (10, 10, 91, 91)  # Adjusted for exclusive bounds

def test_detect_mask_valid_edge_empty_mask():
    """
    Test that an empty mask raises a ValueError.

    This function creates an empty mask image, passes it to `detect_mask_valid_edge`, 
    and checks if it raises a ValueError.
    """
    mask = Image.new("L", (100, 100), color=0)
    with pytest.raises(ValueError):
        detect_mask_valid_edge(mask)

def test_pre_input_and_mask_exact_crop():
    """
    Test exact cropping of the image and mask without forcing multiples of slice_increment.

    This function creates an image and a mask with a specific bounding box, passes 
    them to `pre_input_and_mask` with `force_multiple_of_slice_increment` set to False, 
    and checks if the returned cropped image and mask sizes and slice box are correct.
    """
    image = Image.new("RGB", (100, 100))
    mask = Image.new("L", (100, 100))
    mask.paste(255, [10, 10, 90, 90])
    img_crop, mask_crop, slice_box = pre_input_and_mask(image, mask, force_multiple_of_slice_increment=False)
    assert img_crop.size == (80, 80)
    assert mask_crop.size == (80, 80)
    assert slice_box == (10, 10, 90, 90)

def test_pre_input_and_mask_adjusted_crop():
    """
    Test cropping of the image and mask with slice_increment set to 128.

    This function creates an image and a mask with a specific bounding box, passes 
    them to `pre_input_and_mask` with `slice_increment` set to 128 and 
    `force_multiple_of_slice_increment` set to True, and checks if the returned 
    cropped image and mask sizes and slice box are adjusted correctly.
    """
    image = Image.new("RGB", (100, 100))
    mask = Image.new("L", (100, 100))
    mask.paste(255, [10, 10, 90, 90])
    img_crop, mask_crop, slice_box = pre_input_and_mask(image, mask, slice_increment=128, force_multiple_of_slice_increment=True)
    
    # Check that the crop is at least the minimum size (64x64)
    assert img_crop.size[0] >= 64 and img_crop.size[1] >= 64
    
    # Check that the crop is no larger than the original image
    assert img_crop.size[0] <= 100 and img_crop.size[1] <= 100
    
    # Check that the image and mask crops have the same size
    assert img_crop.size == mask_crop.size
    
    # Check that the slice box matches the crop size
    assert slice_box == (0, 0, img_crop.size[0], img_crop.size[1])
    
    # Optional: Check that the crop contains the entire mask
    mask_crop_array = np.array(mask_crop)
    assert np.any(mask_crop_array > 0)  # Ensure the crop contains some of the mask
    assert np.all(mask_crop_array[0, :] == 0) and np.all(mask_crop_array[-1, :] == 0) and \
           np.all(mask_crop_array[:, 0] == 0) and np.all(mask_crop_array[:, -1] == 0)  # Ensure the mask doesn't touch the edges


def test_pre_input_and_mask_large_image():
    """
    Test the `pre_input_and_mask` function with a very large image and mask.
    This function creates a large image and mask of size 10,000 x 10,000 pixels. 
    The mask is applied to a central region of the image. The `pre_input_and_mask` 
    function is then called with a slice increment of 128 and forcing the increment 
    to be a multiple of the slice increment. It verifies that the cropped image and 
    mask are resized to 1024 x 1024 pixels, and checks that the slice box is approximately
    centered around the mask area and has the correct size.
    """
    image = Image.new("RGB", (10000, 10000))
    mask = Image.new("L", (10000, 10000))
    mask.paste(255, [5000, 5000, 6000, 6000])
    img_crop, mask_crop, slice_box = pre_input_and_mask(image, mask, slice_increment=128, force_multiple_of_slice_increment=True)
    
    assert img_crop.size == (1024, 1024) 
    assert mask_crop.size == (1024, 1024)
    
    # Check that the slice box is approximately centered around the mask area
    assert 4988 <= slice_box[0] <= 4992 and 4988 <= slice_box[1] <= 4992
    assert 6012 <= slice_box[2] <= 6016 and 6012 <= slice_box[3] <= 6016
    
    # Check that the slice box dimensions are multiples of the slice increment
    assert (slice_box[2] - slice_box[0]) % 128 == 0
    assert (slice_box[3] - slice_box[1]) % 128 == 0
    
    # Check that the slice box has the correct size
    assert slice_box[2] - slice_box[0] == 1024
    assert slice_box[3] - slice_box[1] == 1024

def test_pre_input_and_mask_exact_crop():
    """
    Test exact cropping of the image and mask without forcing multiples of slice_increment.

    This function creates an image and a mask with a specific bounding box, passes
    them to `pre_input_and_mask` with `force_multiple_of_slice_increment` set to False,
    and checks if the returned cropped image and mask sizes and slice box are correct.
    """
    image = Image.new("RGB", (100, 100))
    mask = Image.new("L", (100, 100))
    mask.paste(255, [10, 10, 90, 90])
    img_crop, mask_crop, slice_box = pre_input_and_mask(image, mask, force_multiple_of_slice_increment=False)

    # The cropped size should account for the bounding box of the mask.
    expected_slice_box = (10, 10, 90, 90)
    expected_size = (expected_slice_box[2] - expected_slice_box[0], expected_slice_box[3] - expected_slice_box[1])

    assert img_crop.size == expected_size, f"Expected size {expected_size}, but got {img_crop.size}"
    assert mask_crop.size == expected_size, f"Expected size {expected_size}, but got {mask_crop.size}"
    assert slice_box == expected_slice_box, f"Expected slice box {expected_slice_box}, but got {slice_box}"



def test_pre_input_and_mask_invalid_size():
    """
    Test that an image and mask of different sizes raises a ValueError.

    This function creates an image and a mask of different sizes, passes them to 
    `pre_input_and_mask`, and verifies that it raises a ValueError.
    """
    image = Image.new("RGB", (100, 100))
    mask = Image.new("L", (200, 200))
    with pytest.raises(ValueError):
        pre_input_and_mask(image, mask)

def test_calc_out_size():
    """
    Test calculation of output size when resizing is necessary.

    This function calculates the output size for given width and height that 
    require resizing, and verifies that the new dimensions are correct and the 
    ratio is calculated properly.
    """
    width, height = 800, 600
    new_width, new_height, ratio = calc_out_size(width, height)
    assert new_width == 768
    assert new_height == 576
    assert ratio == 1.0416666666666667

def test_calc_out_size_no_resize():
    """
    Test calculation of output size when resizing is not necessary.

    This function calculates the output size for given width and height that do 
    not require resizing, and verifies that the dimensions remain almost the same 
    and the ratio is 1.0.
    """
    width, height = 700, 500
    new_width, new_height, ratio = calc_out_size(width, height)
    assert new_width == 704
    assert new_height == 504
    assert ratio == 1.0

def test_make_multiple_of_8():
    """
    Test calculation of the nearest multiple of 8 for given values.

    This function tests `make_multiple_of_8` with various inputs and verifies that 
    the returned values are the closest multiples of 8 greater than or equal to the inputs.
    """
    assert make_multiple_of_8(5) == 8
    assert make_multiple_of_8(10) == 16
    assert make_multiple_of_8(16) == 16

def test_resize_by_max():
    """
    Test resizing of an image to fit within the maximum size.

    This function creates a large image, resizes it using `resize_by_max` with a 
    specified max size, and verifies that the resized dimensions and the ratio are correct.
    """
    image = Image.new("RGB", (1000, 500))
    resized_img, ratio = resize_by_max(image, 768)
    assert resized_img.size == (768, 384)
    assert ratio == 1.3020833333333333

def test_resize_by_max_no_resize():
    """
    Test resizing of an image that does not need resizing.

    This function creates an image that fits within the max size, resizes it using 
    `resize_by_max`, and verifies that the dimensions and the ratio remain unchanged.
    """
    image = Image.new("RGB", (700, 500))
    resized_img, ratio = resize_by_max(image, 768)
    assert resized_img.size == (700, 500)
    assert ratio == 1.0

def test_slice_image():
    """
    Test slicing of an image into six equal parts.

    This function creates a numpy array representing an image, slices it using 
    `slice_image`, and verifies that the resulting slices are of the correct dimensions 
    and quantity.
    """
    image = np.ones((300, 200, 3), dtype=np.uint8) * 255
    slices = slice_image(image)
    assert len(slices) == 6
    assert all(slice.shape == (100, 100, 3) for slice in slices)

def test_slice_image_invalid_format():
    """
    Test that an invalid image format raises an UnsupportedFormat exception.

    This function passes an invalid type (string) to `slice_image` and verifies 
    that it raises an `UnsupportedFormat` exception.
    """
    with pytest.raises(UnsupportedFormat):
        slice_image("invalid_type")

def test_matte_matting():
    """
    Test the MatteMatting class functionality.

    This function creates an image and a mask, initializes a `MatteMatting` instance 
    with them, and verifies that the instance and the exported image are created correctly.
    """
    image = Image.new("RGB", (100, 100))
    mask = Image.new("L", (100, 100))
    mask.paste(255, [10, 10, 90, 90])
    matte = MatteMatting(image, mask)
    assert isinstance(matte, MatteMatting)
    exported_img = matte.export_image()
    assert isinstance(exported_img, Image.Image)

def test_matte_matting_mask_flip():
    """
    Test the MatteMatting class functionality with mask flip.

    This function creates an image and a mask, initializes a `MatteMatting` instance 
    with them, and verifies that the exported image is created correctly with the mask 
    colors flipped.
    """
    image = Image.new("RGB", (100, 100))
    mask = Image.new("L", (100, 100))
    mask.paste(255, [10, 10, 90, 90])
    matte = MatteMatting(image, mask)
    assert isinstance(matte, MatteMatting)
    exported_img = matte.export_image(mask_flip=True)
    assert isinstance(exported_img, Image.Image)

def test_unsupported_format_exception():
    """
    Test the UnsupportedFormat exception.

    This function raises an `UnsupportedFormat` exception and verifies that it 
    is raised correctly.
    """
    with pytest.raises(UnsupportedFormat):
        raise UnsupportedFormat('invalid_type')
