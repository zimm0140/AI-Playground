from typing import Tuple, Union
import numpy as np
from PIL import Image
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_image_ndarray(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """
    Converts a PIL Image or numpy array to a numpy array.

    Args:
        image: A PIL Image or numpy array.

    Returns:
        A numpy array representation of the image.
    
    Raises:
        TypeError: If the input is neither a PIL Image nor a numpy array.
    """
    if isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise TypeError("Unsupported image type. Use PIL.Image or numpy array.")

def detect_mask_valid_edge(mask_image: Union[Image.Image, np.ndarray]) -> Tuple[int, int, int, int]:
    """
    Detects the valid edges (bounding box) of a mask.

    Args:
        mask_image: A PIL Image or numpy array representing the mask.

    Returns:
        A tuple (left, top, right, bottom) representing the bounding box of the mask.
        The right and bottom coordinates are adjusted to be exclusive bounds by adding 1.

    Raises:
        ValueError: If the mask has no non-zero values.
    """
    mask = get_image_ndarray(mask_image)
    
    # Ensure mask is binary
    mask = np.where(mask > 0, 1, 0)

    indices = np.where(mask > 0)
    if indices[0].size == 0 or indices[1].size == 0:
        raise ValueError("The mask has no non-zero values.")
    
    top, bottom = np.min(indices[0]), np.max(indices[0])
    left, right = np.min(indices[1]), np.max(indices[1])
    logger.info(f"detect top:{top}, bottom:{bottom}, left:{left}, right:{right}")
    return left, top, right + 1, bottom + 1

def pre_input_and_mask(image: Image.Image, mask: Image.Image, slice_increment=128) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
    """
    Prepares the input image and mask for inpainting.

    Args:
        image: The input image.
        mask: The mask image.
        slice_increment: The increment to adjust slice dimensions to, typically required by the model.

    Returns:
        Tuple containing cropped image, cropped mask, and the slice box coordinates.

    Raises:
        ValueError: If the image and mask sizes do not match.
    """
    if image.size != mask.size:
        raise ValueError("The image and mask sizes must match.")
    
    iw, ih = image.size
    mask_resize = mask.resize(image.size)
    ml, mt, mr, mb = detect_mask_valid_edge(mask_resize)

    # If mask covers the entire image, no slicing is needed
    if (ml, mt, mr, mb) == (0, 0, iw, ih):
        logger.info("Mask covers the entire image. No slicing needed.")
        return image, mask_resize, (0, 0, iw, ih) 

    mask_width_half = (mr - ml) // 2
    mask_height_half = (mb - mt) // 2

    slice_width_half = ((mask_width_half + slice_increment - 1) // slice_increment) * slice_increment
    slice_height_half = ((mask_height_half + slice_increment - 1) // slice_increment) * slice_increment

    center_x = ml + mask_width_half
    center_y = mt + mask_height_half

    left = max(0, center_x - slice_width_half)
    top = max(0, center_y - slice_height_half)
    right = min(iw, center_x + slice_width_half)
    bottom = min(ih, center_y + slice_height_half)

    slice_box = (left, top, right, bottom)
    logger.info(f"Cropping image and mask to box: {slice_box}")
    return image.crop(slice_box), mask_resize.crop(slice_box), slice_box


def calc_out_size(width: int, height: int, max_size: int = 768) -> Tuple[int, int, float]:
    """
    Calculates the output size ensuring it's a multiple of 8 and within max_size.

    Args:
        width: The original width of the image.
        height: The original height of the image.
        max_size: The maximum allowed size for the width or height.

    Returns:
        A tuple (new_width, new_height, ratio) where new_width and new_height
        are the adjusted dimensions that are multiples of 8, and ratio is the scaling factor used.
    """
    if width > max_size or height > max_size:
        ratio = max(width / max_size, height / max_size)
        new_width = make_multiple_of_8(int(width / ratio))
        new_height = make_multiple_of_8(int(height / ratio))
        logger.info(f"Resizing to {new_width}x{new_height} with ratio {ratio:.2f} to fit within max size {max_size}.")
        return new_width, new_height, ratio
    logger.info(f"No resizing needed. Original size {width}x{height} is within max size {max_size}.")
    return make_multiple_of_8(width), make_multiple_of_8(height), 1.0

def make_multiple_of_8(value: int) -> int:
    """
    Returns the closest multiple of 8 greater than or equal to the input value.

    Args:
        value: The input value to adjust.

    Returns:
        The closest multiple of 8 greater than or equal to the input value.
    """
    adjusted_value = (value + 7) // 8 * 8
    logger.debug(f"Adjusted {value} to the nearest multiple of 8: {adjusted_value}.")
    return adjusted_value

def resize_by_max(image: Image.Image, max_size: int = 768) -> Tuple[Image.Image, float]:
    """
    Resizes the image if necessary to fit within max_size, maintaining aspect ratio.

    Args:
        image: The input PIL Image to resize.
        max_size: The maximum allowed size for the width or height.

    Returns:
        A tuple (resized_image, ratio) where resized_image is the resized PIL Image
        and ratio is the scaling factor used.
    """
    if image.width > max_size or image.height > max_size:
        ratio = max(image.width / max_size, image.height / max_size)
        new_width = int(image.width / ratio)
        new_height = int(image.height / ratio)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.info(f"Resized image to {new_width}x{new_height} with ratio {ratio:.2f} to fit within max size {max_size}.")
        return resized_image, ratio
    logger.info(f"No resizing needed. Original size {image.width}x{image.height} is within max size {max_size}.")
    return image, 1.0


# def resize_by_max(image: Image.Image, max_size):
#     if image.width > max_size or image.height > max_size:
#         aspect_ratio = image.width / image.height
#         if image.width > image.height:
#             return image.resize(
#                 (max_size, int(image.height / aspect_ratio))
#             ), image.width / max_size
#         else:
#             return image.resize(
#                 (int(image.width * aspect_ratio), max_size)
#             ), image.height / max_size
#     return image, 1


import logging
from typing import Union, List

logger = logging.getLogger(__name__)

def slice_image(image: Union[np.ndarray, Image.Image]) -> List[np.ndarray]: 
    """
    Slices an image into six equal parts.

    Args:
        image: A PIL Image or numpy array to be sliced.

    Returns:
        A list of numpy arrays, each representing a slice of the original image.

    Raises:
        UnsupportedFormat: If the input image format is not supported.
    """
    try:
        image = get_image_ndarray(image)
    except TypeError as e:
        raise UnsupportedFormat(type(image).__name__) from e

    height, width, channels = image.shape
    logger.info(f"Slicing image of size {width}x{height} into six equal parts.")

    slice_width = width // 2
    slice_height = height // 3

    slices = []
    for h in range(3):
        for w in range(2):
            left = w * slice_width
            upper = h * slice_height
            right = min(left + slice_width, width)
            lower = min(upper + slice_height, height)
            slices.append(image[upper:lower, left:right])
            logger.debug(f"Slice {h*2 + w + 1}: ({upper}, {left}) to ({lower}, {right})")

    return slices

class UnsupportedFormat(Exception):
    """
    Custom exception for unsupported image formats.

    Args:
        input_type: The type of the unsupported input.
    """
    def __init__(self, input_type: str):
        super().__init__(f"Unsupported format: '{input_type}'. Use PIL.Image or numpy array.")
        logger.error(f"Unsupported format encountered: {input_type}")


class MatteMatting:
    """Applies a mask to an image for transparency."""

    def __init__(self, image: Image.Image, mask_image: Image.Image):
        """
        Initializes with the image and mask.

        Args:
            image: The input image.
            mask_image: The mask to apply. White areas will be transparent.
        """
        self.image = self._image_to_opencv(image)
        self.mask_image = self._image_to_opencv(mask_image)
        logger.info("MatteMatting instance created.")

    @staticmethod
    def _transparent_back(img: Image.Image, transparent_color=(255, 255, 255, 255)) -> Image.Image:
        """
        Replaces a specific color with transparency.

        Args:
            img: The input image.
            transparent_color: The color to be made transparent.

        Returns:
            Image.Image: The image with transparency applied.
        """
        img = img.convert("RGBA")
        data = np.array(img)
        r, g, b, a = data.T
        areas = (r == transparent_color[0]) & (g == transparent_color[1]) & (b == transparent_color[2])
        data[areas.T] = (0, 0, 0, 0)
        logger.debug(f"Applied transparency to color: {transparent_color}")
        return Image.fromarray(data)

    def export_image(self, mask_flip=False) -> Image.Image:
        """
        Exports the matted image, optionally flipping the mask.

        Args:
            mask_flip (bool): Whether to flip the mask colors.

        Returns:
            Image.Image: The exported image.
        """
        if mask_flip:
            self.mask_image = cv2.bitwise_not(self.mask_image)  # Black and white flip
            logger.info("Mask colors flipped.")
        try:
            image = cv2.add(self.image, self.mask_image)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert OpenCV to PIL.Image format
            logger.info("Image and mask combined.")
            return self._transparent_back(image)
        except Exception as e:
            logger.error("Failed to export image.", exc_info=True)
            raise e

    @staticmethod
    def _image_to_opencv(image: Image.Image) -> np.ndarray:
        """
        Converts a PIL Image to an OpenCV image (BGR).

        Args:
            image: The PIL Image to convert.

        Returns:
            np.ndarray: The converted OpenCV image.
        """
        logger.debug("Converting PIL Image to OpenCV format.")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


# print(arr)
# if __name__ == "__main__":
#     input_image = Image.open("./test/images/women.png")
#     mask_image = Image.open("./test/images/inapint_mask.png")
#     (ori_width, ori_height) = input_image.size

#     slice_image, mask_image, slice_box = pre_input_and_mask(
#         input_image.convert("RGB"), mask_image
#     )
#     slice_image.save("inapint_slice.png")
#     mask_image.save("inapint_mask.png")
#     slice_w, slice_h = slice_image.size
#     pipe = AutoPipelineForInpainting.from_pretrained(
#         "./models/stable_diffusion/checkpoints/Lykon---DreamShaper",
#         torch_type=torch.bfloat16,
#     )
#     pipe.to("xpu")
#     out_width, out_height, out_radio = calc_out_size(
#         slice_w, slice_h, isinstance(pipe, StableDiffusionXLInpaintPipeline)
#     )
#     is_scale_out = False
#     if out_radio != 1:
#         is_scale_out = True
#         slice_image = slice_image.resize((out_width, out_height))
#         mask_image = mask_image.resize((out_width, out_height))

#     i = 0
#     real_out_w = make_multiple_of_8(out_width)
#     real_out_h = make_multiple_of_8(out_height)
#     while i < 1:
#         with torch.inference_mode():
#             gen_image: Image.Image = pipe(
#                 prompt="Beautiful female face",
#                 image=slice_image,
#                 mask_image=mask_image,
#                 strength=0.4,
#                 width=real_out_w,
#                 height=real_out_h,
#                 guidance_scale=7,
#                 num_inference_steps=40,
#             ).images[0]

#         gen_image.save(f"./inapint_gen_{i}.png")

#         if is_scale_out:
#             scalce_radio = 1 // out_radio
#             realESRGANer = RealESRGANer()
#             gen_image = realESRGANer.enhance(gen_image, scalce_radio)

#         if real_out_h != out_height or real_out_w != out_width:
#             combine_mask_image = mask_image.resize((out_width, out_height))
#             gen_image = gen_image.resize((out_width, out_height))

#         else:
#             combine_mask_image = mask_image

#         combine_mask_image = Image.fromarray(
#             cv2.bitwise_not(np.asarray(combine_mask_image))
#         )
#         combine_mask_image.show()
#         mm = MatteMatting(gen_image, combine_mask_image)
#         gen_image = mm.export_image()
#         gen_image.save(f"./inapint_gen_mm_{i}.png")
#         r, g, b, a = gen_image.split()
#         input_image.paste(gen_image, slice_box, a)

#         input_image.save(f"./inpaint_result_{i}.png")

#         i += 1
