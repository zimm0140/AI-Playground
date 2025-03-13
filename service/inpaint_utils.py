'''"""
Inpaint Utilities Module
------------------------
This module provides utility functions for image inpainting tasks.
It contains functions to preprocess images and masks, perform slicing, and compute output sizes.

Functions:
    - get_image_ndarray: Convert an image to a NumPy array if needed.
    - detect_mask_valid_edge: Detect the bounding box of the valid mask region.
    - pre_input_and_mask: Prepare and crop the input image and mask based on valid edges.
    - calc_out_size: Calculate output dimensions and scale ratio for inpainting.
    - make_multiple_of_8: Adjust a value to be a multiple of 8.
    - resize_by_max: Resize an image to a maximum size with optional adjustment to a multiple of 8.
    - slice_image: Slice an image into multiple sub-images.

Classes:
    - UnsupportedFormat: Exception raised for unsupported image formats. Contains non-English error message.
    - MatteMatting: Provides matte matting operations for combining image and mask, creating transparency.

Note: All non-English comments and commented-out code are preserved.
"""'''


import cv2
import numpy as np
from PIL import Image


def get_image_ndarray(image: Image.Image | np.ndarray) -> np.ndarray:
    """
    Convert an input image to a NumPy array.

    Args:
        image: The input image, either as a PIL Image or a NumPy array.

    Returns:
        A NumPy array representation of the image.
    """
    if isinstance(image, Image.Image):
        return np.array(image)
    return image


def detect_mask_valid_edge(
    mask_image: Image.Image | np.ndarray,
) -> tuple[int, int, int, int]:
    """
    Detect the valid edge coordinates in a mask image.

    This function finds the top, bottom, left, and right boundaries of non-zero regions in the mask.

    Args:
        mask_image: The mask image as a PIL Image or NumPy array.

    Returns:
        A tuple (left, top, right, bottom) representing the bounding box of the valid mask area.
    """
    mask = get_image_ndarray(mask_image)

    indices = np.where(mask > 0)

    top, bottom = np.min(indices[0]), np.max(indices[0])

    left, right = np.min(indices[1]), np.max(indices[1])

    print(f"detect top:{top},bottom:{bottom}, left:{left},right:{right}")

    return (left, top, right, bottom)


def pre_input_and_mask(
    image: Image.Image, mask: Image.Image,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    """
    Preprocess and crop the input image and mask for inpainting.

    The function resizes the mask to match the image, detects the valid region of the mask,
    and if the mask valid edge is not equal to the image edge, it crops the image and mask around
    the center of the valid mask region using a calculated slice box.

    Args:
        image: The input image as a PIL Image.
        mask: The input mask as a PIL Image.

    Returns:
        A tuple containing:
            - The cropped image (PIL Image).
            - The resized and cropped mask (PIL Image).
            - The slice box coordinates as a tuple (left, top, right, bottom) or (0, 0) if no cropping was applied.
    """
    iw, ih = image.size
    mask_resize = mask.resize(image.size)
    ml, mt, mr, mb = detect_mask_valid_edge(mask_resize)
    # if mask valid edge equals input image edge, don't slice image
    if ml == 0 and mt == 0 and mb == ih - 1 and mr == iw - 1:
        return image, mask_resize, (0, 0)

    mask_width_half = (mr - ml) // 2
    mask_height_half = (mb - mt) // 2

    slice_width_half = 0
    slice_height_half = 0
    while mask_width_half > slice_width_half:
        slice_width_half += 128
    while mask_height_half > slice_height_half:
        slice_height_half += 128

    center_x = ml + mask_width_half
    center_y = mt + mask_height_half

    left = max(0, center_x - slice_width_half)
    top = max(0, center_y - slice_height_half)
    right = min(iw, center_x + slice_width_half)
    bottom = min(ih, center_y + slice_height_half)

    # slice_height = bottom - top
    # slice_width = right - left

    # calc_out_size(slice_width, slice_height)

    slice_box = (left, top, right, bottom)

    return image.crop(slice_box), mask_resize.crop(slice_box), slice_box


def calc_out_size(width: int, height: int, is_sdxl=False) -> tuple[int, int, int]:
    """
    Calculate the output size and scaling ratio for inpainted image generation.

    If the width (or height) exceeds a maximum value (1536 for SDXL models or 768 otherwise),
    the function calculates a scaling ratio and returns the new dimensions adjusted to a multiple of 8.

    Args:
        width: The width of the input image.
        height: The height of the input image.
        is_sdxl: Whether the model is Stable Diffusion XL (default False).

    Returns:
        A tuple (new_width, new_height, ratio) where ratio is the scaling factor applied.
    """
    max = 1536 if is_sdxl else 768
    if width > height:
        if width > max:
            radio = width / max
            return max, make_multiple_of_8(int(height / radio)), radio
    elif height > max:
        radio = height / max
        return make_multiple_of_8(int(width / radio)), max, radio
    return make_multiple_of_8(width), make_multiple_of_8(height), 1


def make_multiple_of_8(value: int):
    """
    Adjust an integer value to be a multiple of 8.

    Args:
        value: The input integer value.

    Returns:
        The largest multiple of 8 that is less than or equal to the input value.
    """
    return value // 8 * 8


def resize_by_max(image: Image.Image, max_size: int, multiple_of_8=True):
    """
    Resize an image such that its dimensions do not exceed a specified maximum size.

    The function scales the image down based on the larger dimension and,
    if requested, adjusts the new dimensions to be multiples of 8.

    Args:
        image: The input PIL Image.
        max_size: The maximum allowable size for the width or height.
        multiple_of_8: If True, the output dimensions will be adjusted to be multiples of 8.

    Returns:
        A tuple (resized_image, downscale_ratio), where downscale_ratio is the scaling factor applied.
    """
    if image.width > max_size or image.height > max_size:
        if image.width > image.height:
            downscale_ratio = image.width / max_size
            downscale_width = int(image.width / downscale_ratio)
            downscale_height = int(image.height / downscale_ratio)
            if multiple_of_8:
                new_width = make_multiple_of_8(downscale_width)
                new_height = make_multiple_of_8(downscale_height)
            return image.resize((new_width, new_height)), downscale_ratio
        downscale_ratio = image.height / max_size
        downscale_width = int(image.width / downscale_ratio)
        downscale_height = int(image.height / downscale_ratio)
        if multiple_of_8:
            new_width = make_multiple_of_8(downscale_width)
            new_height = make_multiple_of_8(downscale_height)
        return image.resize((new_width, new_height)), downscale_ratio
    return image, 1


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


def slice_image(image: np.ndarray | Image.Image):
    """
    Slice an image into several sub-images.

    The function divides the image into 3 rows and 2 columns (6 slices) based on a computed slice size.
    Adjustments are made for edge cases if slices exceed image dimensions.

    Args:
        image: The input image, either as a NumPy array or a PIL Image.

    Returns:
        A list of slices (sub-images) as NumPy arrays.
    """
    image = get_image_ndarray(image)
    height, width, _ = image.shape
    slice_size = min(width // 2, height // 3)

    slices = []

    for h in range(3):
        for w in range(2):
            left = w * slice_size
            upper = h * slice_size
            right = left + slice_size
            lower = upper + slice_size

            if w == 1 and right > width:
                left -= right - width
                right = width
            if h == 2 and lower > height:
                upper -= lower - height
                lower = height

            slice = image[upper:lower, left:right]
            slices.append(slice)

    return slices


class UnsupportedFormat(Exception):
    """
    Exception raised for unsupported image format conversions.

    The error message is provided in non-English language.
    """

    def __init__(self, input_type):
        self.t = input_type

    def __str__(self):
        return f"不支持'{self.t}'模式的转换，请使用为图片地址(path)、PIL.Image(pil)或OpenCV(cv2)模式"


class MatteMatting:
    """
    Class for performing matte matting on images.

    This class converts images to OpenCV format, processes them to replace white areas with transparency,
    and exports a final image with the matte applied.
    """

    def __init__(self, image: Image.Image, mask_image: Image.Image):
        """
        Initialize with an image and its corresponding mask.

        The images are converted into OpenCV format for further processing.

        Args:
            image: The input image as a PIL Image.
            mask_image: The mask image as a PIL Image.
        """
        self.image = self.__image_to_opencv(image)
        self.mask_image = self.__image_to_opencv(mask_image)

    @staticmethod
    def __transparent_back(img: Image.Image):
        """
        Replace white pixels in an image with transparency.

        Args:
            img: The input image (as a PIL Image) to process.

        Returns:
            A PIL Image with white areas replaced with transparent pixels.

        Note: The docstring below preserves non-English explanation.
        :param img: 传入图片地址
        :return: 返回替换白色后的透明图
        """
        img = img.convert("RGBA")
        W, H = img.size
        color_0 = (255, 255, 255, 255)  # 要替换的颜色
        for h in range(H):
            for w in range(W):
                dot = (w, h)
                color_1 = img.getpixel(dot)
                if color_1 == color_0:
                    color_1 = color_1[:-1] + (0,)
                    img.putpixel(dot, color_1)
        return img

    def export_image(self, mask_flip=False):
        """
        Export the final image after applying matte matting.

        Optionally flips the mask before compositing with the image. The image and mask are
        combined, converted to PIL format, and white pixels are replaced with transparency.

        Args:
            mask_flip: If True, the mask is flipped (inverted) before processing.

        Returns:
            A PIL Image of the final matte-matted result.
        """
        if mask_flip:
            self.mask_image = cv2.bitwise_not(self.mask_image)  # 黑白翻转
        image = cv2.add(self.image, self.mask_image)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # OpenCV转换成PIL.Image格式
        return self.__transparent_back(image)

    @staticmethod
    def __image_to_opencv(image: Image.Image):
        """
        Convert a PIL Image to an OpenCV image (BGR format).

        Args:
            image: A PIL Image.

        Returns:
            An OpenCV image in BGR format as a NumPy array.
        """
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


# print(arr)
# if __name__ == "__main__":
#     input_image = Image.open("./test/images/women.png")
#     mask_image = Image.open("./test/images/inapint_mask.png")
#     (ori_width, ori_height) = input_image.size
#
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
#
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
#
#         gen_image.save(f"./inapint_gen_{i}.png")
#
#         if is_scale_out:
#             scalce_radio = 1 // out_radio
#             realESRGANer = RealESRGANer()
#             gen_image = realESRGANer.enhance(gen_image, scalce_radio)
#
#         if real_out_h != out_height or real_out_w != out_width:
#             combine_mask_image = mask_image.resize((out_width, out_height))
#             gen_image = gen_image.resize((out_width, out_height))
#
#         else:
#             combine_mask_image = mask_image
#
#         combine_mask_image = Image.fromarray(
#             cv2.bitwise_not(np.asarray(combine_mask_image))
#         )
#         combine_mask_image.show()
#         mm = MatteMatting(gen_image, combine_mask_image)
#         gen_image = mm.export_image()
#         gen_image.save(f"./inapint_gen_mm_{i}.png")
#         r, g, b, a = gen_image.split()
#         input_image.paste(gen_image, slice_box, a)
#
#         input_image.save(f"./inpaint_result_{i}.png")
#
#         i += 1
