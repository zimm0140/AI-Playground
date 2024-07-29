from typing import Tuple
import numpy as np
from PIL import Image
import cv2


def get_image_ndarray(image: Image.Image | np.ndarray) -> np.ndarray:
    """Converts a PIL Image or numpy array to a numpy array."""
    return np.array(image) if isinstance(image, Image.Image) else image


def detect_mask_valid_edge(mask_image: Image.Image | np.ndarray) -> Tuple[int, int, int, int]:
    """Detects the valid edges (bounding box) of a mask."""
    mask = get_image_ndarray(mask_image)
    indices = np.where(mask > 0)
    top, bottom = np.min(indices[0]), np.max(indices[0])
    left, right = np.min(indices[1]), np.max(indices[1])
    print(f"detect top:{top}, bottom:{bottom}, left:{left}, right:{right}")
    return left, top, right, bottom


def pre_input_and_mask(image: Image.Image, mask: Image.Image, slice_increment=128) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
    """Prepares the input image and mask for inpainting."""
    iw, ih = image.size
    mask_resize = mask.resize(image.size)
    ml, mt, mr, mb = detect_mask_valid_edge(mask_resize)

    # If mask covers the entire image, no slicing is needed
    if (ml, mt, mb, mr) == (0, 0, ih - 1, iw - 1):
        return image, mask_resize, (0, 0, iw, ih) 

    mask_width_half = (mr - ml) // 2
    mask_height_half = (mb - mt) // 2

    slice_width_half = (mask_width_half + slice_increment - 1) // slice_increment * slice_increment
    slice_height_half = (mask_height_half + slice_increment - 1) // slice_increment * slice_increment

    center_x = ml + mask_width_half
    center_y = mt + mask_height_half

    left = max(0, center_x - slice_width_half)
    top = max(0, center_y - slice_height_half)
    right = min(iw, center_x + slice_width_half)
    bottom = min(ih, center_y + slice_height_half)

    slice_box = (left, top, right, bottom)
    return image.crop(slice_box), mask_resize.crop(slice_box), slice_box


def calc_out_size(width: int, height: int, max_size: int = 768) -> Tuple[int, int, float]:
    """Calculates output size ensuring it's a multiple of 8 and within max_size."""
    if width > max_size or height > max_size:
        ratio = max(width / max_size, height / max_size)
        return make_multiple_of_8(int(width / ratio)), make_multiple_of_8(int(height / ratio)), ratio
    return make_multiple_of_8(width), make_multiple_of_8(height), 1


def make_multiple_of_8(value: int) -> int:
    """Returns the closest multiple of 8."""
    return (value + 7) // 8 * 8


def resize_by_max(image: Image.Image, max_size: int = 768) -> Tuple[Image.Image, float]:
    """Resizes the image if necessary to fit within max_size, maintaining aspect ratio."""
    if image.width > max_size or image.height > max_size:
        ratio = max(image.width / max_size, image.height / max_size)
        new_width = int(image.width / ratio)
        new_height = int(image.height / ratio)
        return image.resize((new_width, new_height)), ratio
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


def slice_image(image: np.ndarray | Image.Image) -> list[np.ndarray]: 
    """Slices an image into six equal parts."""
    image = get_image_ndarray(image)
    height, width, _ = image.shape
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

    return slices


class UnsupportedFormat(Exception):
    """Custom exception for unsupported image formats."""
    def __init__(self, input_type):
        super().__init__(f"Unsupported format: '{input_type}'. Use PIL.Image or numpy array.")

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

    @staticmethod
    def _transparent_back(img: Image.Image, transparent_color=(255, 255, 255, 255)) -> Image.Image:
        """Replaces a specific color with transparency."""
        img = img.convert("RGBA")
        data = np.array(img)
        r, g, b, a = data.T
        areas = (r == transparent_color[0]) & (g == transparent_color[1]) & (b == transparent_color[2])
        data[areas.T] = (0, 0, 0, 0) 
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
            self.mask_image = cv2.bitwise_not(self.mask_image)  # 黑白翻转 # Black and white flip
        image = cv2.add(self.image, self.mask_image)
        image = Image.fromarray(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )  # OpenCV转换成PIL.Image格式 # Convert OpenCV to PIL.Image format
        return self._transparent_back(image)

    @staticmethod
    def _image_to_opencv(image: Image.Image) -> np.ndarray:
        """Converts a PIL Image to an OpenCV image (BGR)."""
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
