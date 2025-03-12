'''"""
LaMa Inpainting Module
----------------------
This module provides utilities for image inpainting using the LaMa model.
It includes functions for pre-processing images and masks and a class for loading and applying
a pre-trained LaMa model for inpainting tasks.
"""'''

import cv2
import numpy as np
import torch
from PIL import Image

LAMA_MODEL_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"


def get_image(img):
    """
    Convert an input image to a normalized numpy array in CHW format.

    If the input is a PIL Image, it is converted to a numpy array. For 3-dimensional arrays,
    the color channels are transposed to the first dimension. If the image is 2-dimensional,
    an extra channel dimension is added. The pixel values are normalized to the range [0,1].

    Args:
        img: Input image as a PIL Image or numpy array.

    Returns:
        A numpy array representing the image in CHW format with normalized pixel values.
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # chw
    elif img.ndim == 2:
        img = img[np.newaxis, ...]
    img = img.astype(np.float32) / 255
    return img


def prepare_img_and_mask(image, mask, device, pad_out_to_modulo=8, scale_factor=None):
    """
    Prepare an image and its corresponding mask for inpainting.

    This function performs several steps:
      - Converts input image and mask to normalized numpy arrays in CHW format.
      - Optionally scales the image and mask by a given factor.
      - Pads the image and mask so that their dimensions are multiples of a specified modulo.
      - Converts the processed image and mask into torch tensors and moves them to the specified device.
      - Binarizes the mask.

    Args:
        image: Input image as a PIL Image or numpy array.
        mask: Input mask as a PIL Image or numpy array.
        device: The device to which the tensors will be moved.
        pad_out_to_modulo: The modulo value for padding (default is 8).
        scale_factor: Optional scaling factor to resize the image and mask.

    Returns:
        A tuple (out_image, out_mask) where:
          - out_image is a torch tensor of shape [1, C, H, W] with normalized pixel values.
          - out_mask is a binary torch tensor of the same shape indicating mask regions.
    """

    def ceil_modulo(x, mod):
        if x % mod == 0:
            return x
        return (x // mod + 1) * mod

    def get_image(img):
        """Convert input image to numpy array in CHW format and normalize it."""
        if isinstance(img, Image.Image):
            img = np.array(img)
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))  # chw
        elif img.ndim == 2:
            img = img[np.newaxis, ...]
        img = img.astype(np.float32) / 255
        return img

    def pad_img_to_modulo(img, mod):
        """Pad the image so that its height and width are multiples of 'mod'."""
        _channels, height, width = img.shape
        out_height = ceil_modulo(height, mod)
        out_width = ceil_modulo(width, mod)
        return np.pad(
            img,
            ((0, 0), (0, out_height - height), (0, out_width - width)),
            mode="symmetric",
        )

    def scale_image(img, factor, interpolation=cv2.INTER_AREA):
        """Resize the image by a given factor using the specified interpolation method."""
        if img.shape[0] == 1:
            img = img[0]
        else:
            img = np.transpose(img, (1, 2, 0))
        img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)
        if img.ndim == 2:
            img = img[None, ...]
        else:
            img = np.transpose(img, (2, 0, 1))
        return img

    out_image = get_image(image)
    out_mask = get_image(mask)
    out_mask.show()
    if scale_factor is not None:
        out_image = scale_image(out_image, scale_factor)
        out_mask = scale_image(out_mask, scale_factor, interpolation=cv2.INTER_NEAREST)
    if pad_out_to_modulo is not None and pad_out_to_modulo > 1:
        out_image = pad_img_to_modulo(out_image, pad_out_to_modulo)
        out_mask = pad_img_to_modulo(out_mask, pad_out_to_modulo)
    out_image = torch.from_numpy(out_image).unsqueeze(0).to(device)
    out_mask = torch.from_numpy(out_mask).unsqueeze(0).to(device)
    out_mask = (out_mask > 0) * 1
    return out_image, out_mask


# def download_model():
#     parts = urlparse(LAMA_MODEL_URL)
#     hub_dir = get_dir()
#     model_dir = os.path.join(hub_dir, "checkpoints")
#     os.makedirs(os.path.join(model_dir, "hub", "checkpoints"), exist_ok=True)
#     filename = os.path.basename(parts.path)
#     cached_file = os.path.join(model_dir, filename)
#     if not os.path.exists(cached_file):
#         log.info(f'LaMa download: url={LAMA_MODEL_URL} file={cached_file}')
#         hash_prefix = None
#         download_url_to_file(LAMA_MODEL_URL, cached_file, hash_prefix, progress=True)
#     return cached_file


class SimpleLama:
    """
    SimpleLaMa inpainting class.

    This class loads a pre-trained LaMa inpainting model via TorchScript and provides a callable
    interface to inpaint an image given a corresponding mask.

    Attributes:
        device (str): The device on which the model is loaded (default is 'xpu').
        model: The loaded TorchScript LaMa model for inpainting.
    """

    def __init__(self):
        """
        Initialize the SimpleLama model by loading the pre-trained TorchScript model.

        The model is set to evaluation mode and moved to the specified device.
        """
        self.device = "xpu"
        model_path = "C:\\Users\\X\\Downloads\\big-lama.pt"
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, image: Image.Image | np.ndarray, mask: Image.Image | np.ndarray):
        """
        Apply the LaMa inpainting model to the provided image and mask.

        If the image or mask is None, the function handles the case appropriately.
        Pre-processing of the image and mask is done before passing them to the model.
        The output is post-processed to convert it back to a PIL Image.

        Args:
            image: An input image as a PIL Image or numpy array.
            mask: An input mask as a PIL Image or numpy array.

        Returns:
            A PIL Image of the inpainted result, or None if inputs are invalid.
        """
        if image is None:
            return None
        if mask is None:
            mask = Image.new("L", image.size, 0)
            return None
        image, mask = prepare_img_and_mask(image, mask, self.device)
        with torch.inference_mode():
            inpainted = self.model(image, mask)
            cur_res = inpainted[0].permute(1, 2, 0).detach().float().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
            cur_res = Image.fromarray(cur_res)
            return cur_res


if __name__ == "__main__":
    lama = SimpleLama()
    image = Image.open("C:\\Users\\X\\Desktop\\inpaint_test.png")
    mask_image = Image.open("C:\\Users\\X\\Desktop\\1mask.png")
    result_image = lama(image, mask_image)
    result_image.show()
