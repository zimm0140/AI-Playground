import cv2
import torch
import numpy as np
from PIL import Image

# URL for downloading the LaMa model
LAMA_MODEL_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"

def get_image(img):
    """
    Converts an input image to a NumPy array and prepares it for model processing.
    
    This function handles both 2D grayscale and 3D color images. It normalizes pixel values 
    to the range [0, 1] and converts the image format to the required shape (channels, height, width).

    Parameters:
    img (PIL.Image.Image or numpy.ndarray): The input image as a PIL image or a NumPy array.

    Returns:
    numpy.ndarray: The processed image as a NumPy array.
    """
    if isinstance(img, Image.Image):  # Check if the input is a PIL image
        img = np.array(img)  # Convert the PIL image to a NumPy array
    
    if img.ndim == 3:  # If the image has three dimensions (HWC format)
        img = np.transpose(img, (2, 0, 1))  # Rearrange to CHW format (channels, height, width)
    elif img.ndim == 2:  # If the image is grayscale (2D)
        img = img[np.newaxis, ...]  # Add a new axis for the channel dimension
    
    img = img.astype(np.float32) / 255  # Normalize the image to the range [0, 1]
    return img

def prepare_img_and_mask(image, mask, device, pad_out_to_modulo=8, scale_factor=None):
    def ceil_modulo(x, mod):
        if x % mod == 0:
            return x
        return (x // mod + 1) * mod

    def get_image(img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))  # chw
        elif img.ndim == 2:
            img = img[np.newaxis, ...]
        img = img.astype(np.float32) / 255
        return img

    def pad_img_to_modulo(img, mod):
        _channels, height, width = img.shape
        out_height = ceil_modulo(height, mod)
        out_width = ceil_modulo(width, mod)
        return np.pad(
            img,
            ((0, 0), (0, out_height - height), (0, out_width - width)),
            mode="symmetric",
        )

    def scale_image(img, factor, interpolation=cv2.INTER_AREA):
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
    def __init__(self):
        self.device = "xpu"
        model_path = "C:\\Users\\X\\Downloads\\big-lama.pt"
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, image: Image.Image | np.ndarray, mask: Image.Image | np.ndarray):
        if image is None:
            return None
        if mask is None:
            mask = Image.new('L', image.size, 0)
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
    result_image = lama(image,mask_image)
    result_image.show()