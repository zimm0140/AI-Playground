"""
Real-ESRGAN Super-Resolution Module
----------------------------------
This module implements image super-resolution using the Real-ESRGAN model.

Real-ESRGAN is an enhanced super-resolution generative adversarial network that
improves upon ESRGAN with pure synthetic data and improves generalization and
performs better restoration quality on real-world images with complex degradations.

The module provides:
- RealESRGANer class for upscaling images with tiling support
- Utility classes for efficient batch processing
- Support for different image formats and alpha channel handling
"""

import gc
import math
import os
import queue
import threading

import cv2
import numpy as np
import PIL
import PIL.Image
import service_config
import torch
import xpu_hijacks
from torch.nn import functional as F

from basicsr.archs.rrdbnet_arch import RRDBNet

# Apply Intel XPU (GPU) hijacks to make PyTorch operations work on Intel GPUs
xpu_hijacks.ipex_hijacks()

# Path and device configuration
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# URLs for pre-trained model weights
ESRGAN_MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
# ESRGAN_MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"


class RealESRGANer:
    """A helper class for upsampling images with RealESRGAN.

    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        model_path (str): The path to the pretrained model. It can be urls (will first download it automatically).
        model (nn.Module): The defined network. Default: None.
        tile (int): As too large images result in the out of GPU memory issue, so this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image.
            0 denotes for do not use tile. Default: 0.
        tile_pad (int): The pad size for each tile, to remove border artifacts. Default: 10.
        pre_pad (int): Pad the input images to avoid border artifacts. Default: 10.
        half (float): Whether to use half precision during inference. Default: False.
    """

    model: RRDBNet
    deivce: torch.device

    def __init__(self, tile=0, tile_pad=10, pre_pad=10, half=False):
        """
        Initialize the RealESRGANer with specified parameters.

        Sets up the model architecture based on the available model weights,
        loads the pre-trained weights, and configures the processing parameters.
        """
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.scale = 2
        self.half = half
        self.deivce = torch.device(DEVICE)
        model_path: str = os.path.abspath(
            os.path.join(service_config.service_model_paths.get("ESRGAN"), ESRGAN_MODEL_URL.split("/")[-1]),
        )
        # Choose model architecture based on model filename (2x or 4x upscaling)
        if model_path.endswith("RealESRGAN_x2plus.pth"):
            self.model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
        else:
            self.model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
        # Load pre-trained weights
        state_dicts = torch.load(model_path, map_location=self.deivce)

        # prefer to use params_ema
        keyname = "params_ema" if "params_ema" in state_dicts else "params"
        self.model.load_state_dict(state_dicts[keyname], strict=True)

        self.model.eval()
        self.model = self.model.to(self.deivce)
        if self.half:
            self.model = self.model.half()

    def to(
        self,
        device: str,
    ):
        """
        Move the model to the specified device.

        Args:
            device: The target device (converts 'xpu' to 'cuda' for Intel GPU compatibility)
        """
        self.model.to(device.replace("xpu", "cuda"))

    def dni(self, net_a, net_b, dni_weight, key="params", loc="cpu"):
        """Deep network interpolation.

        ``Paper: Deep Network Interpolation for Continuous Imagery Effect Transition``

        Performs model interpolation between two pre-trained networks to achieve
        continuous transition effects.

        Args:
            net_a: Path to the first model
            net_b: Path to the second model
            dni_weight: Interpolation weights as a list of two values
            key: Key for accessing model parameters
            loc: Device for loading the models

        Returns:
            Interpolated model state dict
        """
        net_a = torch.load(net_a, map_location=torch.device(loc))
        net_b = torch.load(net_b, map_location=torch.device(loc))
        for k, v_a in net_a[key].items():
            net_a[key][k] = dni_weight[0] * v_a + dni_weight[1] * net_b[key][k]
        return net_a

    def pre_process(self, img):
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible

        Prepares the input image for processing by:
        1. Converting it to a PyTorch tensor
        2. Adding padding to avoid border artifacts
        3. Ensuring dimensions are divisible by the scale factor

        Args:
            img: Input image as a NumPy array
        """
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.deivce)
        if self.half:
            self.img = self.img.half()

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), "reflect")
        # mod pad for divisible borders
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if h % self.mod_scale != 0:
                self.mod_pad_h = self.mod_scale - h % self.mod_scale
            if w % self.mod_scale != 0:
                self.mod_pad_w = self.mod_scale - w % self.mod_scale
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), "reflect")

    def process(self):
        """
        Process the entire image at once using the model.

        Performs a forward pass through the Real-ESRGAN model
        for images that can fit in memory without tiling.
        """
        # model inference
        self.output = self.model(self.img)

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher

        This method handles large images by:
        1. Dividing the image into overlapping tiles
        2. Processing each tile separately
        3. Merging the processed tiles back together
        4. Handling tile overlaps to reduce boundary artifacts
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]

                # upscale tile
                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile)
                except RuntimeError as error:
                    print("Error", error)
                print(f"\tTile {tile_idx}/{tiles_x * tiles_y}")

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]

    def post_process(self):
        """
        Post-process the output to remove padding.

        Removes the padding added during pre-processing to get the
        final enhanced image at the correct dimensions.

        Returns:
            torch.Tensor: The processed output image tensor
        """
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : h - self.mod_pad_h * self.scale,
                0 : w - self.mod_pad_w * self.scale,
            ]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : h - self.pre_pad * self.scale,
                0 : w - self.pre_pad * self.scale,
            ]
        return self.output

    @torch.no_grad()
    def enhance(
        self,
        img: np.ndarray | PIL.Image.Image,
        outscale: int = None,
        alpha_upsampler="realesrgan",
    ):
        """
        Enhance an image using the Real-ESRGAN model.

        Main entry point for image enhancement, handling different image formats,
        color spaces, and optional alpha channel processing.

        Args:
            img: Input image as a PIL Image or NumPy array
            outscale: Optional output scale factor (overrides default model scale)
            alpha_upsampler: Method for alpha channel upsampling ('realesrgan' or default cv2)

        Returns:
            tuple: (Enhanced image as a NumPy array, Image mode string)
        """
        # Prepare the input image
        img, img_mode, alpha = self._prepare_input_image(img)
        
        # Process the main image
        self._process_main_image(img)
        
        # Handle post-processing and potential alpha channel
        return self._finalize_output(img_mode, alpha, alpha_upsampler, outscale)

    def _prepare_input_image(self, img):
        """Prepare input image for enhancement by normalizing and detecting mode."""
        if isinstance(img, PIL.Image.Image):
            img = np.array(img)
        
        # Normalize image values
        img = img.astype(np.float32)
        max_range = 65535 if np.max(img) > 256 else 255
        img = img / max_range
        
        # Determine image mode and handle alpha channel if present
        alpha = None
        if len(img.shape) == 2:  # Grayscale image
            img_mode = "L"
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = "RGBA"
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_mode = "RGB"
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        return img, img_mode, alpha
        
    def _process_main_image(self, img):
        """Process the main image content using the model."""
        self.pre_process(img)
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
    
    def _finalize_output(self, img_mode, alpha, alpha_upsampler, outscale):
        """Apply post-processing and handle alpha channel if needed."""
        # Apply output scaling if specified
        if outscale is not None and outscale != self.scale:
            self.output = self._rescale_output(outscale)
        else:
            self.output = self.post_process()
            
        # Handle alpha channel for RGBA images
        if img_mode == "RGBA":
            return self._process_with_alpha(alpha, alpha_upsampler)
        
        # Handle grayscale images
        if img_mode == "L":
            return self._convert_to_grayscale()
            
        # Default case: return RGB image
        return self.output, img_mode
    
    def _rescale_output(self, outscale):
        """Rescale output to desired scale factor."""
        h, w = self.output.shape[0:2]
        new_h, new_w = int(h * outscale / self.scale), int(w * outscale / self.scale)
        scaled_output = cv2.resize(
            self.output, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
        )
        return scaled_output
    
    def _process_with_alpha(self, alpha, alpha_upsampler):
        """Process image with alpha channel."""
        h, w = self.output.shape[0:2]
        if alpha_upsampler == "realesrgan":
            # Use the same model to upsample alpha channel
            self.pre_process(alpha)
            if self.tile_size > 0:
                self.tile_process()
            else:
                self.process()
            upsampled_alpha = self.post_process()
            upsampled_alpha = upsampled_alpha[:, :, 0]
        else:
            # Use simple upsampling for alpha channel
            upsampled_alpha = cv2.resize(
                alpha, (w, h), interpolation=cv2.INTER_LINEAR
            )
        
        # Merge the RGB channels with the alpha channel
        output_with_alpha = np.concatenate(
            (self.output, upsampled_alpha[:, :, None]), axis=2
        )
        return output_with_alpha, "RGBA"
    
    def _convert_to_grayscale(self):
        """Convert output to grayscale for L mode images."""
        self.output = cv2.cvtColor(self.output, cv2.COLOR_BGR2GRAY)
        return self.output, "L"

    def dispose(self):
        """
        Clean up resources used by the model.

        Moves the model to CPU, deletes it, and clears GPU memory
        to free up resources when the model is no longer needed.
        """
        self.model.cpu()
        del self.model
        gc.collect()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE == "xpu":
            torch.xpu.empty_cache()


class PrefetchReader(threading.Thread):
    """Prefetch images.

    Args:
        img_list (list[str]): A image list of image paths to be read.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, img_list, num_prefetch_queue):
        """
        Initialize the prefetch reader thread.

        Creates a queue and prepares to load images in a separate thread
        to improve processing efficiency.

        Args:
            img_list: List of image file paths to process
            num_prefetch_queue: Size of the prefetch queue
        """
        super().__init__()
        self.que = queue.Queue(num_prefetch_queue)
        self.img_list = img_list

    def run(self):
        """
        Thread execution method that loads images into the queue.

        Reads each image from disk and places it in the queue,
        then signals completion with None.
        """
        for img_path in self.img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            self.que.put(img)

        self.que.put(None)

    def __next__(self):
        """
        Get the next image from the queue for iteration.

        Returns:
            The next loaded image, or raises StopIteration
        """
        next_item = self.que.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        """
        Make the class iterable.

        Returns:
            Self as an iterator
        """
        return self


class IOConsumer(threading.Thread):
    """
    Thread for saving processed images to disk.

    Handles I/O operations in a separate thread to avoid blocking
    the main processing thread.
    """

    def __init__(self, opt, que, qid):
        """
        Initialize the I/O consumer thread.

        Args:
            opt: Options/configuration for the consumer
            que: Queue from which to get image data for saving
            qid: Queue ID for identification
        """
        super().__init__()
        self._queue = que
        self.qid = qid
        self.opt = opt

    def run(self):
        """
        Thread execution method that saves images from the queue.

        Continuously processes messages from the queue:
        - Writes images to disk when image data is received
        - Exits when a 'quit' message is received
        """
        while True:
            msg = self._queue.get()
            if isinstance(msg, str) and msg == "quit":
                break

            output = msg["output"]
            save_path = msg["save_path"]
            cv2.imwrite(save_path, output)
        print(f"IO worker {self.qid} is done.")
