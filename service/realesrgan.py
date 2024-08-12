import gc
import PIL
import PIL.Image
import cv2
import math
import numpy as np
import os
import queue
import threading
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet # Import the RRDBNet architecture from BasicSR
from torch.nn import functional as F # Import PyTorch functional tools
import model_config  # Import the model configuration
import xpu_hijacks # Import XPU compatibility layer 

# Apply Intel's extensions and compatibility fixes for PyTorch on Intel XPUs
xpu_hijacks.ipex_hijacks()

# Define the root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Determine the device to use for PyTorch: CUDA GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# URL for downloading the pre-trained Real-ESRGAN model (x2 upscaling)
ESRGAN_MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
# Alternative model URL (for x4 upscaling, commented out)
# ESRGAN_MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

class RealESRGANer:
    """
    A helper class for upsampling images using the Real-ESRGAN model.

    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        model_path (str): The path to the pretrained model. It can be URLs (will first download it automatically).
        model (nn.Module): The defined network. Default: None.
        tile (int): As too large images result in the out of GPU memory issue, so this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image.
            0 denotes for do not use tile. Default: 0.
        tile_pad (int): The pad size for each tile, to remove border artifacts. Default: 10.
        pre_pad (int): Pad the input images to avoid border artifacts. Default: 10.
        half (float): Whether to use half precision during inference. Default: False.
    """

    model: RRDBNet # The Real-ESRGAN model
    deivce: torch.device # The device on which the model is running (CPU or GPU)

    def __init__(self, tile=0, tile_pad=10, pre_pad=10, half=False):
        """
        Initializes the RealESRGANer with the given configuration.

        Args:
            tile (int, optional): The tile size for processing large images. Defaults to 0 (no tiling).
            tile_pad (int, optional): The padding size for each tile. Defaults to 10.
            pre_pad (int, optional): The padding size for the entire image before processing. Defaults to 10.
            half (bool, optional): Whether to use half-precision (FP16) for inference. Defaults to False.
        """
        self.tile_size = tile # Size of tiles for processing large images
        self.tile_pad = tile_pad # Padding size for each tile
        self.pre_pad = pre_pad # Padding size for the entire image before processing
        self.mod_scale = None # Scaling factor for mod padding (used for making dimensions divisible)
        self.scale = 2 # Default upscaling factor (2x)
        self.half = half # Whether to use half-precision (FP16)
        self.deivce = torch.device(DEVICE)  # Determine and set the device (CUDA or CPU)

        # Get the model path from the configuration
        model_path: str = os.path.abspath(
            os.path.join(
                model_config.config.get("ESRGAN"), ESRGAN_MODEL_URL.split("/")[-1]
            )
        )

        # Load the appropriate Real-ESRGAN model based on the filename
        if model_path.endswith("RealESRGAN_x2plus.pth"):
            self.model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
            )
        else:
            self.model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
            )
        # Load the pre-trained model weights
        state_dicts = torch.load(model_path, map_location=self.deivce) 

        # Prefer to use the "params_ema" (exponential moving average) weights if available
        if "params_ema" in state_dicts:
            keyname = "params_ema" 
        else:
            keyname = "params"
        # Load the model weights from the state dictionary
        self.model.load_state_dict(state_dicts[keyname], strict=True) 

        # Set the model to evaluation mode and move it to the device
        self.model.eval()
        self.model = self.model.to(self.deivce) 
        if self.half:
            self.model = self.model.half() # Convert the model to half-precision if specified

    def to(self, device: str):
        """
        Moves the model to the specified device.

        Args:
            device (str): The target device ("cpu" or "cuda").
        """
        self.model.to(device.replace("xpu", "cuda")) 

    def dni(self, net_a, net_b, dni_weight, key="params", loc="cpu"):
        """
        Deep Network Interpolation (DNI) method.
        This method blends the parameters of two neural networks to create a new model with combined characteristics.

        Args:
            net_a (str or dict): The path to the first network's state dictionary or the state dictionary itself.
            net_b (str or dict): The path to the second network's state dictionary or the state dictionary itself.
            dni_weight (tuple): A tuple of two floats (w_a, w_b), where w_a + w_b = 1.0. 
                               These are the weights for blending the two networks.
            key (str, optional): The key in the state dictionaries for the parameters to be interpolated. Defaults to "params".
            loc (str, optional): The device to load the models onto. Defaults to "cpu".

        Returns:
            dict: The state dictionary of the interpolated network.
        """
        net_a = torch.load(net_a, map_location=torch.device(loc))  # Load the state dictionary of net_a
        net_b = torch.load(net_b, map_location=torch.device(loc))  # Load the state dictionary of net_b

        # Interpolate the parameters of the two networks
        for k, v_a in net_a[key].items():
            net_a[key][k] = dni_weight[0] * v_a + dni_weight[1] * net_b[key][k] 
        return net_a 

    def pre_process(self, img):
        """
        Pre-processes the input image before feeding it to the model.
        This includes converting the image to a PyTorch tensor, adding padding, and 
        making sure the image dimensions are divisible by the upscaling factor.

        Args:
            img (numpy.ndarray): The input image as a NumPy array.
        """
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()  # Convert to tensor
        self.img = img.unsqueeze(0).to(self.deivce)  # Add batch dimension and move to device
        if self.half:
            self.img = self.img.half() # Convert to half-precision

        # Add pre-padding to the image
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), "reflect")
        
        # Add mod padding to make dimensions divisible by the scale factor
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
            self.img = F.pad(
                self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), "reflect"
            )

    def process(self):
        """Performs model inference on the pre-processed image."""
        self.output = self.model(self.img) 

    def tile_process(self):
        """
        Processes the image in tiles to handle large images that might exceed GPU memory.

        This method divides the image into tiles, processes each tile individually, 
        and then combines the processed tiles back into a single output image. 
        It utilizes padding to avoid border artifacts between tiles.
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # Initialize the output tensor with zeros (black image)
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size) # Calculate the number of tiles in the x-direction
        tiles_y = math.ceil(height / self.tile_size) # Calculate the number of tiles in the y-direction

        # Process each tile
        for y in range(tiles_y):
            for x in range(tiles_x):
                # Extract the current tile from the input image
                ofs_x = x * self.tile_size 
                ofs_y = y * self.tile_size
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width) 
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height) 

                # Add padding to the tile
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # Calculate tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                # Extract the tile with padding
                input_tile = self.img[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]

                # Upscale the tile
                try:
                    with torch.no_grad(): # Disable gradient calculation for inference
                        output_tile = self.model(input_tile) 
                except RuntimeError as error:
                    print("Error", error) # Handle potential runtime errors during inference
                print(f"\tTile {tile_idx}/{tiles_x * tiles_y}") # Print tile processing progress

                # Calculate the output tile's coordinates in the final image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # Calculate the tile coordinates without padding for the output image
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # Place the processed tile into the correct position in the output image
                self.output[
                    :, :, output_start_y:output_end_y, output_start_x:output_end_x
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]

    def post_process(self):
        """
        Post-processes the upscaled image. 
        This involves removing any padding added during pre-processing.
        """
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0: h - self.mod_pad_h * self.scale, 0: w - self.mod_pad_w * self.scale]
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0: h - self.pre_pad * self.scale, 0: w - self.pre_pad * self.scale]
        return self.output

    @torch.no_grad()
    def enhance(
        self,
        img: np.ndarray | PIL.Image.Image,
        outscale: int = None,
        alpha_upsampler="realesrgan",
    ):
        """
        Upscales the input image using the Real-ESRGAN model.

        Args:
            img (np.ndarray | PIL.Image.Image): The input image, either as a NumPy array or a PIL Image.
            outscale (int, optional): The desired upscaling factor. If None, the default scaling factor of 
                                    the model is used. Defaults to None.
            alpha_upsampler (str, optional): The method to use for upsampling the alpha channel (if present). 
                                            Defaults to "realesrgan".

        Returns:
            tuple: A tuple containing the upscaled image (as a NumPy array) and the image mode (e.g., "RGB", "RGBA").
        """
        # Convert PIL image to NumPy array if provided as a PIL Image
        if isinstance(img, PIL.Image.Image):
            img = np.array(img)
        h_input, w_input = img.shape[0:2] 

        img = img.astype(np.float32)  # Convert image data to float32

        # Determine the maximum pixel value for normalization (255 for 8-bit, 65535 for 16-bit)
        if np.max(img) > 256: 
            max_range = 65535
            print("\tInput is a 16-bit image")
        else:
            max_range = 255
        img = img / max_range  # Normalize the image data

        # Handle different image modes (grayscale, RGB, RGBA)
        if len(img.shape) == 2:  # Grayscale image
            img_mode = "L"
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image
            img_mode = "RGBA"
            alpha = img[:, :, 3]  # Extract the alpha channel
            img = img[:, :, 0:3]  # Extract the RGB channels
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB (OpenCV uses BGR by default)
            if alpha_upsampler == "realesrgan":
                # Upsample the alpha channel using Real-ESRGAN
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:  # RGB image
            img_mode = "RGB"
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB 

        # ------------------- Process image (without alpha channel) ------------------- #
        self.pre_process(img) # Pre-process the image 
        if self.tile_size > 0:
            self.tile_process() # Process the image in tiles if tile_size is greater than 0
        else:
            self.process() # Otherwise, process the whole image
        output_img = self.post_process() # Post-process the upscaled image
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()  # Convert to NumPy array
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))  # Transpose to correct channel order
        if img_mode == "L":
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY) # Convert back to grayscale if needed

        # ------------------- Process the alpha channel if necessary ------------------- #
        if img_mode == "RGBA":
            if alpha_upsampler == "realesrgan":
                # Upscale alpha channel using Real-ESRGAN
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha = self.post_process()
                output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # Upscale alpha channel using OpenCV's resize method
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(
                    alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR
                )

            # Merge the upscaled alpha channel back into the output image
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA) 
            output_img[:, :, 3] = output_alpha 

        # ------------------------------ Return ------------------------------ #
        if max_range == 65535:  # Scale back to 16-bit if the input was 16-bit
            output = (output_img * 65535.0).round().astype(np.uint16) 
        else:  # Scale back to 8-bit if the input was 8-bit
            output = (output_img * 255.0).round().astype(np.uint8) 

        # If a different output scale is specified, resize the final output
        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (int(w_input * outscale), int(h_input * outscale)), interpolation=cv2.INTER_LANCZOS4
            )

        return output, img_mode

    def dispose(self):
        """Releases the resources used by the model."""
        self.model.cpu()  # Move the model to CPU to free up GPU memory
        del self.model  # Delete the model instance
        gc.collect()  # Trigger garbage collection

        # Clear the GPU cache if using CUDA or XPU
        if DEVICE == "cuda":
            torch.cuda.empty_cache() 
        elif DEVICE == "xpu":
            torch.xpu.empty_cache() 


class PrefetchReader(threading.Thread):
    """
    A thread for pre-fetching images to improve performance.

    Args:
        img_list (list[str]): A list of paths to the images to be prefetched.
        num_prefetch_queue (int): The size of the queue for prefetched images.
    """
    def __init__(self, img_list, num_prefetch_queue):
        super().__init__()
        self.que = queue.Queue(num_prefetch_queue)  # Initialize the image queue
        self.img_list = img_list  # Store the list of image paths

    def run(self):
        """
        The main thread function for prefetching images.
        Reads images from the image list and puts them into the queue. 
        """
        for img_path in self.img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # Read the image from disk
            self.que.put(img) # Put the image into the queue

        self.que.put(None)  # Put None to signal the end of the queue

    def __next__(self):
        """
        Gets the next image from the prefetch queue.
        Raises a StopIteration exception if the queue is empty (signaled by None).
        """
        next_item = self.que.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        """Returns the iterator for the prefetch queue."""
        return self


class IOConsumer(threading.Thread):
    """
    A thread for handling image I/O (saving images to disk) asynchronously.

    Args:
        opt (Any): Options or configuration for the consumer.
        que (queue.Queue): The queue from which to receive image data and save paths.
        qid (int): The ID of this consumer thread.
    """
    def __init__(self, opt, que, qid):
        super().__init__()
        self._queue = que  # Queue to receive data
        self.qid = qid # ID of this IO consumer
        self.opt = opt  # Options for the consumer

    def run(self):
        """
        The main thread function for the IO consumer.
        Continuously retrieves image data and save paths from the queue and saves 
        the images to disk. 
        """
        while True:
            msg = self._queue.get() 
            # If the "quit" message is received, exit the loop
            if isinstance(msg, str) and msg == "quit": 
                break

            output = msg["output"] # Get the output image data from the message
            save_path = msg["save_path"] # Get the path to save the image

            cv2.imwrite(save_path, output) # Save the image to disk

        print(f"IO worker {self.qid} is done.") # Indicate that the worker thread is finished