import numpy as np
from PIL import Image
import cv2
from typing import Any

def preprocess_outpaint(direction: str, image: Image.Image):
    """
    Preprocesses an image for outpainting by adding padding based on the specified direction.

    Args:
        direction (str): The direction to outpaint ("top", "right", "bottom", or "left").
        image (Image.Image): The input PIL Image.

    Returns:
        tuple: A tuple containing the preprocessed image with padding and the corresponding inpaint mask.
    """
    top_pad = 0
    right_pad = 0
    bottom_pad = 0
    left_pad = 0
    out_percent = 0.2 # Percentage of the image dimensions to use for padding 

    image_ndarray = np.array(image) # Convert the PIL Image to a NumPy array
    height, width, _ = image_ndarray.shape # Get the height and width of the image

    # top
    if direction == "top":
        top_pad = int(height * out_percent) // 8 * 8 # Calculate the top padding, making it a multiple of 8
    # right
    elif direction == "right":
        right_pad = int(width * out_percent) // 8 * 8 # Calculate the right padding, making it a multiple of 8
    # bottom
    elif direction == "bottom":
        bottom_pad = int(height * out_percent) // 8 * 8 # Calculate the bottom padding, making it a multiple of 8
    # left
    elif direction == "left":
        left_pad = int(width * out_percent) // 8 * 8 # Calculate the left padding, making it a multiple of 8

    # Apply padding to the image
    image_ndarray = np.pad(
        image_ndarray,
        [
            [top_pad, bottom_pad], # Padding for the height (top, bottom)
            [left_pad, right_pad], # Padding for the width (left, right)
            [0, 0], # No padding for the channels
        ],
        mode="edge", # Padding mode: extend the edge values
        # mode="constant",
        # constant_values=255,
    )
    # Create an inpaint mask 
    inpaint_mask = np.zeros((height, width, 3), dtype=np.uint8) # Initially filled with zeros
    inpaint_mask = np.pad(
        inpaint_mask,
        [
            [top_pad, bottom_pad], # Padding for the height (top, bottom)
            [left_pad, right_pad], # Padding for the width (left, right)
            [0, 0], # No padding for the channels
        ],
        mode="constant", # Padding mode: use a constant value
        constant_values=255, # Pad with white (255)
    )

    image_ndarray = Image.fromarray(image_ndarray) # Convert the padded image back to a PIL Image
    
    # Apply the Canny edge detection and gradient to the mask
    inpaint_mask = outpaint_canny_gradient(
        inpaint_mask, top_pad, bottom_pad, left_pad, right_pad
    )
    # Code to save the padded image and mask (commented out)
    # if not os.path.exists("./static/test"):
    #     os.makedirs("./static/test", exist_ok=True)
    # inpaint_image.save("./static/test/outpaint_input.png")
    # inpaint_mask.save("./static/test/outpaint_mask.png")

    return image_ndarray, inpaint_mask  # Return the padded image and the mask


def gradient_dir(region: np.ndarray[Any, Any], dir):
    """
    Applies a gradient to the specified region of an image in the given direction.

    Args:
        region (np.ndarray[Any, Any]): A NumPy array representing the region of the image to apply the gradient.
        dir (str): The direction of the gradient ("left", "right", "top", or "bottom").

    Returns:
        np.ndarray[Any, Any]: The image region with the gradient applied.
    """
    h, w, _ = region.shape # Get the height, width, and number of channels of the region
    if dir == "left":
        # Gradient from white (255) on the right to black (0) on the left
        for x in range(w):
            region[:, x] = int(255 - 255 * (x / w)) 
    elif dir == "right":
        # Gradient from black (0) on the left to white (255) on the right
        for x in range(w):
            region[:, x] = int(255 * (x / w)) 
    elif dir == "top":
        # Gradient from white (255) on the bottom to black (0) on the top
        for y in range(h):
            region[y:,] = int(255 - 255 * (y / h))
    elif dir == "bottom":
        # Gradient from black (0) on the top to white (255) on the bottom
        for y in range(h):
            region[y:,] = int(255 * (y / h)) 
    return region # Return the region with the gradient


def outpaint_canny_gradient(
    image: Image.Image | np.ndarray,
    top_pad: int,
    bottom_pad: int,
    left_pad: int,
    right_pad: int,
):
    """
    Applies Canny edge detection and a gradient to the padded regions of an image.

    Args:
        image (Image.Image | np.ndarray): The input image, either as a PIL Image or a NumPy array.
        top_pad (int): The padding added to the top of the image.
        bottom_pad (int): The padding added to the bottom of the image.
        left_pad (int): The padding added to the left of the image.
        right_pad (int): The padding added to the right of the image.

    Returns:
        Image.Image: The processed image as a PIL Image.
    """
    if type(image) == Image.Image:
        img_ndata = np.array(image) # Convert the image to a NumPy array if it's a PIL Image
    else:
        img_ndata = image
    h, w, _ = img_ndata.shape  # Get the height, width, and number of channels of the image
    dist = 30  # The distance over which to apply the gradient

    # Apply the gradient based on the padding in each direction
    if top_pad > 0:
        region = img_ndata[top_pad - dist : top_pad + dist, :]
        img_ndata[top_pad - dist : top_pad + dist, :] = gradient_dir(region, "top") 
    if bottom_pad > 0:
        region = img_ndata[h - bottom_pad - dist : h - bottom_pad + dist, :]
        img_ndata[h - bottom_pad - dist : h - bottom_pad + dist, :] = gradient_dir(
            region, "bottom"
        ) 
    if left_pad > 0:
        region = img_ndata[:, left_pad - dist : left_pad + dist]
        img_ndata[:, left_pad - dist : left_pad + dist] = gradient_dir(region, "left") 
    if right_pad > 0:
        region = img_ndata[:, w - right_pad - dist : w - right_pad + dist]
        img_ndata[:, w - right_pad - dist : w - right_pad + dist] = gradient_dir(
            region, "right"
        )
    # This code is commented out, likely for applying a Gaussian blur to the padded region
    # top, bottom, left, right = inpaint_utils.detect_mask_valid_edge(img_ndata)
    # img_ndata = cv2.GaussianBlur(img_ndata[top:bottom, left:right], (5, 5), 0)
    return Image.fromarray(img_ndata) # Convert the modified image back to a PIL Image


def outpaint_canny_blur(
    image: Image.Image | np.ndarray,
    top_pad: int,
    bottom_pad: int,
    left_pad: int,
    right_pad: int,
):
    """
    Applies a Gaussian blur to the padded regions of the image.

    Args:
        image (Image.Image | np.ndarray): The input image, either as a PIL Image or a NumPy array.
        top_pad (int): The amount of padding at the top of the image.
        bottom_pad (int): The amount of padding at the bottom of the image.
        left_pad (int): The amount of padding at the left of the image.
        right_pad (int): The amount of padding at the right of the image.

    Returns:
        Image.Image: The image with Gaussian blur applied to the padded regions, returned as a PIL Image.
    """
    if type(image) == Image.Image:
        img_ndata = np.array(image) # Convert to NumPy array if it's a PIL Image
    else:
        img_ndata = image
    h, w, _ = img_ndata.shape

    # Apply Gaussian blur to each padded region
    if top_pad > 0:
        region = img_ndata[top_pad - 10 : top_pad + 10, :] # Select the top region
        img_ndata[top_pad - 10 : top_pad + 10, :] = cv2.GaussianBlur(region, (5, 5), 0) # Apply blur
    if bottom_pad > 0:
        region = img_ndata[h - bottom_pad - 10 : h - bottom_pad + 10, :] # Select the bottom region
        img_ndata[h - bottom_pad - 10 : h - bottom_pad + 10, :] = cv2.GaussianBlur(
            region, (5, 5), 0 
        )
    if left_pad > 0:
        region = img_ndata[:, left_pad - 10 : left_pad + 10] # Select the left region
        img_ndata[:, left_pad - 10 : left_pad + 10] = cv2.GaussianBlur(
            region, (5, 5), 0
        )
    if right_pad > 0:
        region = img_ndata[:, w - right_pad - 10 : w - right_pad + 10] # Select the right region
        img_ndata[:, w - right_pad - 10 : w - right_pad + 10] = cv2.GaussianBlur(
            region, (5, 5), 0
        )

    return Image.fromarray(img_ndata)  # Return the image as a PIL Image


def slice_by_direction(
    inpaint_image: Image.Image, mask_image: Image.Image, direction: int, max_size: int
):
    """
    Slices an image and a mask image based on a specified direction and maximum size.

    Args:
        inpaint_image (Image.Image): The image to be sliced.
        mask_image (Image.Image): The mask image to be sliced.
        direction (int): An integer representing the direction of the slice:
                         1: Up, 2: Right, 4: Down, 8: Left.
                         Combinations can be used (e.g., 3 for up and right).
        max_size (int): The maximum allowed size for the sliced region.

    Returns:
        tuple: A tuple containing:
            - Image.Image: The sliced image.
            - Image.Image: The sliced mask image.
            - tuple: The coordinates (top, right, bottom, left) of the slice box. 
    """
    top = 0
    right = 0
    bottom = 0
    left = 0

    # up
    if direction & 1 == 1:
        # Slice from the top 
        right = inpaint_image.width
        bottom = min(inpaint_image.height, max_size)
    # right
    if direction & 2 == 2:
        # Slice from the right
        left = max(inpaint_image.width - max_size, 0)
        right = inpaint_image.width
        bottom = inpaint_image.height
    # bottom
    if direction & 4 == 4:
        # Slice from the bottom
        top = max(inpaint_image.height - max_size, 0) 
        right = inpaint_image.width
        bottom = inpaint_image.height
    # left
    if direction & 8 == 8:
        # Slice from the left
        right = min(inpaint_image.width, max_size)
        bottom = inpaint_image.height

    # Crop the image and the mask based on the calculated boundaries
    return (
        inpaint_image.crop((top, right, bottom, left)),
        mask_image.crop((top, right, bottom, left)),
        (top, right, bottom, left),
    )