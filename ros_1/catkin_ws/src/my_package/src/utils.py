#!/usr/bin/env python
import cv2 as cv
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor


def image_file_to_tensor(path_to_file, precision=torch.float32, device='cpu'):
    # Read the image
    image = cv.imread(path_to_file)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # Define a transform to convert the image to tensor
    tensor = to_tensor(image).unsqueeze(0)
    # Convert the image to PyTorch tensor according to requirements
    return tensor.to(precision).to(device)


def image2array(image, image_type='rgb'):
    """
    Method to convert image message to Numpy RGB array or RGBA
    """
    image_data = list(image.data)
    image_array = np.array(image_data).astype('uint8')

    if image_type == 'rgba':
        channels = 4
    else:
        channels = 3
    
    reshaped_image = image_array.reshape(image.height, image.width, channels)
    
    if image_type == 'rgba':
        output = cv.cvtColor(reshaped_image, cv.COLOR_BGRA2RGBA)
    else:
        output = cv.cvtColor(reshaped_image, cv.COLOR_BGR2RGB)
    
    return output
