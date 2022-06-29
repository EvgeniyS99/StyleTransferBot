#import necessary libs for image processing

import torch
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def loader_images(dir):
    """Open image as PIL Image and append it in the list of loaded images"""
    loaded_images = []
    for file in os.listdir(dir)[1:]:
        image = Image.open('/content/gdrive/MyDrive/photos' + '/' + file)
        loaded_images.append(image)
    return loaded_images

def image_loader(image):
    """Transforms images to tensors"""
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    image = loader(image).unsqueeze(0)  # add fake batch dimension required to fit network's input dimensions
    return image.to(device, torch.float)

def resize(prior_image, output):
    """Resizes ouput images images to their original size"""
    arr = np.array(prior_image)
    H, W = arr.shape[0], arr.shape[1]
    resize = transforms.Resize((H, W))
    resize_output = resize(output)
    return resize_output

def content_and_style(dir):
    """Loads content and style images"""
    loaded_images = loader_images(dir)
    style_img = image_loader(loaded_images[-1])
    content_img = image_loader(loaded_images[-2])
    return style_img, content_img, loaded_images