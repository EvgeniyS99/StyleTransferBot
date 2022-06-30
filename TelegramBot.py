import telebot
from pathlib import Path
import os
import torch
import torchvision.models as models
from torchvision.utils import save_image
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

cnn = models.vgg19(pretrained=True).features.to(device).eval()

# functions for processing images
def loader_images(dir):
    """Open image as PIL Image and append it in the list of loaded images"""
    loaded_images = []
    for file in os.listdir(dir)[1:]:
        image = Image.open('./MyDrive/photos' + '/' + file)
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

# Telegram Bot
TOKEN = '' # your TOKEN recieved from BotFather
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 'Send your image and then style photo that you want')

@bot.message_handler(content_types=['photo', 'text'])
def get_content_photo(message):
    """Gets user's content photo and saves it to gdrive"""
    file_path = bot.get_file(message.photo[-1].file_id).file_path
    file = bot.download_file(file_path)
    # file path looks like 'photos/file_No.jpg'
    # so you need to create 'photos' folder on your gdrive
    src = './MyDrive/' + file_path
    with open(src, 'wb') as f:
        f.write(file)
    bot.register_next_step_handler(message, get_style_photo) # jump to the next function

def get_style_photo(message):
    """Gets user's style photo, saves to gdrive and runs the style trasfer model"""
    file_path = bot.get_file(message.photo[-1].file_id).file_path
    file = bot.download_file(file_path)
    src = './MyDrive/' + file_path
    with open(src, 'wb') as f:
        f.write(file)
    dir = Path('./MyDrive/photos') # directory where bot saves images from the user
    bot.send_message(message.chat.id, 'Magic is happening')
    style_img, content_img, loaded_images = content_and_style(dir) # load user's content and style images
    input_img = content_img.clone() # clone content image for training
    style_transfer = Style_Transfer(cnn, content_img, style_img, input_img)
    output = style_transfer.run_style_transfer()
    resize_output = resize(loaded_images[-2], output) # resize output image to its original size
    # create folder where you want to save your output image
    save_image(resize_output, './output_img.jpg')
    dir1 = Path('./MyDrive/StyleTransfer/') # directory of output image
    for file in os.listdir(dir1)[1:]:
        bot.send_photo(message.chat.id,
                      photo=open('./MyDrive/StyleTransfer' + '/' + file, 'rb')) # send output photo in telegram
    os.remove('./MyDrive/StyleTransfer' + '/' + file) # remove output photo
    for file in os.listdir(dir)[1:]:
        os.remove('./MyDrive/photos' + '/' + file) # remove content and style photos

