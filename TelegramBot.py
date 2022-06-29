#install pyTelegramBotAPI
#and import necessary libs for building the telegram StyleTransfer bot

import telebot
from pathlib import Path
import os
from torchvision.utils import save_image

TOKEN = '5451898321:AAFC3iv6WTHJ1GYzyvqRn9yG-PTvcYXIP1k'
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 'Send your image and then style photo that you want')

@bot.message_handler(content_types=['photo', 'text'])
def get_content_photo(message):
    """Gets user's content photo and saves it to gdrive"""
    file_path = bot.get_file(message.photo[-1].file_id).file_path
    file = bot.download_file(file_path)
    src = '/content/gdrive/MyDrive/' + file_path
    with open(src, 'wb') as f:
        f.write(file)
    bot.register_next_step_handler(message, get_style_photo) # jump to the next function

def get_style_photo(message):
    """Gets user's style photo, saves to gdrive and runs the style trasfer model"""
    file_path = bot.get_file(message.photo[-1].file_id).file_path
    file = bot.download_file(file_path)
    print(file_path)
    src = '/content/gdrive/MyDrive/' + file_path
    with open(src, 'wb') as f:
        f.write(file)
    dir = Path('/content/gdrive/MyDrive/photos') # directory where bot saves images from the user
    bot.send_message(message.chat.id, 'Magic is happening')
    style_img, content_img, loaded_images = content_and_style(dir) # load user's content and style images
    input_img = content_img.clone() # clone content image for training
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img) # run model for style transfer
    resize_output = resize(loaded_images[-2], output) # resize output image to its original size
    save_image(resize_output, '/content/gdrive/MyDrive/StyleTransfer/output_img.jpg') # choose folder where you save output image from torch tensor
    dir1 = Path('/content/gdrive/MyDrive/StyleTransfer/') # directory of output image
    for file in os.listdir(dir1)[1:]:
        bot.send_photo(message.chat.id,
                      photo=open('/content/gdrive/MyDrive/StyleTransfer' + '/' + file, 'rb')) # send output photo in telegram
    os.remove('/content/gdrive/MyDrive/StyleTransfer' + '/' + file) # remove output photo
    for file in os.listdir(dir)[1:]:
        os.remove('/content/gdrive/MyDrive/photos' + '/' + file) # remove content and style photos