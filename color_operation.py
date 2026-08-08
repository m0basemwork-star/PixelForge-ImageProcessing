import numpy as np

def changing_the_image_lighting_color(image, channel, C):
    new_image = image.copy()
    new_image[:, :, channel] = np.clip(image[:, :, channel] + C, 0, 255)
    return new_image

def swapping_image_channels(image, channel1, channel2):
    new_image = image.copy()
    new_image[:, :, [channel1, channel2]] = new_image[:, :, [channel2, channel1]]
    return new_image

def eliminating_color_channels(image, channel):
    new_image = image.copy()
    new_image[:, :, channel] = 0
    return new_image