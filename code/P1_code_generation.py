import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def get_data():

    color = (0, 0, 0)
    img = Image.new(mode="RGB", size=(150, 50), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, mode="RGB")
    font = ImageFont.truetype("arial.ttf", 28)
    label = ''

    # draw random text
    for i in range(5):
        # ascii 65 to 90, plus 0-9
        #char = random.choice([chr(random.randint(65, 90)), str(random.randint(0, 9))])
        char = str(random.randint(0, 9))
        label += char
        # generate random color
        #color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # Add text to image, width is 150, each character take up 30
        draw.text([i * random.randint(23,24), random.randint(-3,3)], char, color, font=font)

    # add random points
    for i in range(400):
        #color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.point([(random.randint(1,150), random.randint(1,150))], fill=color)

    # add random lines:
    for i in range(2):
        #color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.line([(random.randint(0,120), random.randint(0,30)), (random.randint(0,120), random.randint(0,30))], color)

    # transform
    param = [1-float(random.randint(1,2))/10, 0, 0, 0, 1-float(random.randint(1,2))/10,
             float(random.randint(1,2))/1000, 0.0001, float(random.randint(1,2)/1000)]
    img = img.transform((150,50), Image.PERSPECTIVE, param)
    # filter
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)

    return img, label


def save_img(img, label):
    # save image
    with open("{}.png".format(label), "wb") as f:
        img.save(f, format="png")

img, label = get_data()
save_img(img, label)