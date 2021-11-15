import cv2
import numpy as np
from PIL import Image

img_cv = cv2.imread("G:\dataset/17386-12-34.png")
img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
img_i = Image.fromarray(img_cv)
img_i.save('1.png')

x1 = Image.open("G:\dataset/17386-12-34.png")
x2 = Image.open("1.png")
if x1.mode != 'RGB':
    x1 = x1.convert('RGB')
if x2.mode != 'RGB':
    x2 = x2.convert('RGB')
print(x1)
print(x2)