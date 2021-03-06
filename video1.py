
import cv2
import numpy as np
import glob

img_array = []
for i in range(0,16):
    img = cv2.imread("images/img%i.1.png"%i)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('videos/video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
