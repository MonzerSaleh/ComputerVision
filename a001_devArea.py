from matplotlib import pyplot as plt
import numpy as np
import cv2
import os

src_images = "images/input"
out_images = "images/output"

def loadImage(imgName,method=1):
    """
    basic image loading 
    An image can be loaded in 3 ways
    cv2.IMREAD_COLOR(1)       Loads a color image. Transparency is ignored  - default
    cv2.IMREAD_GRAYSCALE(0)   Loads image in grayscale mode
    cv2.IMREAD_UNCHANGED(-1)  Loads image as such including alpha channel
    """
    imPath = os.path.join(src_images,imgName)
    imTemp = cv2.imread(imPath,method)
    return imTemp