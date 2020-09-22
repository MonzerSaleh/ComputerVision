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

img = loadImage('z9999_TorontoTrafficCongestion.jpg',0)

# Print the shape
# The image shape = (Height > y-axis, Width > x-axis, Channels > z-axis)
print(img.shape)

# Display the img in a window named "Example"
cv2.imshow('Example',img)
cv2.imwrite('zz_Example.jpg',img)
# create a keyboard binding event listener for a specified time 
# the 0 indicates -> wait indefinitely
##### code execution stops here until a keyboard signal is received
cv2.waitKey(0)
# destroy the window created above
cv2.destroyAllWindows()

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()