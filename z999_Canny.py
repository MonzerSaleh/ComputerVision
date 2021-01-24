import cv2 as cv
import argparse

window_name = 'Edge Map'
lowThreshold = 100
hghThreshold = 500
title_trackbar1 = 'Min/Low Threshold:'
title_trackbar2 = 'Max/High Threshold:'

ratio = 3
kernel_size = 3
img = 'images/ex000_lena.png'

def CannyThreshold(val):
    low_threshold = cv.getTrackbarPos(title_trackbar1, window_name)
    hgh_threshold = cv.getTrackbarPos(title_trackbar2, window_name)
    img_blur = cv.blur(src_gray, (3,3))
    detected_edges = cv.Canny(img_blur, low_threshold, hgh_threshold, kernel_size)
    #mask = detected_edges != 0
    #dst = src * (mask[:,:,None].astype(src.dtype))
    cv.imshow(window_name, detected_edges )
    
parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
parser.add_argument('--input', help='Path to input image.', default='fruits.jpg')
args = parser.parse_args()
src = cv.imread(img)

if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
    
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.namedWindow(window_name)
cv.createTrackbar(title_trackbar1, window_name , 0, lowThreshold, CannyThreshold)
cv.createTrackbar(title_trackbar2, window_name , 0, hghThreshold, CannyThreshold)

CannyThreshold(0)
cv.waitKey()