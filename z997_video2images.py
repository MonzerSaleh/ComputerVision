import os
import cv2
import numpy as np

video_name = "a0_video.mp4"
OUT_DIR = "./a0_frames"
SCALE_DOWN = 0.2
# A simple script to parse an mp4 video into a series of images named frame0000.png to whatever
# Save an mp4 video into this folder and put the name here


if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)


def video_frame_generator(filename):
    """
    Simple generator function that returns a frame on each 'next()' call.
    Return 'None' when there are no frames left.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)

    # Check if camera opened successfully
    if video.isOpened() == False:
        print("Error opening video stream or file")

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break
    video.release()

    yield None
def save_image(filename, image):
    """Convenient wrapper for writing images to the output directory."""
    im = cv2.resize(image, None, fx=SCALE_DOWN, fy=SCALE_DOWN)
    cv2.imwrite(os.path.join(OUT_DIR, filename), im)


video_gen = video_frame_generator(video_name)
# in order to write out the video
videoFrame = video_gen.__next__()
h, w, d = videoFrame.shape

frame_num = 0
while videoFrame is not None:
    fnum = f"{frame_num:05}"
    print("processing ", fnum)

    frame_name = "frame{}.png".format(fnum)
    save_image(frame_name, videoFrame)

    frame_num += 1
    videoFrame = video_gen.__next__()

# When everything done, release the capture
cv2.destroyAllWindows()
