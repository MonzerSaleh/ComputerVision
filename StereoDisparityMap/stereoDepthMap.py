
import cv2
import os
import numpy as np

INPUT_DIR = "input_images"

#####################################################
# Utility function
def get_image_pair(name):
    """
        simply retreives a pair of images file
        im{0,1}.png                  -- default left and right view
        Args: 
            name (string): that correlates to the image pair prefix
            
        Returns:
    """
    left = os.path.join(INPUT_DIR, name + "_im0.png")
    print(left)
    right = os.path.join(INPUT_DIR, name + "_im1.png")
    
    img_lft = cv2.resize(cv2.imread(left), None, fx=0.3, fy=0.3)
    img_rht = cv2.resize(cv2.imread(right), None, fx=0.3, fy=0.3)

    return img_lft, img_rht

def to_greyscale(img):
    """
    Converts image to greyscale
    Args:
        img (np.ndarray)
    Returns
        img_grey (np.ndarray): with only 2 channels
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def normalize_and_scale(img_in):
    return cv2.normalize(
        img_in, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

#####################################################
# 2.2 Energy Function
# Eq1
# E(f) = E_data(f)+E_occlusions(f)+E_smoothness+E_uniqueness(f)
# Eq2
# E_data(f) = SUM_over{a,f(a)=1} D(a) = sum D(a)* 1(f(a)==1)
#           where 1(*) = 1 when * is true, = 0 otherwise
#           and D(a) is either absolute/squared difference
def absolute_difference(im1, im2, p, q):
    return abs(im1[p]-im2[q])

# Eq3
def squared_difference(im1, im2, p, q):
    return (im1[p]-im2[q])**2

# Eq5
def trim_difference(x,thresh=30):
    return np.min(x,thresh)  
#####################################################
# Occlusion Term
#####################################################

# https://sites.google.com/site/5kk73gpu2010/assignments/stereo-vision#TOC-Update-Disparity-Map
def disparity_v2(left_gry, right_gry, k_size, MAX_SHIFT):
    """ Create a "minSSD" array equal to the size of the image, with large initial values.
        Create a "disparity" array equal to the size of the image.
        for k = 0 to MAX_SHIFT do
            Step 1: Shift right image to the right by k pixels
            Step 2: Perform Sum of Squared Differences (SSD) between left image and shifted right image
            Step 3: Update the minSSD and disparity array.
                    for each pixel coordinate (i,j) do
                        if ssd(i,j) < minSSD(i,j) do
                            minSSD(i,j) <= ssd(i,j)
                            disparity(i,j) <= k
                        end
                    end
        end
    """
    w_size = 2 * k_size + 1
    rows, cols = left_gry.shape

    kernel = np.ones([w_size, w_size]) / w_size
    disparity_maps = np.zeros((rows, cols, MAX_SHIFT))

    for d in range(0, MAX_SHIFT):
        # Shift right image to the right k-pixels
        # https://stackoverflow.com/questions/19068085/shift-image-content-with-opencv
        affine_mtx = np.array([[1, 0, d], [0, 1, 0]], dtype=np.float)
        rimg_shifted = cv2.warpAffine(right_gry, affine_mtx, (cols, rows))
        # Perform Sum of Squared Differences (SSD) between left image and shifted right image
        square_Diff = (left_gry.astype(np.float) - rimg_shifted.astype(np.float)) ** 2
        sum_square_Diff = cv2.filter2D(square_Diff, -1, kernel)
        disparity_maps[:, :, d] = sum_square_Diff

    disparity = np.argmin(disparity_maps, axis=2)
    # disparity = np.uint8(disparity * 255 / MAX_SHIFT)
    # disparity = cv2.equalizeHist(disparity)

    return disparity

def get_disparity_image(left, right):
    # https://stackoverflow.com/questions/59478962/how-to-convert-a-grayscale-image-to-heatmap-image-with-python-opencv
    disp_img2 = disparity_v2(left, right, k_size=7, MAX_SHIFT=120)
    heatmap2 = cv2.applyColorMap(normalize_and_scale(disp_img2), cv2.COLORMAP_JET)
    cv2.imwrite("ps1-1-a-1.png", heatmap2)

    disp_img2 = disparity_v2(left, right, k_size=7, MAX_SHIFT=110)
    heatmap2 = cv2.applyColorMap(normalize_and_scale(disp_img2), cv2.COLORMAP_JET)
    cv2.imwrite("ps1-1-a-2.png", heatmap2)

    disp_img2 = disparity_v2(left, right, k_size=7, MAX_SHIFT=100)
    heatmap2 = cv2.applyColorMap(normalize_and_scale(disp_img2), cv2.COLORMAP_JET)
    cv2.imwrite("ps1-1-a-3.png", heatmap2)
    
if __name__ == "__main__":
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # # step 1 - we get the images
    left_im, right_im = get_image_pair("art")
    # # step 2 - convert to greyscale (float)
    left_gry = to_greyscale(left_im)
    right_gry = to_greyscale(right_im)
    # # step 3 - compute sum of square differences
    # #   3a along x axis=0
    # #   3a along y axis=1
    # # l = cv2.medianBlur(left_gry, 7)
    # # r = cv2.medianBlur(right_gry, 7)

    get_disparity_image(left_gry, right_gry)

    # disp_img3 = cv2.equalizeHist(disp_img2)
    # # cv2.imshow("disparity Image C equalized", disp_img3.astype(np.uint8))
    # heatmap = cv2.applyColorMap(fps.normalize_and_scale(disp_img3), cv2.COLORMAP_JET)
    # cv2.imshow("Image C equalized Heatmap", heatmap.astype(np.uint8))

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()