import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from skimage import color
import scipy
from PIL import Image
from scipy.stats import itemfreq
import cv2

def getPixelVal(img,center,x,y):
    binary_value = 0
    if img[y,x] >= center:
        binary_value = 1

    return binary_value

def LBPValue(img,x,y):
    #using a square neighborhood

    current_center = img[y,x]
    values = []
    #weights occur in a clockwise fashion
    #top_left
    values.append(getPixelVal(img,current_center,x-1,y-1))
    #top
    values.append(getPixelVal(img,current_center,x-1,y))
    #top_right
    values.append((getPixelVal(img,current_center,x-1,y+1)))
    #right
    values.append(getPixelVal(img,current_center,x,y+1))
    # bottom_right
    values.append(getPixelVal(img, current_center, x + 1, y + 1))
    # bottom
    values.append(getPixelVal(img, current_center, x + 1, y))
    #bottom_left
    values.append(getPixelVal(img,current_center,x+1,y-1))
    # left
    values.append(getPixelVal(img, current_center, x, y - 1))
    lbp_value = 0
    for i in range(len(values)):
        lbp_value += values[i] * 2**i
    return lbp_value

def getImageLBP(im):
    img = plt.imread(im)
    img = color.gray2rgb(img)
    gray_img = color.rgb2gray(img)
    img_lbp = np.zeros((img.shape[0]-1,img.shape[1]-1,3))
    for y in range(1,img.shape[0]-1):
        for x in range(1,img.shape[1]-1):
            img_lbp[y,x] = LBPValue(gray_img,x,y)

    return img_lbp

    #show histogram
def getHistogram(img_lbp):
        n_bins = int(img_lbp.max() + 1)
        hist,_ = np.histogram(img_lbp, density=True, bins=n_bins, range=(0, n_bins))
        plt.hist(img_lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins))
        plt.xlabel('LBP Value')
        plt.ylabel('Percentage ')
        plt.title('Histogram of Pearl Sugar')
        plt.show()
        return hist

def getFeature(im):
    img_lbp = getImageLBP(im)
    implot = plt.imshow(img_lbp, cmap='gray')
    plt.title('LBP Representation of Pearl Sugar')
    plt.show()
    return getHistogram(img_lbp)



if __name__ == '__main__':
    getFeature('pearlsugar1-a-p001.png')
