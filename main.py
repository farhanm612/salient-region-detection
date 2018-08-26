import cv2
import numpy as np
import math
import cut as cv

def euclideanDistance(v,quantLAB,size):
    sumDistance = 0.0
    for i in range(0,size[0]):
        for j in range(0,size[1]):

            sumDistance += math.sqrt(((float(quantLAB[i][j][0])-float(v[0]))**2)+
                                    ((float(quantLAB[i][j][1])-float(v[1]))**2)+
                                    ((float(quantLAB[i][j][2])-float(v[2]))**2))
    return sumDistance

def imageQuantization(path,levels):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    RGB = img.reshape((-1, 3))
    RGB = np.float32(RGB)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = levels
    ret, label, center = cv2.kmeans(RGB, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    quantRGB = res.reshape(img.shape)
    return quantRGB

def labColorDistance(image_LAB,size):
    dist_matrix = np.zeros(size)
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            dist_matrix[i][j] = euclideanDistance(image_LAB[i][j], image_LAB,size)
    return dist_matrix

if __name__ == "__main__":
    path = '101.jpg'

    quantRGB = imageQuantization(path,12)
    quantLAB = cv2.cvtColor(quantRGB, cv2.COLOR_BGR2Lab)
    # dist_matrix = labColorDistance(quantLAB,size)
    size = quantLAB.shape
    dist_matrix = cv2.blur(labColorDistance(quantLAB,size),(5,5))
    dst = np.zeros(shape=size)

    saliency = cv2.normalize(dist_matrix,dst, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # # print np.unique(norm_image)
    # img = Image.fromarray(norm_image)
    # print np.unique(img)

    cv2.imshow('image', cv.saliency)
    cv2.waitKey(0)
    cv2.destroyAllWindows()