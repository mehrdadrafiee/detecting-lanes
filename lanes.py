import cv2
import numpy

def canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0) # optional. since the canny function will already apply a 5x5 Gaussian
    canny_image = cv2.Canny(blur_image, 50, 150)
    return canny_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = numpy.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = numpy.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

image = cv2.imread('test_image.jpg')
lane_image = numpy.copy(image)
canny = canny(lane_image)
cropped_image = region_of_interest(canny)

cv2.imshow('result', region_of_interest(cropped_image))
cv2.waitKey(0)
