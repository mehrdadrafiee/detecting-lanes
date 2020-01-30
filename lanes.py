import cv2
import numpy

image = cv2.imread('test_image.jpg')
lane_image = numpy.copy(image)
gray_image = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0) # optional. since the canny function will already apply a 5x5 Gaussian
canny_image = cv2.Canny(blur_image, 50, 150)

cv2.imshow('result', canny_image)
cv2.waitKey(0)
