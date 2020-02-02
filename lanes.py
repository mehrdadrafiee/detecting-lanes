import cv2
import numpy

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return numpy.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
         x1, y1, x2, y2 = line.reshape(4)
         parameters = numpy.polyfit((x1, x2), (y1, y2), 1)
         slope = parameters[0]
         intercept = parameters[1]
         if slope < 0:
             left_fit.append((slope, intercept))
         else:
             right_fit.append((slope, intercept))
    left_fit_average = numpy.average(left_fit, axis=0)
    right_fit_average = numpy.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return numpy.array([left_line, right_line])

def canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0) # optional. since the canny function will already apply a 5x5 Gaussian
    canny_image = cv2.Canny(blur_image, 50, 150)
    return canny_image

def display_lines(image, lines):
    line_image = numpy.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = numpy.array([
    [(200, height), (1100, height), (550, 250)]
    ])
    mask = numpy.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# uncomment this section to detect lanes in the image

# image = cv2.imread('test_image.jpg')
# lane_image = numpy.copy(image)
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, numpy.pi/180, 100, numpy.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combined_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

# cv2.imshow('result', region_of_interest(combined_image)) # something is wrong with this line

# cv2.imshow('result', combined_image)
# cv2.waitKey(0) == ord('q')

video_capture = cv2.VideoCapture('test2.mp4')

while(video_capture.isOpened()):
    _, current_video_frame = video_capture.read()
    canny_image = canny(current_video_frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, numpy.pi/180, 100, numpy.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(current_video_frame, lines)
    line_image = display_lines(current_video_frame, averaged_lines)
    combined_image = cv2.addWeighted(current_video_frame, 0.8, line_image, 1, 1)
    # cv2.imshow('result', region_of_interest(combined_image))
    cv2.imshow('result', combined_image)
    if cv2.waitKey(1) == ord('q'):
        break

current_video_frame.release()
cv2.destroyAllWindows()