import cv2.cv2 as cv
import numpy as np
import math
import time


def variance_initialization(filename, nb_frame):
    """
    Computation of map of variance

    Computation of black and white pixels in order to create a map
    according to the variance using several frames  collected during
    the initialization of the camera

    Parameters
    ----------
    filename : str
        the name of the filename containing the frames of the initialization
    nb_frame : int
        the number of available frames

    Returns
    -------
    mean : np.array (shape : (480, 640))
        mean value of all the frames
    variance : np.array (shape : (480, 640))
        white or black pixel for respectively high or small variance
    """
    # Load the pictures
    buf_img = np.zeros((nb_frame, 480, 640, 3))
    for nb in range(nb_frame):
        img = cv.imread(f"{filename}{str(nb).zfill(5)}.jpg", cv.IMREAD_COLOR)
        buf_img[nb] = img

    # Computation of the mean value between all the pictures for each pixel(x, y) and channel(RGB) and int8 conversion
    mean = np.mean(buf_img, axis=0)
    mean = np.uint8(mean)

    # Computation of the variance between all the picture for each pixel(x, y) and channel(RGB)
    variance = np.var(buf_img, axis=0)

    # Computation of the norm of the three channels RGB for each pixel(x, y) and int8 conversion
    variance = np.linalg.norm(variance, axis=2)
    variance = np.uint8(variance)

    # Use of a threshold to separate the variance values into 2 categories represented by black and white pixels
    variance = cv.medianBlur(variance, 25)
    ret, variance = cv.threshold(variance, 110, 255, cv.THRESH_BINARY)
    variance = cv.medianBlur(variance, 25)

    # n=14
    # kernel = np.ones((2*n+1, 2*n+1), np.uint8)
    # variance = cv.morphologyEx(variance, cv.MORPH_CLOSE, kernel)
    # n=21
    # kernel = np.ones((2*n+1, 2*n+1), np.uint8)
    # variance = cv.morphologyEx(variance, cv.MORPH_OPEN, kernel)

    # Return the mean and the 2 classes created according to the variance
    return mean, variance


def surface_detection(filename, nb_frame, debug=False, adjust_pt1=0, adjust_pt2=20):
    """
    Detection of the surface of the water

    Thanks to somes frames collected during the initialization, the method reaches to
    find the surface of the water by using the variance over the frames

    Parameters
    ----------
    filename : str
        the name of the filename containing the frames of the initialization
    nb_frame : int
        the number of available frames
    debug : boolean
        if true, display the best line founded ( default is false )
    adjust_pt1 : int
        decreasing the height of the left point ( default is 0 )
    adjust_pt2 : int
        increasing the height of the right point ( default is 0 )

    Returns
    -------
    a : float
        the slope of the line
    b : float
        intercept of the line
    """

    # Computation of the variance and the mean according to the method variance_initialization
    mean, variance = variance_initialization(filename, nb_frame)

    if debug:
        cv.imshow("variance", variance)
        cv.waitKey(1)

    # Declaration of the range values to use in order to find the best line
    y_height = np.arange(150, 400, 2)
    degrees = np.arange(-30, 30, 2)

    # Search of the best line by minimizing the heterogeneity of the two areas
    best_line = []
    min_het = 1000000000
    for y in y_height:
        for degree in degrees:

            # Definition of the coordinates of the line
            pt_left = (0, y)
            pt_right = (640, int(y - np.tan(degree*np.pi/360)*480))

            # Copy of the variance map for the two areas
            data_up = variance.copy()
            data_bot = variance.copy()

            # Construction of the other points
            pt3_up = (640, 0)
            pt4_up = (0, 0)
            pt3_bot = (640, 480)
            pt4_bot = (0, 480)

            # Area construction
            polygon_up = np.array([pt_left, pt_right, pt3_up, pt4_up])
            polygon_bot = np.array([pt_left, pt_right, pt3_bot, pt4_bot])

            # Metric computation
            # Upper area
            area_up = cv.contourArea(polygon_up)
            cv.fillPoly(data_up, [polygon_bot], 0)
            nb_white_pixel = np.sum(data_up)/255
            per_up = nb_white_pixel/area_up
            # Bottom area
            area_bot = cv.contourArea(polygon_bot)
            cv.fillPoly(data_bot, [polygon_up], 0)
            nb_white_pixel = np.sum(data_bot)/255
            per_bot = nb_white_pixel / area_bot

            # Heterogeneity computation
            # Gini concentration
            het = 2 * per_up * (1 - per_up) * area_up + 2 * per_bot * (1 - per_bot) * area_bot
            # Cross entropy
            # het = per_up*np.log(per_up) + (1-per_up)*np.log(1-per_up) + per_bot*np.log(per_bot) + (
            # 1-per_bot)*np.log(1-per_bot)

            # Update of the minimum
            if het < min_het:
                best_line = [pt_left, pt_right]
                min_het = het

    # Retrieval of the best line
    pt_left, pt_right = best_line

    # Height adjustment of the line
    pt_left = (pt_left[0], pt_left[1] - adjust_pt1)
    pt_right = (pt_right[0], pt_right[1] + adjust_pt2)

    # Line construction y = a*x + b
    a = (pt_right[1] - pt_left[1]) / (pt_right[0] - pt_left[0])
    b = pt_left[1] - a * pt_left[0]

    if debug:
        cv.line(mean, pt_left, pt_right, (0, 0, 255), 3)
        cv.putText(mean, "Water Surface", (30, int(a * 30 + b) - 5), 0, 0.5, (0, 0, 255), 2)
        cv.imshow("Water Surface Detector", mean)
        cv.waitKey(0)

    return a, b


if __name__ == '__main__':
    a, b = surface_detection("../../data/images/Valset/background/V2V4", 117, debug=True, adjust_pt1=0, adjust_pt2=30)
    print(a, b)
