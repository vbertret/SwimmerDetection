import cv2.cv2 as cv
import numpy as np
import time
from src.preprocessing.bb_tools import union, intersection
from src.annotations.read_annotation import read_annotation
from src.preprocessing.water_surface_detection import surface_detection

class ColorBB():

    def __init__(self, color_space="hsv", lower_hsv=(0, 0, 0), upper_hsv=(179, 235, 120), lower_yuv=(0, 0, 70),
                 upper_yuv=(90, 150, 255), margin=100, detect_surface=True, adjust_pt1=0, adjust_pt2=30, use_time=True):
        """
        Parameters
        -----------
        filename_img : str
            the name of the image to use for the prediction
        color_space : str
            the name of the color_space in lowercase ( default is "hsv" )
        lower_hsv : tuple
            the list of the lower bound values for the HSV colour mode ( default is (0, 0, 0) )
        upper_hsv : tuple
            the list of the upper bound values for the HSV colour mode ( default is (179, 210, 110) )
        lower_yuv : tuple
            the list of the lower bound values for the HSV colour mode ( default is (0, 0, 0) )
        upper_yuv : tuple
            the list of the upper bound values for the HSV colour mode ( default is (179, 140, 255) )
        margin : int
            the margin used to enlarge the box of the previous frame ( default is 100 )
        detect_surface : boolean
            if true, the method uses the detection of the surface
        adjust_pt1 : int
            decreasing the height of the left point ( default is 0 )
        adjust_pt2 : int
            increasing the height of the right point ( default is 0 )
        use_time : boolean
            if true, the method uses the precedent box to make the prediction ( default is True )
        """

        # Different cases according to the color space
        if color_space == "hsv":
            # Definition of the convertor
            self.color_cvt = cv.COLOR_BGR2HSV

            # Thresholds for the upper and lower bounds for HSV
            self.lower = np.array(lower_hsv)
            self.upper = np.array(upper_hsv)
        elif color_space == "yuv":
            # Definition of the convertor
            self.color_cvt = cv.COLOR_BGR2YUV

            # Thresholds for the upper and lower bounds for YUV
            self.lower = np.array(lower_yuv)
            self.upper = np.array(upper_yuv)

        # Set the margin
        self.margin = margin

        # Set the detect surface
        self.detect_surface = detect_surface
        if self.detect_surface:
            self.adjust_pt1 = adjust_pt1
            self.adjust_pt2 = adjust_pt2

        # Set the time approach
        self.use_time = use_time

    def set_params(self, **params):
        """
        Set the parameters of the estimator

        Parameters
        ----------
        **params : dict
            Estimator parameters
        """

        if 'lower_hsv' in params:
            self.lower = np.array(params.get('lower_hsv'))

        if 'upper_hsv' in params:
            self.upper = np.array(params.get('upper_hsv'))

        if 'lower_yuv' in params:
            self.lower = np.array(params.get('lower_yuv'))

        if 'upper_yuv' in params:
            self.upper = np.array(params.get('upper_yuv'))

        if 'margin' in params:
            self.margin = params.get('margin')

        if 'detect_surface' in params:
            self.detect_surface = params.get('detect_surface')

        if 'adjust_pt1' in params:
            self.adjust_pt1 = params.get('adjust_pt1')

        if 'adjust_pt2' in params:
            self.adjust_pt2 = params.get('adjust_pt2')

        if 'use_time' in params:
            self.use_time = params.get('use_time')

    def predict(self, filename_img, debug=False, a=0, b=0, precBB=[]):
        """
        Prediction of the Color Segmentation

        This method makes the prediction of the bounding box for one image

        Parameters
        -----------
        filename_img : str
            the name of the image to use for the prediction
        debug : boolean
            If true, the method shows different step of the algorithm ( default is false )
        a : float
            if different from 0, it's the slope of the surface line ( default is 0 )
        b : float
            if different from 0, it's the intercept of the surface line ( default is 0 )
        precBB : [x, y, w, h] list
            the coordinates of the bounding box found at the previous frame ( default is [] )

        Returns
        -------
        best_rectangle : [x, y, w, h] list
            the coordinates of the bounding box
        """

        # Loading image
        img_ini = cv.imread(filename_img, cv.IMREAD_COLOR)

        if debug:
            cv.imshow("image", img_ini)
            cv.waitKey(1)

        # Adding median noises
        #img = cv.medianBlur(img_ini, 5)
        img = img_ini

        # Transformation of the colour mode of the image
        img_transform = cv.cvtColor(img, self.color_cvt)

        # Mask creation
        mask = cv.inRange(img_transform, self.lower, self.upper)

        if debug:
            cv.imshow('mask1', mask)
            cv.waitKey(1)

        #kernel = np.ones((5, 5), np.uint8)
        #mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        # Instead of searching the swimmer on all the images, the method searches
        # the swimmer on a box around the box found in the previous frame
        if len(precBB) != 0:
            x_prec, y_prec, w_prec, h_prec = precBB

            x_prec = max(x_prec - self.margin, 0)
            y_prec = max(y_prec - self.margin, 0)

            w_prec = min(w_prec + 2 * self.margin, 640 - x_prec)
            h_prec = min(h_prec + 2 * self.margin, 480 - y_prec)

            mask[:, 0:x_prec] = 0
            mask[0:y_prec, :] = 0
            mask[y_prec + h_prec:-1, :] = 0
            mask[:, x_prec + w_prec:-1] = 0

        # Texture/Non texture segmentation
        if a != 0 and b != 0:
            pt1 = (0, int(a * 0 + b))
            pt2 = (640, int(a * 640 + b))
            #cv.line(mask, pt1, pt2, 255, 3)
            pt3 = (640, 0)
            pt4 = (0, 0)
            polygon = np.array([pt1, pt2, pt3, pt4])
            mask = cv.fillPoly(mask, [polygon], 0)

        # A pousser, choisir les meilleurs filtres
        # kernel = np.ones((3, 3))
        #  = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        if debug:
            cv.imshow('mask2', mask)
            cv.waitKey(1)

        # Find contours
        contours, hierarchy = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2:]

        # Building the rectangle (only one)
        rects = []
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if cv.contourArea(cnt) > 800:
                find = False
                for i, (x_r, y_r, w_r, h_r) in enumerate(rects):
                    x_in, y_in, w_in, h_in = intersection([x_r - 20, y_r, w_r + 40, h_r], [x - 20, y, w + 40, h])
                    if w_in != 0 and h_in != 0:
                        new_rect = union([x_r, y_r, w_r, h_r], [x, y, w, h])
                        rects[i] = new_rect
                        find = True
                if not find:
                    rects.append([x, y, w, h])

        best_rectangle = []
        if len(rects) != 0:
            x, y, w, h = min(rects, key=lambda p: p[1])
            best_rectangle = [x, y, w, h]
            cv.rectangle(img_ini, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if debug:
            cv.imshow('bounding box', img_ini)
            cv.waitKey(0)
            cv.destroyAllWindows()

        #time.sleep(0.1)

        return best_rectangle

