import cv2.cv2 as cv
import numpy as np
from scipy.sparse import coo
from src.preprocessing.bb_tools import bb_building, neighborhood_bb
from src.preprocessing.water_surface_detection import polygon_construction

class ColorBB():
    """
    This class is designed in order to make a color segmentation of an image. The segmentation can be done in
    2 color spaces : HSV and YUV. Some parameters are used outside the class in order to make a segmentation 
    on a video.

    Attributes
    ----------
    color_cvt : int
        Color spaces conversion
    lower : list
        the list of the 3 lower bounds for the segmentation
    upper : list
        the list of the 3 upper bounds for the segmentation
    margin : int
        the margin used to enlarge the box of the previous frame
    detect_surface : boolean
        if true, the method uses the detection of the surface
    adjust_pt1 : int
        decreasing the height of the left point for the detection of the surface
    adjust_pt2 : int
        increasing the height of the right point for the detection of the surface
    use_time : boolean
        if true, the method uses the precedent box to make the prediction
    Methods
    --------
    set_params(**params)
        Set the parameters of the estimator
    predict(filename_img, debug=False, a=0, b=0, precBB=[])
        Prediction of the Color Segmentation
    """

    def __init__(self, color_space="hsv", lower_hsv=(0, 0, 0), upper_hsv=(179, 235, 110), lower_yuv=(0, 0, 70),
                 upper_yuv=(90, 150, 255), margin=100, detect_surface=True, adjust_pt1=0, adjust_pt2=30, use_time=True):
        """
        Parameters
        -----------
        color_space : str
            the name of the color_space in lowercase ( default is "hsv" )
        lower_hsv : tuple
            the list of the lower bound values for the HSV colour mode ( default is (0, 0, 0) )
        upper_hsv : tuple
            the list of the upper bound values for the HSV colour mode ( default is (179, 235, 110) )
        lower_yuv : tuple
            the list of the lower bound values for the HSV colour mode ( default is (0, 0, 70) )
        upper_yuv : tuple
            the list of the upper bound values for the HSV colour mode ( default is (90, 150, 255) )
        margin : int
            the margin used to enlarge the box of the previous frame ( default is 100 )
        detect_surface : boolean
            if true, the method uses the detection of the surface ( default is True )
        adjust_pt1 : int
            decreasing the height of the left point ( default is 0 )
        adjust_pt2 : int
            increasing the height of the right point ( default is 30 )
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

        #Copy of the image for the processing
        img = img_ini.copy()

        #Search the swimmer in the neighborhood of the precedent bounding box
        img, coord = neighborhood_bb(img, self.margin, precBB)

        # Transformation of the colour mode of the image
        img_transform = cv.cvtColor(img, self.color_cvt)

        # Mask creation
        mask = cv.inRange(img_transform, self.lower, self.upper)

        if debug:
            cv.imshow('mask1', mask)
            cv.waitKey(1)

        # Texture/Non texture segmentation
        if a != 0 and b != 0:
            polygon = polygon_construction(a, b, coord)
            mask = cv.fillPoly(mask, [polygon], 0)

        if debug:
            cv.imshow('mask2', mask)
            cv.waitKey(1)

        # Building the best bounding box according to the mask
        if self.color_cvt == cv.COLOR_BGR2HSV:
            best_rectangle = bb_building(mask, 25)
        else:
            best_rectangle = bb_building(mask, 5)

        # Ajustement of the best box for the initial size of the image
        if len(best_rectangle)!=0 and coord[2] != 640 and coord[3] !=480:
            x_prec, y_prec, _, _ = coord
            best_rectangle[0] = best_rectangle[0] + x_prec
            best_rectangle[1] = best_rectangle[1] + y_prec

        if debug:
            if len(best_rectangle)!=0:
                x, y, w, h = best_rectangle
                cv.rectangle(img_ini, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.imshow('bounding box', img_ini)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return best_rectangle

