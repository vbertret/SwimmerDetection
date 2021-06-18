import cv2.cv2
import numpy as np
import cv2.cv2 as cv
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing.create_dataframe import createDataframe
from src.annotations.read_annotation import read_annotation
from src.preprocessing.water_surface_detection import surface_detection
from src.preprocessing.bb_tools import union, intersection
import pickle
import time

class RandomForestBB:
    """
    A class used to construct a RandomForestModel

    Attributes
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        the RandomForest model

    Methods
    -------
    predict(filename_img, threshold=0.5, debug=False)
        Return the prediction made by the model for one image

    """

    def __init__(self, filename=None, n_estimators=100, margin=100, threshold=0.5,  detect_surface=True, adjust_pt1=15, adjust_pt2=15, use_time=True):
        """
        Parameters
        -----------
        filename : str
            the name of the model (default is None)
        n_estimators : int
            the number of trees in the forest
        margin : int
            the margin used to enlarge the box of the previous frame ( default is 100 )
        threshold : float
            the threshold in order to convert probabilities in (0,1) values ( default is 0.5 )
        detect_surface : boolean
            if true, the method uses the detection of the surface
        adjust_pt1 : int
            decreasing the height of the left point ( default is 0 )
        adjust_pt2 : int
            increasing the height of the right point ( default is 0 )
        use_time : boolean
            if true, the method uses the precedent box to make the prediction ( default is True )
        """
        if filename is None:
            self.model = RandomForestClassifier(n_estimators=n_estimators)
        else:
            self.model = pickle.load(open(filename, 'rb'))

        # Set the margin and the threshold
        self.margin = margin
        self.threshold = threshold

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

        if 'n_estimators' in params:
            self.model = RandomForestClassifier(n_estimators=params.get('n_estimators'))

        if 'margin' in params:
            self.margin = params.get('margin')

        if 'threshold' in params:
            self.threshold = params.get('threshold')

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
        Prediction of the Random Forest

        This method makes the prediction of the bounding box for one image

        Parameters
        -----------
        filename_img : str
            the name of the image to use for the prediction
        debug : boolean
            If true, the method shows different step of the algorithm
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
        img = cv.imread(filename_img, cv.IMREAD_COLOR)

        if debug:
            cv.imshow("image", img)
            cv.waitKey(1)

        if len(precBB)!=0:
            x_prec, y_prec, w_prec, h_prec = precBB

            x_prec = max(x_prec - self.margin, 0)
            y_prec = max(y_prec - self.margin, 0)

            w_prec = min(w_prec + 2*self.margin, 640 - x_prec)
            h_prec = min(h_prec + 2*self.margin, 480 - y_prec)

            img = img[y_prec:(y_prec + h_prec), x_prec:(x_prec + w_prec), :]

        # Creation dataframe
        X = createDataframe(img)

        # Compute prediction
        prediction_RF = self.model.predict_proba(X)[:, 1]

        # Building the mask
        if len(precBB)!=0:
            score_img = prediction_RF.reshape((h_prec, w_prec))
        else:
            score_img = prediction_RF.reshape((480, 640))
        mask = np.array((score_img > self.threshold) * 255, dtype=np.uint8)

        if a!=0 and b!=0:

            if len(precBB)!=0:
                pt1 = (0, int(a * x_prec + b) - y_prec)
                pt2 = (w_prec, int(a * (x_prec + w_prec) + b) - y_prec)
                pt3 = (w_prec, 0)
                pt4 = (0, 0)
            else:
                pt1 = (0, int(a*0+b))
                pt2 = (640, int(a*640+b))
                pt3 = (640, 0)
                pt4 = (0, 0)

            polygon = np.array([pt1, pt2, pt3, pt4])
            mask = cv.fillPoly(mask, [polygon], 0)

        if debug:
            cv.imshow('mask', mask)
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
            x, y, w, h = max(rects, key=lambda p: p[1])
            best_rectangle = [x, y, w, h]

            if len(precBB)!=0:
                best_rectangle[0] = best_rectangle[0] + x_prec
                best_rectangle[1] = best_rectangle[1] + y_prec

            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if debug:
            cv.imshow('bounding box', img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return best_rectangle

