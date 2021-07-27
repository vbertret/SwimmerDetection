from src.color_segmentation import ColorBB
from src.gaussian_mixture import GaussianMixtureBB
from src.deep_learning import Swimnet
from src.metrics.model_performance import IoU, score
from src.preprocessing.bb_tools import union, intersection
import torch
import numpy as np
import cv2.cv2 as cv


class CombiningBB:
    """
    Class that combines different models

    The class combines the HSV, YUV color segmentation, the GMM model and the Deep Learning model

    Attributes
    ----------
    hsv : src.color_segmentation.ColorBB
        the HSV segmentation model
    yuv : src.color_segmentation.ColorBB
        the YUV segmentation model
    gmm : src.gaussian_mixture.GaussianMixtureBB
        the GMM model
    deep : src.deep_learning.Swimnet
        the Deep learning model
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
    -------
    predict(filename_img, debug=False, a=0, b=0, precBB=[]):
        make the prediction of the bounding box of the combining class
    """

    def __init__(self, margin=100, detect_surface=True, adjust_pt1=0, adjust_pt2=30, use_time=True):
        """
        Parameters
        ----------
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
        """

        self.hsv = ColorBB("hsv", margin=margin, detect_surface=detect_surface, adjust_pt1=adjust_pt1, adjust_pt2=adjust_pt2, use_time=use_time)
        self.yuv = ColorBB("yuv", margin=margin, detect_surface=detect_surface, adjust_pt1=adjust_pt1, adjust_pt2=adjust_pt2, use_time=use_time)

        self.gmm = GaussianMixtureBB("../models/GMM_model_test_vid_23_full", threshold=0.01, margin=margin, detect_surface=detect_surface, adjust_pt1=adjust_pt1, adjust_pt2=adjust_pt2, use_time=True)

        PATH = "../models/dataaug6_real_end"
        self.deep = Swimnet("mobilenet-v3-small")
        self.deep.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

        # Set the margin
        self.margin = margin

        # Set the detect surface
        self.detect_surface = detect_surface
        if self.detect_surface:
            self.adjust_pt1 = adjust_pt1
            self.adjust_pt2 = adjust_pt2

        # Set the time approach
        self.use_time = use_time

    def predict(self, filename_img, debug=False, a=0, b=0, precBB=[]):
        """
        Prediction of the Combining Methods

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

        # Make the prediction for all the models
        bb_hsv = self.hsv.predict(filename_img, a=a, b=b, precBB=precBB)
        bb_yuv = self.yuv.predict(filename_img, a=a, b=b, precBB=precBB)
        bb_gmm = self.gmm.predict(filename_img, a=a, b=b, precBB=precBB)
        bb_deep = self.deep.predict(filename_img, a=a, b=b, precBB=precBB)

        # Delete the prediction with no boxes
        list_bb = [bb_hsv, bb_yuv, bb_gmm, bb_deep]
        while [] in list_bb:
            i = list_bb.index([])
            list_bb.remove([])
            weights.pop(i)

        # Choose a strategy among the methods below
        best_rectangle = best_iou(list_bb)

        return best_rectangle


def mean_bb(list_bb, weights):
    """
    This method returns the mean of all the bounding box in list_bb

    Parameters
    ----------
    list_bb : list of list
        list of predicted bounding boxes
    weights : list
        list of the weights for the mean

    Return
    ------
    the bounding box which is the mean of all the bounding box
    """

    # Compute the number of predicted boxes and the sum of the weights
    n = len(list_bb)
    total_weights = np.sum(weights)

    # if there are more than 0 predicted boxes, compute the mean for each coordinates of the bounding box
    if n != 0:
        mean_x = 0
        mean_y = 0
        mean_w = 0
        mean_h = 0
        for i in range(n):
            mean_x += weights[i]*list_bb[i][0]
            mean_y += weights[i]*list_bb[i][1]
            mean_w += weights[i]*list_bb[i][2]
            mean_h += weights[i]*list_bb[i][3]

        return [int(mean_x/total_weights), int(mean_y/total_weights), int(mean_w/total_weights), int(mean_h/total_weights)]
    else:
        return []


def best_loss(list_bb):
    """
    This method returns the union of the two boxes which have the smallest L2 Loss between them.

    Parameters
    ----------
    list_bb : list of list
        list of predicted bounding boxes

    Return
    ------
    the union between the 2 bounding box
    """

    # Compute the number of predicted boxes
    n = len(list_bb)

    # if there are more than 0 predicted boxes, search for the 2 boxes
    if n != 0:
        tab_loss = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                loss_val = np.sum([(list_bb[i][k]-list_bb[j][k]) ** 2 for k in range(4)])
                tab_loss[i, j] = loss_val
                tab_loss[j, i] = loss_val

        # Find the minimum
        amin = np.unravel_index(tab_loss.argmin(), tab_loss.shape)

        return union(list_bb[amin[0]], list_bb[amin[1]])
    else:
        return []


def best_score(list_bb):
    """
    This method returns the union of the two boxes which have the largest score between them.

    Parameters
    ----------
    list_bb : list of list
        list of predicted bounding boxes

    Return
    ------
    the union between the 2 bounding box
    """

    # Compute the number of predicted boxes
    n = len(list_bb)

    # if there are more than 0 predicted boxes, search for the 2 boxes
    if n != 0:
        tab_score = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                score_val = score(list_bb[i], list_bb[j])
                tab_score[i, j] = score_val

        # Find the maximum
        amax = np.unravel_index(tab_score.argmax(), tab_score.shape)

        return union(list_bb[amax[0]], list_bb[amax[1]])
    else:
        return []


def best_iou(list_bb):
    """
    This method returns the union of the two boxes which have the largest IoU between them.

    Parameters
    ----------
    list_bb : list of list
        list of predicted bounding boxes

    Return
    ------
    the union between the 2 bounding box
    """

    # Compute the number of predicted boxes
    n = len(list_bb)

    # if there are more than 0 predicted boxes, search for the 2 boxes
    if n != 0:
        tab_IoU = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                iou_val = IoU(list_bb[i], list_bb[j])
                tab_IoU[i, j] = iou_val
                tab_IoU[j, i] = iou_val

        # Find the maximum
        amax = np.unravel_index(tab_IoU.argmax(), tab_IoU.shape)

        return union(list_bb[amax[0]], list_bb[amax[1]])
    else:
        return []


def best_iou2(list_bb):
    """
    This method returns the intersection of the two boxes which have the largest IoU between them.

    Parameters
    ----------
    list_bb : list of list
        list of predicted bounding boxes

    Return
    ------
    the intersection between the 2 bounding box
    """

    # Compute the number of predicted boxes
    n = len(list_bb)

    # if there are more than 0 predicted boxes, search for the 2 boxes
    if n != 0:
        tab_IoU = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                iou_val = IoU(list_bb[i], list_bb[j])
                tab_IoU[i, j] = iou_val
                tab_IoU[j, i] = iou_val

        # Find the maximum
        amax = np.unravel_index(tab_IoU.argmax(), tab_IoU.shape)

        return intersection(list_bb[amax[0]], list_bb[amax[1]])
    else:
        return []


def deep_iou(list_bb, threshold):
    """
    This method returns the union of the boxes which have an IoU larger than the threshold with the box of the Deep Learning

    Parameters
    ----------
    list_bb : list of list
        list of predicted bounding boxes
    threshold : float
        the threshold for the IoUs

    Return
    ------
    the union between the valid bounding boxes and the box of the Deep Learning model
    """

    # Compute the number of predicted boxes
    n = len(list_bb)

    # if there are more than 0 predicted boxes, search for the valid boxes
    if n != 0:
        tab_IoU = np.zeros((n-1))
        for i in range(n-1):
            iou_val = IoU(list_bb[i], list_bb[n-1])
            tab_IoU[i] = iou_val

        # Take only the box with an IoU larger than the threshold
        valid_bb = [i > threshold for i in tab_IoU]

        # Make the union
        best_rectangle = list_bb[-1]
        for i, ind in enumerate(valid_bb):
            if ind:
                best_rectangle = union(best_rectangle, list_bb[i])

        return best_rectangle
    else:
        return []


def threshold_iou(list_bb, threshold):
    """
    This method returns the union of the boxes which have an IoU larger than the threshold

    Parameters
    ----------
    list_bb : list of list
        list of predicted bounding boxes
    threshold : float
        the threshold for the IoUs

    Return
    ------
    best rectangle : list
        the union between the valid bounding boxes
    """

    # Compute the number of predicted boxes
    n = len(list_bb)

    # if there are more than 0 predicted boxes, search for the 2 boxes
    if n != 0:

        # Find the valid bounding boxes
        good_bb = set()
        for i in range(n):
            for j in range(i + 1, n):
                iou_val = IoU(list_bb[i], list_bb[j])
                if iou_val > threshold:
                    good_bb.add(i)
                    good_bb.add(j)

        # Make the union
        if len(good_bb) != 0:
            best_rectangle = list_bb[good_bb.pop()]
        else:
            best_rectangle = best_iou(list_bb)
        while len(good_bb) > 0:
            best_rectangle = union(best_rectangle, list_bb[good_bb.pop()])

        return best_rectangle
    else:
        return []
