import cv2.cv2
import numpy as np


def intersection(box1, box2):
    """
    Intersection

    Compute the intersection between box1 and box2

    Parameters
    -----------
    box1: (x1, y1, w1, h1) tuple
        parameters of the first rectangle box1
    box2: (x2, y2, w2, h2) tuple
        parameters of the second rectangle box2

    Returns
    -------
    (x_inter, y_inter, w_inter, h_inter) tuple
        parameters of the intersection box
    """

    # Computation of the intersection
    x_inter = max(box1[0], box2[0])
    y_inter = max(box1[1], box2[1])
    w_inter = min(box1[0] + box1[2], box2[0] + box2[2]) - x_inter
    h_inter = min(box1[1] + box1[3], box2[1] + box2[3]) - y_inter

    # If there is no intersection, it returns an empty tuple
    if w_inter < 0 or h_inter < 0:
        return 0, 0, 0, 0

    return x_inter, y_inter, w_inter, h_inter


def union(box1, box2):
    """
    Union

    Compute the union between box1 and box2

    Parameters
    -----------
    box1: (x1, y1, w1, h1) tuple
        parameters of the first rectangle box1
    box2: (x2, y2, w2, h2) tuple
        parameters of the second rectangle box2

    Returns
    -------
    (x_un, y_un, w_un, h_un) : tuple
        parameters of the union box
    """

    # Computation of the union
    x_un = min(box1[0], box2[0])
    y_un = min(box1[1], box2[1])
    w_un = max(box1[0] + box1[2], box2[0] + box2[2]) - x_un
    h_un = max(box1[1] + box1[3], box2[1] + box2[3]) - y_un

    return x_un, y_un, w_un, h_un