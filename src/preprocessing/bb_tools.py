import cv2.cv2 as cv
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


def bb_building(mask):
    """
    Find the best bounding box according to the mask

    Parameter
    ---------
    mask : np.array
        a binary picture corresponding to background and foreground pixels

    Return
    ------
    best_rectangle : list
        the coordinates of the best bouding box
    """

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

    # Find the lowest according to y-axis
    best_rectangle = []
    if len(rects) != 0:
        x, y, w, h = min(rects, key=lambda p: p[1])
        best_rectangle = [x, y, w, h]
        
    return best_rectangle


def neighborhood_bb(img, margin, precBB):
    """
    Make a local research

    Given the image and the coordinates of the precedent region of interest(RoI), the method crops the
    image in order to make a local research around the precedent location of the RoI.

    Parameters
    ----------
    img : array
        the image with the RoI
    margin : int
        the margin used to enlarge the box of the previous frame
    precBB : [x, y, w, h] list
        the coordinates of the bounding box found at the previous frame ( default is [] )

    Returns
    ------
    img : array
        the new image cropped
    coord : list
        the coordinate of the new image on the whole image
    """

    # If a RoI was found on the precedent image, the image can be cropped
    if len(precBB) != 0:
        # Retrieval of the coordinates
        x_prec, y_prec, w_prec, h_prec = precBB

        # Take a neighborhood of the precedent bounding box
        x_prec = max(x_prec - margin, 0)
        y_prec = max(y_prec - margin, 0)
        w_prec = min(w_prec + 2*margin, 640 - x_prec)
        h_prec = min(h_prec + 2*margin, 480 - y_prec)

        # Cropping the image
        img = img[y_prec:(y_prec + h_prec), x_prec:(x_prec + w_prec), :]   

        # Save the coordinates
        coord = [x_prec, y_prec, w_prec, h_prec]
    else:
        coord = [0, 0, 640, 480]

    return img, coord   