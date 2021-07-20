import json
import numpy as np


def read_annotation(filename):
    """
    Read the annotation

    Read the annotations of the bounding box which contains the coordinates
    of bottom left and upper left vertices. It also converts the format of
    the bouding box into (x, y, w, h) with (x,y) the coordinates of the bottom
    left vertex and w, h respectively the width and the height.

    Parameters
    ----------
    filename : str
        The name of the file containing the annotations

    Returns
    -------
    BB: [x, y, w, h] list
        the coordinates of the bounding box
    """
    with open(filename) as json_file:

        # Loading data
        data = json.load(json_file)

        # Retrieval of the annotations
        vertices = data['annotations'][0]['geometry']["vertices"]
        bo_left = vertices[0:2]
        up_right = vertices[2:4]

        # Transformation to pixel value
        x1 = int(bo_left[0] * 640)
        y1 = int(bo_left[1] * 480)
        x2 = int(up_right[0] * 640)
        y2 = int(up_right[1] * 480)

        # Transformation to the type (x, y, w, h)
        BB = [x1, y1, x2-x1, y2-y1]

    return BB


