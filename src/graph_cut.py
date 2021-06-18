import maxflow
import cv2.cv2 as cv
import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd
from src.preprocessing.water_surface_detection import surface_detection
import pickle
import time
from skimage.util import img_as_ubyte
from skimage import exposure

gmm = pickle.load(open("../Models/GMM_model", 'rb'))
a, b = surface_detection(f"../Images/Testset/background/T1T2", 117, adjust_pt1=0, adjust_pt2=30)

for nb in range(200):
    img = cv.imread(f"../Images/Testset/T1{str(nb).zfill(5)}.jpg")

    t = time.time()

    img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img2 = img2.reshape(-1, 3)

    df = pd.DataFrame()
    df['ColourCode(H)'] = img2[:, 0]
    df['ColourCode(S)'] = img2[:, 1]
    df['ColourCode(V)'] = img2[:, 2]

    labels = gmm.predict_proba(df)
    labels = labels.reshape((480, 640, 2))
    labels1 = labels[:, :, 1].reshape((480, 640))
    labels0 = labels[:, :, 0].reshape((480, 640))

    mask2 = np.uint8(((df['ColourCode(V)'].values < 240)*255).reshape((480, 640)))

    cv.imshow("mask2", mask2)
    mask2 = cv.morphologyEx(mask2, cv.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))

    #labels1 = cv.morphologyEx(labels1, cv.MORPH_OPEN, np.ones((2,2), dtype=np.uint8))
    #labels1 = cv.dilate(labels1, None, iterations=3)

    if nb!=0 and np.sum(labels1) > 640*480*0.05:
        labels1 = labels1/dst_int
        labels0 = labels0*dst_int

    labels1 = np.uint8(labels1*255) & mask2
    labels0 = np.uint8(labels0*255)

    #a, b = 0.0328125, 200.0
    pt1 = (0, int(a * 0 + b))
    pt2 = (640, int(a * 640 + b))
    pt3 = (640, 0)
    pt4 = (0, 0)
    polygon = np.array([pt1, pt2, pt3, pt4])
    cv.fillPoly(labels1, [polygon], 0)
    cv.fillPoly(labels0, [polygon], 255)

    # cv.imshow("GMM0", labels0)
    # cv.imshow("GMM1", labels1)
    # cv.waitKey(1)

    g = maxflow.Graph[float]()

    img_shape = (480, 640)

    nodeids = g.add_grid_nodes(img_shape)

    beta1 = 1.7638
    beta2 = 0.2452
    beta3 = 0.1231
    beta4 = 0.1288
    # x axis
    structure = 2 * np.array([[0    , 0    , 0    ],
                              [0    , 0    , beta1],
                              [beta3, beta2, beta4]])
    weights = 20

    g.add_grid_edges(nodeids, weights, structure, symmetric=True)

    # # y axis
    # structure = 0.4892 * np.array([[0, 0, 0],
    #                                [0, 0, 0],
    #                                [0, 2, 0]])
    # weights = lmd
    #
    # g.add_grid_edges(nodeids, weights, structure, symmetric=True)
    #
    # # diagonal 1
    # structure = 0.0665 * np.array([[0, 0, 0],
    #                                [0, 0, 0],
    #                                [0, 0, 2]])
    # weights = lmd
    #
    # g.add_grid_edges(nodeids, weights, structure, symmetric=True)
    #
    # # diagonal 2
    # structure = -0.1329 * np.array([[0, 0, 0],
    #                                [0, 0, 0],
    #                                [2, 0, 0]])
    # weights = lmd

    #g.add_grid_edges(nodeids, weights, structure, symmetric=True)

    prob = labels1/255 > 0.5
    eps= 10 ** (-10)
    weights = np.log(labels1/255 + eps) - np.log(labels0/255 + eps)

    g.add_grid_tedges(nodeids, -(1-prob)*weights, prob*weights)

    flow = g.maxflow()

    print("Maximum flow:", flow)

    segments = np.uint8(g.get_grid_segments(nodeids)*255)

    dst = cv.distanceTransform(255 - segments, cv.DIST_L1, 3, dstType = cv.CV_8U)
    dst_int = np.uint8(exposure.rescale_intensity(dst, in_range=(0, np.max(dst)), out_range=(0, 255)))
    dst_int[dst_int < 20] = 1

    mask = cv.bitwise_and(img, img, mask=segments)

    print("fps : ", 1/(time.time()-t))

    color = cv.applyColorMap(dst_int, cv.COLORMAP_HOT)
    cv.imshow("Distance", dst_int)
    cv.imshow("GMM", labels1)
    cv.imshow("partition", mask)
    cv.waitKey(1)
