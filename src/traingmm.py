import maxflow
import cv2.cv2 as cv
import numpy as np
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
import pandas as pd
from Code.Preprocessing.WaterSurfaceDetection import surface_detection
import pickle

gmm = pickle.load(open("../../Models/GMM_model_diag", 'rb'))

filename = "../../Images/Valset/V2"

df = pd.DataFrame()
for i in range(100):

    img = cv.imread(f"{filename}{str(i).zfill(5)}.jpg")

    img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    H, S, V = cv.split(img2)

    img2 = img2.reshape(-1, 3)

    df2 = pd.DataFrame()
    df2['ColourCode(H)'] = img2[:, 0]
    df2['ColourCode(S)'] = img2[:, 1]
    df2['ColourCode(V)'] = img2[:, 2]

    pred = gmm.predict_proba(df2)[:, 1]

    score_img = pred.reshape((480, 640))

    cv.imshow("mask", score_img)

    cv.waitKey(0)


    df = pd.concat([df, df2], axis=0)

print("V3 OK")

filename = "../../Images/Valset/V3"
for i in range(100):

    img = cv.imread(f"{filename}{str(i).zfill(5)}.jpg")

    img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    H, S, V = cv.split(img2)

    img2 = img2.reshape(-1, 3)

    df2 = pd.DataFrame()
    df2['ColourCode(H)'] = img2[:, 0]
    df2['ColourCode(S)'] = img2[:, 1]
    df2['ColourCode(V)'] = img2[:, 2]

    pred = gmm.predict_proba(df2)[:, 1]

    score_img = pred.reshape((480, 640))

    cv.imshow("mask", score_img)

    cv.waitKey(0)

    df = pd.concat([df, df2], axis=0)

print("V2 OK")


# gmm = GaussianMixture(n_components=2, covariance_type='diag').fit(df)
#
# filename = "../../Models/GMM_model_diag"
# pickle.dump(gmm, open(filename, 'wb'))


