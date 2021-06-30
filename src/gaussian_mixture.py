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
from src.preprocessing.bb_tools import union, intersection

class GaussianMixtureBB():

    def __init__(self, filename=None, margin=100, threshold=0.5,  detect_surface=True, adjust_pt1=0, adjust_pt2=30, use_time=True, graph_cut=True):

        if filename is None:
            self.model = GaussianMixture(n_components=2)
        else:
            self.model = pickle.load(open(filename, 'rb'))

        # Set the margin and the threshold
        self.margin = margin
        self.threshold = threshold

        self.graph_cut = graph_cut

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
        if 'margin' in params:
            self.margin = params.get('margin')

        if 'detect_surface' in params:
            self.detect_surface = params.get('detect_surface')

        if 'threshold' in params:
            self.threshold = params.get('threshold')

        if 'adjust_pt1' in params:
            self.adjust_pt1 = params.get('adjust_pt1')

        if 'adjust_pt2' in params:
            self.adjust_pt2 = params.get('adjust_pt2')

        if 'use_time' in params:
            self.use_time = params.get('use_time')

    def predict(self, filename_img, debug=False, a=0, b=0, precBB=[]):
        """
        Prediction of the Gaussian Mixture

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

            if len(precBB) != 0 and self.graph_cut:
                self.dst = self.dst[y_prec:(y_prec + h_prec), x_prec:(x_prec + w_prec)]
                self.dst = self.dst.reshape(-1)

        # Creation dataframe
        img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        img2 = img2.reshape(-1, 3)

        df = pd.DataFrame()
        df['ColourCode(H)'] = img2[:, 0]/255
        df['ColourCode(S)'] = img2[:, 1]/255
        df['ColourCode(V)'] = img2[:, 2]/255

        # Compute prediction
        prediction_gm = self.model.predict_proba(df)

        if len(precBB) != 0 and self.graph_cut:
            temp1 = prediction_gm[:, 1] / self.dst
            temp2 = prediction_gm[:, 0] * self.dst
            prediction_gm[:, 1] = temp1 / (temp1 + temp2)
            prediction_gm[:, 0] = temp2 / (temp1 + temp2)

        if len(precBB) != 0:
            mask_light = np.uint8((df['ColourCode(V)'].values < 250/255) * 255).reshape((w_prec, h_prec))
        else:
            mask_light = np.uint8((df['ColourCode(V)'].values < 250 / 255) * 255).reshape((480, 640))

        kernel = (2, 2)
        mask_light = cv.morphologyEx(mask_light, cv.MORPH_CLOSE, kernel).reshape(-1)/255

        prediction_gm[mask_light == 0, 1] = 0
        prediction_gm[mask_light == 0, 0] = 1

        score_img = prediction_gm[:, 1]

        # Building the mask
        if len(precBB)!=0:
            score_img = score_img.reshape((h_prec, w_prec))
        else:
            score_img = score_img.reshape((480, 640))

        if self.graph_cut:

            if len(precBB)!=0:
                labels0 = prediction_gm[:, 0].reshape((h_prec, w_prec))
                labels1 = prediction_gm[:, 1].reshape((h_prec, w_prec))
            else:
                labels0 = prediction_gm[:, 0].reshape((480, 640))
                labels1 = prediction_gm[:, 1].reshape((480, 640))

            mask, self.dst = graph_cut(labels0, labels1)
        else:
            mask = np.array((score_img > self.threshold) * 255, dtype=np.uint8)


        #Texture/Non Texture segmentation
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
            x, y, w, h = min(rects, key=lambda p: p[1])
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

    def train(self, dir_name, choice=None):

        # Reading the info.txt file
        f = open(f"{dir_name}/info.txt")
        data = f.readlines()
        f.close()

        # Preprocessing to have a structured format
        data_videos = [line.split(";") for line in data]
        data_videos = [[line[0], int(line[1]), int(line[2]), line[3].replace("\n", "")] for line in data_videos]

        # Choice of the videos on which the model will be computed
        if choice == None:
            queue = range(len(data_videos))
        else:
            queue = [i - 1 for i in choice]

        # Initialization of the dataframe
        df = pd.DataFrame()

        # Computation of the rows for all the frames in the videos from the Queue
        for i in queue:

            # Retrieving of the video title and the range the values of the frames
            video_title = data_videos[i][0]
            start = data_videos[i][1]
            end = data_videos[i][2]
            nb = end - start

            print(f"Video Treated : {video_title}...")

            # Computation of the rows for all the frames of the video
            for nb_filename in range(start, end, 4):

                file_name = dir_name + "/" + str(video_title) + str(nb_filename).zfill(5) + ".jpg"
                img = cv.imread(file_name)

                img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)

                img2 = img2.reshape(-1, 3)

                df2 = pd.DataFrame()
                df2['ColourCode(H)'] = img2[valid, 0] / 255
                df2['ColourCode(S)'] = img2[valid, 1] / 255
                df2['ColourCode(V)'] = img2[valid, 2] / 255

                df = pd.concat([df, df2], axis=0)

        # Fit the model
        self.model.fit(df)


def graph_cut(labels0, labels1):

    # Initialization of the graph
    g = maxflow.Graph[float]()

    # Retrieval of the shape of the picture
    img_shape = labels0.shape

    # Add a node for each pixel
    nodeids = g.add_grid_nodes(img_shape)

    # Parameters for the anisotropic Ising model
    beta1 = 1.7638
    beta2 = 0.2452
    beta3 = 0.1231
    beta4 = 0.1288

    # Add vertexes between neighboorhood pixels
    structure = 2 * np.array([[0, 0, 0],
                              [0, 0, beta1],
                              [beta3, beta2, beta4]])
    weights = 10

    g.add_grid_edges(nodeids, weights, structure, symmetric=True)

    # Add vertexes between each node and the background and foreground node
    prob = labels1 > 0.5
    eps = 10 ** (-10)
    weights = np.log(labels1 + eps) - np.log(labels0 + eps)

    g.add_grid_tedges(nodeids, -(1-prob)*weights, prob*weights)

    # Computation of the flow
    flow = g.maxflow()

    # Min/Max cup
    mask = np.uint8(g.get_grid_segments(nodeids) * 255)

    # Computation of the geodesic distance between all the pixels and the mask
    dst = cv.distanceTransform(255 - mask, cv.DIST_L1, 3, dstType = cv.CV_8U)

    return mask, dst




