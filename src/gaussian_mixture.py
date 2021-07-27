import maxflow
import cv2.cv2 as cv
import numpy as np
from scipy.sparse import coo
from sklearn.mixture import GaussianMixture
import pandas as pd
from src.preprocessing.water_surface_detection import polygon_construction
import pickle
import time
from skimage.util import img_as_ubyte
from skimage import exposure
from src.preprocessing.bb_tools import bb_building, neighborhood_bb

class GaussianMixtureBB():
    """
    This class is designed in order to make a segmentation of an image with a Gaussian Mixture Model.
    Some parameters are used outside the class in order to make a segmentation on a video. There is also
    the possibility to combine the probabilities of the model with a Markov Random Field model.

    Attributes
    ----------
    model : sklearn.mixture.GaussianMixture
        the Gaussian Mixture model
    margin : int
        the margin used to enlarge the box of the previous frame ( default is 100 )
    threshold : float
        the threshold value to classify value between 0 and 1 ( default is 0.5 )
    ising : tuple
        the parameters of the Ising Model
    weight : float
        the regularization term
    detect_surface : boolean
        if true, the method uses the detection of the surface ( default is True )
    adjust_pt1 : int
        decreasing the height of the left point ( default is 0 )
    adjust_pt2 : int
        increasing the height of the right point ( default is 30 )
    use_time : boolean
        if true, the method uses the precedent box to make the prediction ( default is True )
    graph_cut : boolean
        if true, the method uses the graph cut algorithm to make the prediction ( default is True )
    Methods
    --------
    set_params(**params)
        Set the parameters of the estimator
    predict(filename_img, debug=False, a=0, b=0, precBB=[])
        Prediction of a Gaussian Mixture Model
    train(self, dir_name, choice=None)
        Training of a Gaussian Mixture Model
    """
    def __init__(self, filename=None, margin=100, threshold=0.5, ising=(1.7638, 0.2452, 0.1231, 0.1288), weight=10, detect_surface=True, adjust_pt1=0, adjust_pt2=0, use_time=True, graph_cut=False):
        """
        Parameters
        -----------
        margin : int
            the margin used to enlarge the box of the previous frame ( default is 100 )
        threshold : float
            the threshold value to classify value between 0 and 1 ( default is 0.5 )
        ising : tuple
            the parameters of the Ising Model
        weight : float
            the regularization term
        detect_surface : boolean
            if true, the method uses the detection of the surface ( default is True )
        adjust_pt1 : int
            decreasing the height of the left point ( default is 0 )
        adjust_pt2 : int
            increasing the height of the right point ( default is 30 )
        use_time : boolean
            if true, the method uses the precedent box to make the prediction ( default is True )
        graph_cut : boolean
            if true, the method uses the graph cut algorithm to make the prediction ( default is False )
        """

        # If filename is defined, load the model
        if filename is None:
            self.model = GaussianMixture(n_components=2)
        else:
            self.model = pickle.load(open(filename, 'rb'))

        # Set the margin and the threshold
        self.margin = margin
        self.threshold = threshold

        # Set the graph cut variables
        self.graph_cut = graph_cut
        self.ising = ising
        self.weight = weight

        # Set the detect surface variables
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

        if 'ising' in params:
            self.ising = params.get('ising')

        if 'weight' in params:
            self.weight = params.get('weight')

        if 'adjust_pt1' in params:
            self.adjust_pt1 = params.get('adjust_pt1')

        if 'adjust_pt2' in params:
            self.adjust_pt2 = params.get('adjust_pt2')

        if 'use_time' in params:
            self.use_time = params.get('use_time')

        if 'graph_cut' in params:
            self.graph_cut = params.get('graph_cut')

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

        #Search the swimmer in the neighborhood of the precedent bounding box
        img, coord = neighborhood_bb(img, self.margin, precBB)
        x_prec, y_prec, w_prec, h_prec = coord
        if len(precBB) != 0 and self.graph_cut:
            self.dst = self.dst[y_prec:(y_prec + h_prec), x_prec:(x_prec + w_prec)]
            self.dst = self.dst.reshape(-1)

        # Create the dataframe
        img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img2 = img2.reshape(-1, 3)
        df = pd.DataFrame()
        df['ColourCode(H)'] = img2[:, 0]/255
        df['ColourCode(S)'] = img2[:, 1]/255
        df['ColourCode(V)'] = img2[:, 2]/255

        # Compute prediction
        prediction_gm = self.model.predict_proba(df)

        # Apply a mask to eliminate the light in the swimming pool
        mask_light = np.uint8((df['ColourCode(V)'].values < 250/255) * 255).reshape((h_prec, w_prec))

        kernel = (2, 2)
        mask_light = cv.morphologyEx(mask_light, cv.MORPH_CLOSE, kernel).reshape(-1)/255
        prediction_gm[mask_light == 0, 1] = 0
        prediction_gm[mask_light == 0, 0] = 1

        # Modification of the probabilities according to the distance of the last mask
        # founded
        if len(precBB) != 0 and self.graph_cut:
            temp1 = prediction_gm[:, 1] / self.dst
            temp2 = prediction_gm[:, 0] * self.dst
            prediction_gm[:, 1] = temp1 / (temp1 + temp2)
            prediction_gm[:, 0] = temp2 / (temp1 + temp2)


        #Texture/Non Texture segmentation
        polygon = polygon_construction(a, b, coord)

        # 2 cases according to the variable graph_cut
        if self.graph_cut:
            # Restructure the probabilities as image
            labels0 = prediction_gm[:, 0].reshape((h_prec, w_prec))
            labels1 = prediction_gm[:, 1].reshape((h_prec, w_prec))

            # Apply the texture/non texture segmentation
            labels0 = cv.fillPoly(np.ascontiguousarray(labels0*255, dtype=np.uint8), [polygon], 255)/255
            labels1 = cv.fillPoly(np.ascontiguousarray(labels1*255, dtype=np.uint8), [polygon], 0)/255

            # Compute the mask using graph cut
            mask, self.dst = graph_cut(labels0, labels1, coord, self.ising, self.weight)
        else:
            # Restructure the probabilities of each pixel to be one as an image
            score_img = prediction_gm[:, 1]
            score_img = score_img.reshape((h_prec, w_prec))

            # Convert the [0,1] range image to a binary image
            mask = np.array((score_img > self.threshold) * 255, dtype=np.uint8)

            # Apply Texture/Non Texture segmentation
            mask = cv.fillPoly(mask, [polygon], 0)

        if debug:
            cv.imshow('mask', mask)
            cv.waitKey(1)

        # Building the best bounding box according to the mask
        best_rectangle = bb_building(mask, 0)

        # Ajustement of the best box for the initial size of the image
        if len(best_rectangle)!=0 and len(precBB)!=0:
            best_rectangle[0] = best_rectangle[0] + x_prec
            best_rectangle[1] = best_rectangle[1] + y_prec
            x, y, w, h = best_rectangle
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if debug:
            cv.imshow('bounding box', img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return best_rectangle

    def train(self, dir_name, choice=None):
        """
        Training of the Gaussian Mixture

        This method makes the training of a Gaussian Mixture model

        Parameters
        -----------
        dir_name : str
            the directory name of the directory containing the frames
        choice : list
            the list of the videos of interest ( default is None )
        """
        # Reading the info.txt file
        f = open(f"{dir_name}/info.txt")
        data = f.readlines()
        f.close()

        # Preprocessing to have a structured format
        data_videos = [line.split(";") for line in data]
        data_videos = [[line[0], int(line[1]), int(line[2]), line[3].replace("\n", "")] for line in data_videos]

        # Choice of the videos on which video the model will be trained
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

                # Load the image
                file_name = dir_name + "/" + str(video_title) + str(nb_filename).zfill(5) + ".jpg"
                img = cv.imread(file_name)

                # Convert the image to HSV color space and flatten it
                img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)
                img2 = img2.reshape(-1, 3)

                # Compute the features for each pixel
                df2 = pd.DataFrame()
                df2['ColourCode(H)'] = img2[:, 0] / 255
                df2['ColourCode(S)'] = img2[:, 1] / 255
                df2['ColourCode(V)'] = img2[:, 2] / 255

                # Concatenation with precedent values
                df = pd.concat([df, df2], axis=0)

        # Fit the model
        self.model.fit(df)


def graph_cut(labels0, labels1, coord, ising, weight):
    """
    Compute the graph cut of the model defined in the report

    This method minimize the energy of the energy function
    defined in the report. The energy is defined in 2 parts :
        - the first part is linked with the probabilities given by
        labels0 and labels1
        - the second part is linked to anisotropic Ising model
    For more details, go to the report.

    Parameters
    -----------
    labels0 : np.array
        Map of probabilities for each pixel to be in the background
    labels1 : np.array
        Map of probabilities for each pixel to be in the foreground
    coord : list
        the coordinate of the researh area over the all image
    ising : tuple
        the parameters of the Ising Model
    weight : float
        the regularization term

    Returns
    -------
    mask : np.array
        the mask given by the minimization of the energy function
    dst : np.array
        the distance of each pixel to the mask
    """
    # Initialization of the graph
    g = maxflow.Graph[float]()

    # Retrieval of the shape of the picture
    img_shape = labels0.shape

    # Add a node for each pixel
    nodeids = g.add_grid_nodes(img_shape)

    # Parameters for the anisotropic Ising model
    beta1 = ising[0]
    beta2 = ising[1]
    beta3 = ising[2]
    beta4 = ising[3]

    # Add vertexes between neighboorhood pixels
    structure = 2 * np.array([[0, 0, 0],
                              [0, 0, beta1],
                              [beta3, beta2, beta4]])

    g.add_grid_edges(nodeids, weight, structure, symmetric=True)

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
    mask2 = np.zeros((480, 640), dtype=np.uint8)
    x_prec, y_prec, w_prec, h_prec = coord
    mask2[y_prec:(y_prec + h_prec), x_prec:(x_prec + w_prec)] = mask
    dst = cv.distanceTransform(255 - mask2, cv.DIST_L1, 3, dstType = cv.CV_32F)
    dst = dst/np.max(dst) + 0.1

    # Amplification of the distance
    limit = 0.2
    dst[dst < limit] = dst[dst < limit]*0.5
    dst[dst >= limit] = dst[dst >= limit]*100

    return mask, dst




