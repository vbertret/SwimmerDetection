import numpy as np

class KalmanFilter(object):
    """
    Implementation of the Kalman Filter

    This class implements a kalman filter in order to predict
    the coordinates of the middle of the bounding box containing 
    the swimmer. For more details, see the report.

    Attributes
    ----------
    dt : float
        the sampling time
    x : list [x_middle, y_middle, vx, vy]
        the coordinate of the middle and the speed of the bounding box
    A : np.array
        the state transition matrix
    H : np.array
        the Measurement Mapping Matrix
    Q : np.array
        the Process Noise Covariance
    R : np.array
        the Initial Measurement Noise Covariance
    P : np.array
        the initial covariance matrix
    method : str
        the name of the method to compute the height and the width of the bounding box
    beta, nbstep, S_w, S_h : float
        the values used to compute the exponentially weighted average for the width and the height
    max_h, max_w : list
        the values used to find the maximum among the 20 last value of the width and the height

    Methods
    -------
    predict():
        Make a prediction of the middle of the bounding box
    predictBB(bounding_box, method, w, h):
        Make the prediction of the entire bounding box
    update(z):
        Update the parameter according to the coordinates founded by an estimator
    """
    def __init__(self, dt, x_ini, method, w, h):
        """
        Parameters
        ----------
        dt : float
            the time between 2 frames
        x_ini : [x_middle, y_middle, vx, vy]
            the initialization values
        method : str
            the name of the method to compute the height and the width of the bounding box
        w : int
            the width of the first bounding box found
        h : int
            the height of the first bounding box found
        """
        # Define sampling time ( time between 2 frames)
        self.dt = dt

        # Initial State
        self.x = x_ini

        # Define the state transition matrix A
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Define Measurement Mapping Matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Process Noise Covariance
        self.Q = 0.1 * np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 10, 0],
                           [0, 0, 0, 10]])

        # Initial Measurement Noise Covariance
        self.R = np.array([[1, 0],
                           [0, 1]])

        # Initial Covariance Matrix
        self.P = np.array([[5, 0, 0, 0],
                           [0, 5, 0, 0],
                           [0, 0, 10, 0],
                           [0, 0, 0, 10]])

        # Initialization of different values according to the method
        self.method = method
        if self.method == "avg":
            self.beta = 0.95
            self.nbstep=1
            self.S_w = (1-self.beta) * w
            self.S_h = (1-self.beta) * h
        elif self.method == "max":
            self.max_h = [h]
            self.max_w = [w]

    def predict(self):
        """
        Prediction of the Kalman Filter

        Return
        ------
        x : [x_middle, y_middle]
            the coordinates of the middle of the bouding box
        """
        # Update time state
        # x_k = A*x_(k-1)
        self.x = np.dot(self.A, self.x)

        # Compute error covariance
        # P = A*P*A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[:2]

    def predictBB(self, bounding_box, method):
        """
        Make the prediction of the entire bounding box.

        Parameters
        ----------
        bounding_box : list
            The coordinates of the bounding box founded by an algorithm. If it has found no
            bounding box, the list is empty.
        method : str
            the name of the method used to compute the height and the width of the bounding box
        
        Return
        ------
        BB : [x, y, w, h] list
            the coordinnates of the bounding box predicted by the Kalman Filter
        """
        
        # Compute the prediction of the Kalman filter
        middle_x, middle_y = self.predict()

        if len(bounding_box) != 0:


            # Recovering the coordinates of the bounding box
            x, y, w, h = bounding_box

            # Incrementation of the nb of value for the exponentially weighted average
            if self.method == "avg":
                self.nbstep += 1

            # Update the max or the weighted average for h and w
            if self.method == "avg":
                self.S_h = self.beta * self.S_h + (1-self.beta) * h
                self.S_w = self.beta * self.S_w + (1-self.beta) * w
            elif self.method == "max":
                if len(self.max_h)>=20:
                    self.max_h.pop(0)
                    self.max_w.pop(0)
                self.max_h.append(h)
                self.max_w.append(w)

            # Update the prediction of the Kalman filter with the value found by the estimator
            middle_x, middle_y = self.update([[x + w/2], [y + h/2]])

        # Compute the height and the width of the bounding box according to the method
        if self.method == "avg":
            S_h_adj = self.S_h / (1 - self.beta ** self.nbstep)
            S_w_adj = self.S_w / (1 - self.beta ** self.nbstep)
            x = int(middle_x - S_w_adj/2)
            y = int(middle_y - S_h_adj/2)
            w = int(S_w_adj)
            h = int(S_h_adj)
        elif self.method == "max":
            w = int(max(self.max_w))
            h = int(max(self.max_h))
            x = int(middle_x - w/2)
            y = int(middle_y - h/2)
            
        BB = [x, y, w, h]

        return BB

    def update(self, z):
        """
        Update of the Kalman Filter

        Parameter
        ---------
        z : list [x_middle, y_middle]
            the coordinates of the middle of the bounding box founded at a step

        Return
        ------
        x : [x_middle, y_middle]
            the coordinates of the middle of the bouding box
        """
        # S = H*P*H' + R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))
        I = np.eye(self.H.shape[1])

        # Update error covariance matrix
        self.P = np.dot((I - (np.dot(K, self.H))), self.P)

        return self.x[:2]