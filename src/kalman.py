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
        the Initial Covariance Matrix

    Methods
    -------
    predict():
        Make a prediction of the bounding box
    update(z):
        Update the parameter according to the coordinates founded by an estimator
    """
    def __init__(self, dt, x_ini):
        """
        Parameters
        ----------
        dt : float
            the time between 2 frames
        x_ini : [x_middle, y_middle, vx, vy]
            the initialization values
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
        self.R = np.array([[20, 0],
                           [0, 20]])

        # Initial Covariance Matrix
        self.P = np.array([[5, 0, 0, 0],
                           [0, 5, 0, 0],
                           [0, 0, 10, 0],
                           [0, 0, 0, 10]])

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