import numpy as np
import cv2
import pandas as pd
from skimage.filters import roberts, sobel, scharr, prewitt


def createDataframe(img, bounding_box=[]):
    ########################################
    # Lecture de l image
    ########################################
    # Prétraitement : Applatissement de l'image et enregistrement dans un dataframe
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img2 = img2.reshape(-1, 3)

    df = pd.DataFrame()
    df['ColourCode(H)'] = img2[:, 0]
    df['ColourCode(S)'] = img2[:, 1]
    df['ColourCode(V)'] = img2[:, 2]

    # df['yval'] = np.repeat([i for i in range(640)], 480)
    # df['xval'] = np.tile([i for i in range(640)], 480)

    ########################################
    # Generation de plusieurs filtre de Gabor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    num = 1  # To count numbers up in order to give Gabor features a lable in the data frame
    kernels = []
    for theta in range(2):  # Define number of thetas
        theta = theta / 4. * np.pi
        for sigma in (1, 3):  # Sigma with 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4):  # Range of wavelengths
                for gamma in (0.05, 0.5):  # Gamma values of 0.05 and 0.5

                    gabor_label = 'Gabor' + str(num)  # Label Gabor columns as Gabor1, Gabor2, etc.
                    #                print(gabor_label)
                    ksize = 9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)
                    # Now filter the image and add values to a new column
                    fimg = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  # Labels columns as Gabor1, Gabor2, etc.
                    # print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1  # Increment for gabor column label

    ########################################
    # Gerate OTHER FEATURES and add them to the data frame

    # CANNY EDGE
    edges = cv2.Canny(img, 100, 200)  # Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1  # Add column to original dataframe

    # ROBERTS EDGE
    edge_roberts = roberts(gray)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1

    # SOBEL
    edge_sobel = sobel(gray)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1

    # SCHARR
    edge_scharr = scharr(gray)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1

    # PREWITT
    edge_prewitt = prewitt(gray)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1

    # GAUSSIAN with sigma=3
    from scipy import ndimage as nd

    gaussian_img = nd.gaussian_filter(gray, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

    # GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(gray, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3

    # MEDIAN with sigma=3
    median_img = nd.median_filter(gray, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1

    ########################################
    # Génération des labels de chaque pixel
    if len(bounding_box) != 0:
        df['Labels'] = bounding_box.reshape(-1)

    return df
