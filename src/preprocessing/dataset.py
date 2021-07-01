import numpy as np
import cv2
import pandas as pd
from skimage import io, transform
from skimage.filters import roberts, sobel, scharr, prewitt
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import os
from src.annotations.read_annotation import read_annotation

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


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bounding_box = sample['image'], sample['bounding_box']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'bounding_box': torch.from_numpy(bounding_box)}


class Normalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, mean, std):
        assert isinstance(mean, (list, tuple))
        assert isinstance(std, (list, tuple))
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        nmtf = transforms.Normalize(self.mean, self.std)
        sample['image'] = nmtf(sample['image'])

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, bounding_box = sample['image'], sample['bounding_box']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w), preserve_range=True)/255

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        bounding_box = np.array(
            [bounding_box[0] / w, bounding_box[1] / h, bounding_box[2] / w,
             bounding_box[3] / h])

        return {'image': img, 'bounding_box': bounding_box}


class SwimmerDataset(Dataset):
    """Swimmer dataset."""

    def __init__(self, img_dir, ant_dir, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            ant_dir (string): Directory with all the annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ant_dir = ant_dir
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):

        # Initialization of the number of frames
        nb_frames = 0

        # Reading the info.txt file
        f = open(f"{self.img_dir}/info.txt")
        data = f.readlines()
        f.close()

        # Preprocessing to have a structured format
        data_videos = [line.split(";") for line in data]
        data_videos = [[line[0], int(line[1]), int(line[2]), line[3].replace("\n", "")] for line in data_videos]

        # Compute the nb of frame
        for video in data_videos:
            nb_frames += video[2] - video[1]

        return nb_frames

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filenames = os.listdir(self.img_dir)
        filenames.remove("info.txt")
        filenames.remove("background")

        image = io.imread(self.img_dir + "/" + filenames[idx])

        ant_name = self.ant_dir + "/" + filenames[idx].split("/")[-1].split(".jpg")[0] + ".json"

        bounding_box = read_annotation(ant_name)
        bounding_box = [bounding_box[0], bounding_box[1], bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]]

        sample = {'image': image, 'bounding_box': np.array(bounding_box)}

        if self.transform:
            sample = self.transform(sample)

        return sample
