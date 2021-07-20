import numpy as np
from skimage import io, transform
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import os
from src.annotations.read_annotation import read_annotation
from torch.utils.data import DataLoader
import random
import cv2.cv2 as cv

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, bounding_box = sample['image'], sample['bounding_box']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image.copy()),
                'bounding_box': torch.from_numpy(bounding_box.copy())}

class Normalize(object):
    """Normalize dataset

    Args:
        mean : tuple
            the mean used for the normalization
        std : tuple
            the standard deviation used for the normalization
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (list, tuple))
        assert isinstance(std, (list, tuple))
        self.mean = mean # utiliser moyenne
        self.std = std

    def __call__(self, sample):
        nmtf = transforms.Normalize(self.mean, self.std)
        sample['image'] = nmtf(sample['image'])

        return sample

class ColorJitter(object):
    """
    Randomly change the brightness, contrast, saturation and hue of an image.

    Args:
        brightness (float or tuple of python:float (min, max)) – How much to jitter brightness. brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. Should be non negative numbers.

        contrast (float or tuple of python:float (min, max)) – How much to jitter contrast. contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. Should be non negative numbers.

        saturation (float or tuple of python:float (min, max)) – How much to jitter saturation. saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. Should be non negative numbers.

        hue (float or tuple of python:float (min, max)) – How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue] or the given [min, max]. Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        cj = transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)
        sample['image'] = cj(sample['image'])

        return sample

class RandomHorizontalFlip(object):
    """
    Horizontally flip the given image randomly with a given probability.

    Args : 
        p (float) – probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p 

    def __call__(self, sample):
        img_center = np.array(sample['image'].shape[0:2])[::-1]/(2*224)
        img_center = np.hstack((img_center, img_center))
        if random.random() < self.p:
            sample['image'] = sample['image'][::-1, :, :]
            sample['bounding_box'][[1,3]] += 2*(img_center[[1,3]] - sample['bounding_box'][[1,3]])

            box_h = abs(sample['bounding_box'][1] - sample['bounding_box'][3])
             
            sample['bounding_box'][1] -= box_h
            sample['bounding_box'][3] += box_h
        return sample

class RandomVerticalFlip(object):
    """
    Vertically flip the given image randomly with a given probability.

    Args : 
        p (float) – probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p 

    def __call__(self, sample):
        img_center = np.array(sample['image'].shape[0:2])[::-1]/(2*224)
        img_center = np.hstack((img_center, img_center))
        if random.random() < self.p:
            sample['image'] = sample['image'][:, ::-1, :]
            sample['bounding_box'][[0,2]] += 2*(img_center[[0,2]] - sample['bounding_box'][[0,2]])

            box_w = abs(sample['bounding_box'][0] - sample['bounding_box'][2])
             
            sample['bounding_box'][0] -= box_w
            sample['bounding_box'][2] += box_w
        return sample

class GaussianBlur(object):
    """
    Blurs image with randomly chosen Gaussian blur.

    Args :
        kernel_size (int or sequence) – Size of the Gaussian kernel.

        sigma (float or tuple of python:float (min, max)) – Standard deviation to be used for creating kernel to perform blurring. If float, sigma is fixed. If it is tuple of float (min, max), sigma is chosen uniformly at random to lie in the given range.
    
    """
    def __init__(self, kernel_size, sigma=(0.1, 2)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, sample):
        blur = transforms.GaussianBlur(self.kernel_size, self.sigma)
        sample['image'] = blur(sample['image'])

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size : tuple or int
            Desired output size. If tuple, output is
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
    """Swimmer dataset.
    
    The class implements different methods in order to load and make transformation 
    on all the images of the dataset.

    Attributes
    ----------
    transform : torch.transfroms.compose
        a list of transformations to do on the picture
    length : int
        the number of images of the dataset
    """

    def __init__(self, img_dir, ant_dir, transform=None):
        """
        Parameters
        ----------
        img_dir : str
            Directory with all the images.
        ant_dir : str
             Directory with all the annotations.
        transform : callable, optional
            Optional transform to be applied on a sample.
        """

        # Initialization of the attributes
        self.ant_dir = ant_dir
        self.img_dir = img_dir
        self.transform = transform

        # Read the length of the dataset
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

        self.length = nb_frames

    def __len__(self):

        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Retrieval of all the filenames and remove some of them
        filenames = sorted(os.listdir(self.img_dir))
        filenames.remove("info.txt")
        filenames.remove("background")

        # Read the picture corresponding to the index idx
        image = io.imread(self.img_dir + "/" + filenames[idx])

        # Read the annotation corresponding to the index idx
        ant_name = self.ant_dir + "/" + filenames[idx].split("/")[-1].split(".jpg")[0] + ".json"
        bounding_box = read_annotation(ant_name)

        # Convert the format of the bounding from [x, y ,w, h] to [x1, y1, x2, y2]
        bounding_box = [bounding_box[0], bounding_box[1], bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]]

        # Return a dict with 2 keys, one for the image and the other for the annotation
        sample = {'image': image, 'bounding_box': np.array(bounding_box)}

        # If specified, make the transformation on the image
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

def get_mean_std(dataset):
    """
    Compute the mean and the standard deviation of each channel of a dataset
    
    Parameter
    ---------
    dataset : torch.utils.data.Dataset
        a dataset which each sample is a dictionnary with the key 'image' for the picture
    
    Return
    ------
    mean : list
        list of the mean for each channel
    std : list
        list of the standard deviation for each channel
    """

    # Load the data into a dataloader
    loader = DataLoader(dataset, batch_size=len(dataset), num_workers=8)

    # Initialize the value for the sum
    channels_sum, channels_squares_sum, num_batches = 0, 0, 0

    # Make the summation of the mean and the mean square along all the batches
    for sample in loader:
        data = sample['image']
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squares_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    # Compute the mean and the standard deviation
    mean = channels_sum/num_batches
    std = (channels_squares_sum/num_batches - mean**2/num_batches)**0.5

    return mean.numpy().tolist(), std.numpy().tolist()
