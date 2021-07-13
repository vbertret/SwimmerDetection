import torch
from torchvision import transforms
from src.deep_learning import Swimnet, train_and_test, plot_bouding_box
from src.preprocessing.dataset import Rescale, ToTensor, Normalize, SwimmerDataset, get_mean_std
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2.cv2 as cv

if __name__ == '__main__':

    img_dir = "../data/images/"
    ant_dir = "../data/annotations/"

    input_size = 224
    # transformations = transforms.Compose([
    #         Rescale((input_size, input_size)),
    #         ToTensor()
    #     ])
    #
    # trainset = SwimmerDataset(img_dir + "Trainset", ant_dir + "Trainset", transform=transformations)
    # mean, std = get_mean_std(trainset)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transformations2 = transforms.Compose([
            Rescale((input_size, input_size)),
            ToTensor(),
            Normalize(mean, std),
        ])

    trainset = SwimmerDataset(img_dir + "Trainset", ant_dir + "Trainset", transform=transformations2)
    valset = SwimmerDataset(img_dir + "Valset", ant_dir + "Valset", transform=transformations2)

    batch_size = 16
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=8)



