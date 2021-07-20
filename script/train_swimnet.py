import torch
from torchvision import transforms
from torchvision.transforms import autoaugment
from src.deep_learning import Swimnet, train, train_and_test, plot_bouding_box
from src.preprocessing.dataset import RandomHorizontalFlip, RandomVerticalFlip, Rescale, ToTensor, Normalize, SwimmerDataset, get_mean_std, TransformedDataset, ColorJitter, GaussianBlur
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2.cv2 as cv
import numpy as np

if __name__ == '__main__':

    # Directory of the images and the annotations
    img_dir = "data/images/"
    ant_dir = "data/annotations/"

    # Input size of the network
    input_size = 224

    # Mean and standard deviation for the normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Transformations for the ValidationSet
    transformations = transforms.Compose([
            Rescale((input_size, input_size)),
            ToTensor(),
            Normalize(mean, std)
        ])

    # Transformations for the TrainingSet with augmentation of the data
    transformations2 = transforms.Compose([
            Rescale((input_size, input_size)),
            RandomVerticalFlip(0.5),
            ToTensor(),
            ColorJitter(0.2, 0.1, 0.7, 0.2),
            Normalize(mean, std)
        ])

    # Definition of the trainset and valset
    trainset = SwimmerDataset(img_dir + "Trainset", ant_dir + "Trainset", transform=transformations2)
    valset = SwimmerDataset(img_dir + "Valset", ant_dir + "Valset", transform=transformations)
    
    # Definition of the dataloaders
    batch_size = 16
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=8)

    # Declaration of the model
    model = Swimnet("mobilenet-v3-small")

    # Declaration of the optimizer and the criterion
    criterion = torch.nn.L1Loss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training and Test of the model 
    max_epochs = 2000
    model = train_and_test(model, train_loader, val_loader, criterion, optimizer, max_epochs, tensorboard=None)

    



