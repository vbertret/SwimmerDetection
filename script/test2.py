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

    img_dir = "data/images/"
    ant_dir = "data/annotations/"

    input_size = 224

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transformations = transforms.Compose([
            Rescale((input_size, input_size)),
            ToTensor(),
            Normalize(mean, std)
        ])

    transformations2 = transforms.Compose([
            Rescale((input_size, input_size)),
            RandomVerticalFlip(0.5),
            ToTensor(),
            ColorJitter(0.2, 0.1, 0.7, 0.2),
            Normalize(mean, std)
        ])

    trainset = SwimmerDataset(img_dir + "Trainset", ant_dir + "Trainset", transform=transformations2)
    valset = SwimmerDataset(img_dir + "Valset", ant_dir + "Valset", transform=transformations)

    print(len(trainset))

    # for idx, data in enumerate(trainset):
    #     print(idx)
    #     torch.save(data['image'],f'{img_dir}TransformedSet/{str(idx).zfill(5)}')
    #     torch.save(data['bounding_box'],f'{ant_dir}TransformedSet/{str(idx).zfill(5)}')
    # trainset = TransformedDataset(img_dir + "TransformedSet/", ant_dir + "TransformedSet/")
    
    batch_size = 16
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=8)

    model = Swimnet("mobilenet-v3-small")

    criterion = torch.nn.L1Loss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    max_epochs = 2000

    model = train_and_test(model, train_loader, val_loader, criterion, optimizer, max_epochs, tensorboard=None)

    



