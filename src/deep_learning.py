import torch
from skimage import io, transform
import torch.nn as nn
import torchvision
from src.metrics.model_performance import IoU
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import cv2.cv2 as cv
from torchvision import transforms
from src.preprocessing.dataset import Rescale, ToTensor, Normalize, SwimmerDataset

# All the different feature extractor available
model_builder = {'resnet18'     : [lambda:torchvision.models.resnet18(pretrained=True), 512],
                 'resnet34'     : [lambda:torchvision.models.resnet34(pretrained=True), 25_088],
                 'resnet50'     : [lambda:torchvision.models.resnet50(pretrained=True), 2048],
                 'resnet152'    : [lambda:torchvision.models.resnet152(pretrained=True), 100_352],
                 'densenet121'  : [lambda:torchvision.models.densenet121(pretrained=True), 50_176],
                 'squeezenet1_1': [lambda:torchvision.models.squeezenet1_1(pretrained=True), 86_528],
                 'mobilenet-v3-large': [lambda:torchvision.models.mobilenet_v3_large(pretrained=True), 960],
                 'mobilenet-v3-small': [lambda:torchvision.models.mobilenet_v3_small(pretrained=True), 576],
                 'mobilenet-v2': [lambda:torchvision.models.mobilenet_v2(pretrained=True), 1280],
                 'efficientnet' : [None, 2]}



class FeatureExtractor(nn.Module):

    def __init__(self, model_name):
        super(FeatureExtractor, self).__init__()

        # Load the model
        model = model_builder[model_name][0]()
        self.num_features = model_builder[model_name][1]

        # Freeze the network. We don't want to update this part of the model
        for param in model.parameters():
            param.requires_grad = True

        # We keep only the feature maps of all the models
        if 'resnet' in model_name:
            self.body = nn.Sequential(*list(model.children())[:-1])
        elif 'densenet' in model_name or 'squeezenet' in model_name:
            self.body = model.features
        elif 'mobilenet-v3' in model_name:
            self.body = nn.Sequential(*list(model.children())[:-1])
        elif 'mobilenet-v2' in model_name:
            self.body = model.features

    def forward(self, x):
        return self.body(x)


class Swimnet(nn.Module):

    def __init__(self, feature_extractor_name, type=1):
        super(Swimnet, self).__init__()

        # Initialize the feature extractor
        self.feature_extractor = FeatureExtractor(feature_extractor_name)

        # Initialize the head bbox
        """
        self.head = nn.Sequential(
                                       nn.Linear(self.feature_extractor.num_features, 1024), nn.ReLU(),
                                       nn.BatchNorm1d(1024),
                                       nn.Linear(1024, 256), nn.ReLU(),
                                       nn.BatchNorm1d(256),
                                       nn.Linear(256, 64), nn.ReLU(),
                                       nn.BatchNorm1d(64),
                                       nn.Linear(64, 4), nn.Sigmoid())
        """ # before, only one layer with 1024 neurons
        self.head = nn.Sequential(nn.Dropout(),
                              nn.Linear(self.feature_extractor.num_features, 1024), nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.BatchNorm1d(1024),
                              nn.Linear(1024, 4), nn.Sigmoid()) #alban nicolas beuve  #pas de generalisation car pas assez de video #data augmentation (flippé, decallé, rajouter de la luminosité)

        self.detect_surface = False
        self.use_time = False
        self.type = type

    def forward(self, x):

        # Flatten the inputs
        features = self.feature_extractor(x)
        if self.type == 2:
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size()[0], -1)
        bbox = self.head(features)

        return bbox

    def predict(self, file_name, a=0, b=0, precBB=[]):
    #[0.485, 0.456, 0.406] # [0.229, 0.224, 0.225] #
        mean_nm = [0.11356854810507422, 0.45112476323143036, 0.6064378756359288]
        std_nm = [0.14808664052222273, 0.17381441351050686, 0.1814740496541503]  #

        input_size = 224
        transformations = transforms.Compose([
            Rescale((input_size, input_size)),
            ToTensor(),
            Normalize(mean_nm, std_nm)
        ])

        img = io.imread(file_name)
        sample = {'image': img, 'bounding_box': np.array([0, 0, 0, 0])}
        sample = transformations(sample)
        x = sample['image']
        x = torch.unsqueeze(x, 0).float()

        self.eval()
        bbox = self.forward(x)[0].detach().numpy()
        bbox = [int(bbox[0]*640), int(bbox[1]*480), int((bbox[2] - bbox[0])*640), int((bbox[3] - bbox[1])*480)]

        return bbox


def train(model, dataloader, criterion, optimizer, device, tensorboard=(None, 0)):
    # Set the model to training mode. This will turn on layers that would
    # otherwise behave differently during evaluation, such as dropout.
    model.train()

    # Sum the Intersection over Union for every batch
    iou_total = 0

    # Configuration of tensorboard
    if tensorboard[0] is not None:
        writer_cop = tensorboard[0]
        epoch = tensorboard[1]

    # Iterate over every batch of sequences.
    for num_batch, sample_batched in enumerate(dataloader):

        # Request a batch of images and bounding boxes, convert them into tensors
        # of the correct type, and then send them to the appropriate device.
        inputs = sample_batched['image'].float().to(device)
        bbs = sample_batched['bounding_box'].float().to(device)

        # Perform the forward pass of the model
        outputs = model(inputs)  # Step ①

        # Compute the value of the loss and the Intersection over Union for this batch
        loss = criterion(outputs, bbs)
        iou = [IoU([outputs[i][0], outputs[i][1], outputs[i][2] - outputs[i][0], outputs[i][3] - outputs[i][1]],
                          [bbs[i][0], bbs[i][1], bbs[i][2] - bbs[i][0], bbs[i][3] - bbs[i][1]]).cpu().detach().item() for i in range(outputs.shape[0])]
        iou = np.sum(iou)
        iou_total += iou

        # Clear the gradient buffers of the optimized parameters.
        # Otherwise, gradients from the previous batch would be accumulated.
        optimizer.zero_grad()  # Step ③

        loss.backward()  # Step ④

        optimizer.step()  # Step ⑤

        if tensorboard[0] is not None and num_batch % 10 == 0:
            writer_cop.add_scalar('training loss per batch', loss / dataloader.batch_size, epoch * np.ceil(len(dataloader.dataset))/dataloader.batch_size + num_batch)
            writer_cop.add_scalar('training IoU per batch', iou / dataloader.batch_size, epoch * np.ceil(len(dataloader.dataset))/dataloader.batch_size + num_batch)

    return iou_total, loss.item()


def test(model, dataloader, criterion, device):
    # Set the model to evaluation mode. This will turn off layers that would
    # otherwise behave differently during training, such as dropout.
    model.eval()

    # Sum the Intersection over Union for every batch
    iou = 0
    loss = 0

    # A context manager is used to disable gradient calculations during inference
    # to reduce memory usage, as we typically don't need the gradients at this point.
    with torch.no_grad():
        for sample_batched in dataloader:
            # Request a batch of images and bounding boxes, convert them into tensors
            # of the correct type, and then send them to the appropriate device.
            inputs = sample_batched['image'].float().to(device)
            bbs = sample_batched['bounding_box'].float().to(device)

            # Perform the forward pass of the model
            outputs = model(inputs)  # Step ①

            # Compute the value of the loss and the Intersection over Union for this batch
            loss += criterion(outputs, bbs).cpu().detach().item()
            iou += np.sum([IoU([outputs[i][0], outputs[i][1], outputs[i][2] - outputs[i][0], outputs[i][3] - outputs[i][1]],
                          [bbs[i][0], bbs[i][1], bbs[i][2] - bbs[i][0], bbs[i][3] - bbs[i][1]]).cpu().detach().item() for i in range(outputs.shape[0])])

    return iou, loss


def train_and_test(model, train_dataloader, test_dataloader, criterion, optimizer, max_epochs, verbose=True, tensorboard=None):

    # Automatically determine the device that PyTorch should use for computation
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # Store the best model
    best_iou = 0

    # Move model to the device which will be used for train and test
    model.to(device)

    # Track the value of the loss function and model IoU across epochs
    history_train = {'loss': [], 'IoU': []}
    history_test = {'loss': [], 'IoU': []}

    # Initialize the tensorboard
    if tensorboard is not None:
        writer = SummaryWriter(f'../data/runs/{tensorboard}')
    else:
        writer = None

    for epoch in range(max_epochs):
        # Run the training loop and calculate the IoU.
        sum_iou, loss = train(model, train_dataloader, criterion, optimizer, device, tensorboard=[writer, epoch])
        iou = sum_iou / len(train_dataloader.dataset) * 100
        loss = loss / train_dataloader.batch_size
        history_train['loss'].append(loss)
        history_train['IoU'].append(iou)

        if tensorboard is not None:
            writer.add_scalar('training loss per epoch', loss, epoch)
            writer.add_scalar('training IoU per epoch', iou, epoch)

        # Do the same for the testing loop
        sum_iou, loss = test(model, test_dataloader, criterion, device)
        iou = sum_iou / len(test_dataloader.dataset) * 100
        loss = loss / len(test_dataloader.dataset)
        history_test['loss'].append(loss)
        history_test['IoU'].append(iou)

        if tensorboard is not None:
            writer.add_scalar('test loss per epoch', loss, epoch)
            writer.add_scalar('test IoU per epoch', iou, epoch)

        if verbose or epoch + 1 == max_epochs:
            print(f'[Epoch {epoch + 1}/{max_epochs}]'
                  f" loss: {history_train['loss'][-1]:.4f}, IoU: {history_train['IoU'][-1]:2.2f}%"
                  f" - test_loss: {history_test['loss'][-1]:.4f}, test_IoU: {history_test['IoU'][-1]:2.2f}%")

        if verbose or epoch % 5 == 0:
            writer.add_figure(f'predictions vs. actuals, epoch : {epoch}',
                                  plot_bouding_box(model, train_dataloader, device),
                                  global_step=epoch)

        if tensorboard is not None and history_test['IoU'][-1] > best_iou:
            best_iou = history_test['IoU'][-1]
            torch.save(model.state_dict(), f'../models/{tensorboard}')

    # Generate diagnostic plots for the loss and accuracy
    fig, axes = plt.subplots(ncols=2, figsize=(9, 4.5))
    for ax, metric in zip(axes, ['loss', 'IoU']):
        ax.plot(history_train[metric])
        ax.plot(history_test[metric])
        ax.set_xlabel('epoch', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.legend(['Train', 'Test'], loc='best')
    plt.show()

    writer.close()

    return model.load_state_dict(torch.load(f'../models/{tensorboard}'))


def plot_bouding_box(model, dataloader, device):

    model.eval()

    batch_sample = next(iter(dataloader))
    batch_img = batch_sample['image']
    batch_bbox = batch_sample['bounding_box'].float()

    with torch.no_grad():
        inputs = batch_img.float().to(device)
        pred_bb = model(inputs).cpu()

    mean_nm = [0.485, 0.456, 0.406]
    std_nm = [0.229, 0.224, 0.225]

    fig = plt.figure(figsize=(48, 12))
    for idx in range(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])

        img2 = batch_img[idx].cpu().numpy()
        b, h, w = img2.shape
        img2 = np.transpose(img2, (1, 2, 0))

        img2[:, :, 0] = (img2[:, :, 0]*std_nm[0] + mean_nm[0])*255
        img2[:, :, 1] = (img2[:, :, 1]*std_nm[1] + mean_nm[1])*255
        img2[:, :, 2] = (img2[:, :, 2]*std_nm[2] + mean_nm[2])*255
        img2 = np.ascontiguousarray(img2, dtype=np.uint8)

        x1_gt, y1_gt, x2_gt, y2_gt = batch_bbox[idx].cpu().numpy()
        x1_gt = int(x1_gt * w)
        y1_gt = int(y1_gt * h)
        x2_gt = int(x2_gt * w)
        y2_gt = int(y2_gt * h)

        cv.rectangle(img2, (x1_gt, y1_gt), (x2_gt, y2_gt), (255, 0, 0), 2)
        cv.putText(img2, "True BB", (x1_gt, y1_gt - 10), 0, 0.5, (255, 0, 0), 2)

        x1_pred, y1_pred, x2_pred, y2_pred = pred_bb[idx].cpu().numpy()
        x1_pred = int(x1_pred * w)
        y1_pred = int(y1_pred * h)
        x2_pred = int(x2_pred * w)
        y2_pred = int(y2_pred * h)

        gt = (x1_gt, y1_gt, x2_gt - x1_gt, y2_gt - y1_gt)
        pred = (x1_pred, y1_pred, x2_pred - x1_pred, y2_pred - y1_pred)

        iou = IoU(gt, pred)*100
        cv.putText(img2, f"IoU : {iou}", (10, 15), 0, 0.5, (255, 255, 0), 2)

        cv.rectangle(img2, (x1_pred, y1_pred), (x2_pred, y2_pred), (0, 0, 255), 2)
        cv.putText(img2, "Predicted BB", (x1_pred, y2_pred + 20), 0, 0.5, (0, 0, 255), 2)
        #cv.putText(img2, f"x1 : {x1_pred}, y1 : {y1_pred}", (10, 215), 0, 0.5, (0, 255, 0), 2)
        #cv.putText(img2, f"x2 : {x2_pred}, y2 : {y2_pred}", (10, 190), 0, 0.5, (0, 255, 0), 2)
        ax.imshow(img2)

    return fig







