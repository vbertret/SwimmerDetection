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
                 'resnet34'     : [lambda:torchvision.models.resnet34(pretrained=True), None],
                 'resnet50'     : [lambda:torchvision.models.resnet50(pretrained=True), 2048],
                 'resnet152'    : [lambda:torchvision.models.resnet152(pretrained=True), None],
                 'densenet121'  : [lambda:torchvision.models.densenet121(pretrained=True), None],
                 'squeezenet1_1': [lambda:torchvision.models.squeezenet1_1(pretrained=True), None],
                 'mobilenet-v3-large': [lambda:torchvision.models.mobilenet_v3_large(pretrained=True), 960],
                 'mobilenet-v3-small': [lambda:torchvision.models.mobilenet_v3_small(pretrained=True), 576],
                 'mobilenet-v2': [lambda:torchvision.models.mobilenet_v2(pretrained=True), 1280]}


class FeatureExtractor(nn.Module):
    """
    Class for the feature extractor

    Create a class for the feature extractor. The output of the feature
    extractor is a flat vector. Indeed, the global pooling have been already
    done.

    Attributes
    ----------
    num_features : int
        number of features at the end of the feature extractor
    body : nn.Sequential
        the feature extractor
    """

    def __init__(self, model_name):
        """
        Attribute
        ---------
        model_name : str
            one the model name chosen in the model_builder dict
        """
        super(FeatureExtractor, self).__init__()

        # Load the model
        model = model_builder[model_name][0]()
        self.num_features = model_builder[model_name][1]

        # We can if we want freeze the model and don't update the weights
        for param in model.parameters():
            param.requires_grad = True

        # We kremove the classfication layer
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
    """
    Deep Neural Network

    The class implements a deep neural network in order to find the coordinate
    of a bounding box which contains the swimmer.

    Attributes
    ----------
    feature_extractor : nn.Sequential
        the feature extractor
    head : nn.Sequential
        the regression head in order to find the coordinates
    detect_surface : boolean
        if true, the method uses the detection of the surface
    use_time : boolean
        if true, the method uses the precedent box to make the prediction
    
    """

    def __init__(self, feature_extractor_name):
        """
        Parameter
        ---------
        feature_extractor_name : str
            one the model name chosen in the model_builder dict
        """
        super(Swimnet, self).__init__()

        # Initialize the feature extractor
        self.feature_extractor = FeatureExtractor(feature_extractor_name)

        # Initialize the head bbox
        self.head = nn.Sequential(
                              nn.Linear(self.feature_extractor.num_features, 512), nn.ReLU(),
                              nn.Linear(512, 4), nn.Sigmoid()) 

        #Set the variables for the use of the time and the detection of surface
        self.detect_surface = False
        self.use_time = False

    def forward(self, x):

        # Extraction of the features
        features = self.feature_extractor(x)
        
        # Flatten the features
        features = features.view(features.size()[0], -1)

        # Regression of the coordinates
        bbox = self.head(features)

        return bbox

    def predict(self, file_name, a=0, b=0, precBB=[]):
        """
        Prediction of the Color Segmentation

        This method makes the prediction of the bounding box for one image

        Parameters
        -----------
        file_name : str
            the name of the image to use for the prediction
        a : float
            if different from 0, it's the slope of the surface line ( default is 0 )
        b : float
            if different from 0, it's the intercept of the surface line ( default is 0 )
        precBB : [x, y, w, h] list
            the coordinates of the bounding box found at the previous frame ( default is [] )

        Returns
        -------
        bbox : [x, y, w, h] list
            the coordinates of the bounding box
        """

        # Declaration of the mean and standard deviation of the dataset
        mean_nm = [0.485, 0.456, 0.406]
        std_nm = [0.229, 0.224, 0.225]

        # Transformations on the images
        input_size = 224
        transformations = transforms.Compose([
            Rescale((input_size, input_size)),
            ToTensor(),
            Normalize(mean_nm, std_nm)
        ])

        # Load the picture and make the transformations
        img = io.imread(file_name)
        sample = {'image': img, 'bounding_box': np.array([0, 0, 0, 0])}
        sample = transformations(sample)
        x = sample['image']
        x = torch.unsqueeze(x, 0).float()

        # Compute the prediction of the deep neural network
        self.eval()
        bbox = self.forward(x)[0].detach().numpy()

        # Convert the bouding box type from [x1, y1, x1, y2] to [x, y, w, h] and change the range
        bbox = [int(bbox[0]*640), int(bbox[1]*480), int((bbox[2] - bbox[0])*640), int((bbox[3] - bbox[1])*480)]

        return bbox


def train(model, dataloader, criterion, optimizer, device, tensorboard=(None, 0)):
    """
    Train the network

    Parameters
    ----------
    model : Swimnet
        the network
    dataloader : torch.utils.data.DataLoader
        the dataloader with all the data for the training
    criterion : torch.nn
        the loss function to be minimized
    optimizer : torch.optim
        the optimizer to minimize the cost function
    device : torch.device
        the device on which you want to train the model
    tensorboard : torch.utils.tensorboard.SummaryWriter
        If is different from None, the training results are saved into tensorboard environnement
    
    Returns
    -------
    iou_total : float
        the sum of the IoU of all the images
    loss : float
        the loss of the last batch
    """
    # Set the model to training mode.
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

        # Add data to tensorboard if the number of batch is divisible by 10
        if tensorboard[0] is not None and num_batch % 10 == 0:
            writer_cop.add_scalar('training loss per batch', loss / dataloader.batch_size, epoch * np.ceil(len(dataloader.dataset))/dataloader.batch_size + num_batch)
            writer_cop.add_scalar('training IoU per batch', iou / dataloader.batch_size, epoch * np.ceil(len(dataloader.dataset))/dataloader.batch_size + num_batch)

    return iou_total, loss.item()


def test(model, dataloader, criterion, device):
    """
    Test the network

    Parameters
    ----------
    model : Swimnet
        the network
    dataloader : torch.utils.data.DataLoader
        the dataloader with all the data for the testing
    criterion : torch.nn
        the loss function to be minimized
    device : torch.device
        the device on which you want to train the model
    
    Returns
    -------
    iou : float
        the sum of the IoU of all the images
    loss : float
        the loss of all the images
    """

    # Set the model to evaluation mode
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
    """
    Train and test the network for max_epochs times.

    Parameters
    ----------
    model : Swimnet
        the network
    train_dataloader : torch.utils.data.DataLoader
        the dataloader with all the data for the training
    test_dataloader : torch.utils.data.DataLoader
        the dataloader with all the data for the testing
    criterion : torch.nn
        the loss function to be minimized
    optimizer : torch.optim
        the optimizer to minimize the cost function
    max_epochs : int
        the maximal number of epochs
    verbose : bool
        if true, a summary of each epoch is printed ( default is True )
    tensorboard : str
        If is different from None, it's the name the file which contains the results
    
    Returns
    -------
    model : Swimnet
        the best model which was founded
    """

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

        # If tensorboard is specified, add data to tensorboard
        if tensorboard is not None:
            writer.add_scalar('test loss per epoch', loss, epoch)
            writer.add_scalar('test IoU per epoch', iou, epoch)

        # If verbose is true, print a summary of the epoch
        if verbose or epoch + 1 == max_epochs:
            print(f'[Epoch {epoch + 1}/{max_epochs}]'
                  f" loss: {history_train['loss'][-1]:.4f}, IoU: {history_train['IoU'][-1]:2.2f}%"
                  f" - test_loss: {history_test['loss'][-1]:.4f}, test_IoU: {history_test['IoU'][-1]:2.2f}%")

        # If tensorboard is specified, add data to tensorboard
        if tensorboard is not None and epoch % 5 == 0:
            writer.add_figure(f'predictions vs. actuals, epoch : {epoch}',
                                  plot_bouding_box(model, train_dataloader, device),
                                  global_step=epoch)

        # If tensorboard is specified, save the best model during the training
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
    """
    Compute the predicted bounding box of the model and plot it for 4 pictures.
    
    Parameters
    ----------
    model : Swimnet
        the network
    dataloader : torch.utils.data.DataLoader
        the dataloader with all the data
    device : torch.device
        the device on which you want to train the model
    
    Returns
    -------
    fig : matplotlib.pyplot.figure
        the figure with the 4 pictures annotated with the predicted and true bounding box
    """

    # Set the model to eval mode
    model.eval()

    # Recovering some data
    batch_sample = next(iter(dataloader))
    batch_img = batch_sample['image']
    batch_bbox = batch_sample['bounding_box'].float()

    # Compute the predcited bounding box for the data
    with torch.no_grad():
        inputs = batch_img.float().to(device)
        pred_bb = model(inputs).cpu()

    # Declaration of the mean and the standard deviation which was applied on the dataset
    mean_nm = [0.485, 0.456, 0.406]
    std_nm = [0.229, 0.224, 0.225]

    # Initialization of the figure
    fig = plt.figure(figsize=(24, 6))
    for idx in range(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])

        # Reshape the image
        img2 = batch_img[idx].cpu().numpy()
        b, h, w = img2.shape
        img2 = np.transpose(img2, (1, 2, 0))

        # Remove the normalization of the image
        img2[:, :, 0] = (img2[:, :, 0]*std_nm[0] + mean_nm[0])*255
        img2[:, :, 1] = (img2[:, :, 1]*std_nm[1] + mean_nm[1])*255
        img2[:, :, 2] = (img2[:, :, 2]*std_nm[2] + mean_nm[2])*255
        img2 = np.ascontiguousarray(img2, dtype=np.uint8)

        # Change the range of the coordinates of the ground truth bounding boxes
        x1_gt, y1_gt, x2_gt, y2_gt = batch_bbox[idx].cpu().numpy()
        x1_gt = int(x1_gt * w)
        y1_gt = int(y1_gt * h)
        x2_gt = int(x2_gt * w)
        y2_gt = int(y2_gt * h)

        cv.rectangle(img2, (x1_gt, y1_gt), (x2_gt, y2_gt), (255, 0, 0), 2)
        cv.putText(img2, "True BB", (x1_gt, y1_gt - 10), 0, 0.5, (255, 0, 0), 2)

        # Change the range of the coordinates of the predicted bounding boxes
        x1_pred, y1_pred, x2_pred, y2_pred = pred_bb[idx].cpu().numpy()
        x1_pred = int(x1_pred * w)
        y1_pred = int(y1_pred * h)
        x2_pred = int(x2_pred * w)
        y2_pred = int(y2_pred * h)

        # Compute the IoU
        gt = (x1_gt, y1_gt, x2_gt - x1_gt, y2_gt - y1_gt)
        pred = (x1_pred, y1_pred, x2_pred - x1_pred, y2_pred - y1_pred)
        iou = IoU(gt, pred)*100
        cv.putText(img2, f"IoU : {iou}", (10, 15), 0, 0.5, (255, 255, 0), 2)

        cv.rectangle(img2, (x1_pred, y1_pred), (x2_pred, y2_pred), (0, 0, 255), 2)
        cv.putText(img2, "Predicted BB", (x1_pred, y2_pred + 20), 0, 0.5, (0, 0, 255), 2)
        ax.imshow(img2)

    return fig







