import torch
import torch.nn as nn
import torchvision
from src.metrics.model_performance import IoU
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# All the different feature extractor available
model_builder = {'resnet18'     : [lambda:torchvision.models.resnet18(pretrained=True), 150_528], #25_088
                 'resnet34'     : [lambda:torchvision.models.resnet34(pretrained=True), 25_088],
                 'resnet50'     : [lambda:torchvision.models.resnet50(pretrained=True), 100_352],
                 'resnet152'    : [lambda:torchvision.models.resnet152(pretrained=True), 100_352],
                 'densenet121'  : [lambda:torchvision.models.densenet121(pretrained=True), 50_176],
                 'squeezenet1_1': [lambda:torchvision.models.squeezenet1_1(pretrained=True), 86_528]}


class FeatureExtractor(nn.Module):

    def __init__(self, model_name):
        super(FeatureExtractor, self).__init__()

        # Load the model
        model = model_builder[model_name][0]()
        self.num_features = model_builder[model_name][1]

        # Freeze the network. We don't want to update this part of the model
        for param in model.parameters():
            param.requires_grad = False

        # We keep only the feature maps of all the models
        if 'resnet' in model_name:
            self.body = nn.Sequential(*list(model.children())[:-2])
        elif 'densenet' in model_name or 'squeezenet' in model_name:
            self.body = model.features

    def forward(self, x):
        return self.body(x)


class Swimnet(nn.Module):

    def __init__(self, feature_extractor_name):
        super(Swimnet, self).__init__()

        # Initialize the feature extractor
        self.feature_extractor = FeatureExtractor(feature_extractor_name)

        # Initialize the head bbox
        self.head = nn.Sequential(nn.Dropout(),
                                       nn.Linear(self.feature_extractor.num_features, 128), nn.ReLU(),
                                       nn.BatchNorm1d(128),
                                       nn.Dropout(),
                                       nn.Linear(128, 4), nn.Sigmoid())

    def forward(self, x):

        # Flatten the inputs
        features = x.view(x.size()[0], -1)

        bbox = self.head(features)

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
        iou = np.sum([IoU(outputs[i], bbs[i]).cpu().detach().item() for i in range(outputs.shape[0])])
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
            iou += np.sum([IoU(outputs[i], bbs[i]).cpu().detach().item() for i in range(outputs.shape[0])])

    return iou, loss


def train_and_test(model, train_dataloader, test_dataloader, criterion, optimizer, max_epochs, verbose=True, tensorboard=None):

    # Automatically determine the device that PyTorch should use for computation
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

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

    return model











