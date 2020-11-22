import sys
import os
from contextlib import contextmanager

import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data

from .torchhacks import FastDataLoader
from .reports import ImageClassificatioanReport, ValueAverager
from .images import CifarImages

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu:0")
    
class PadLayer (nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return self.lambda_(x)

class Conv2ResidualBlock (nn.Sequential):
    def __init__ (self, in_channels, out_channels, stride, repeat):
        assert out_channels % 4 == 0

        super().__init__()

        for index in range(repeat):
            self.add_module(f"subblock{index}", Conv2ResidualSubBlock(in_channels, out_channels, stride))
            # Only first layer uses in_channels and stride defined by user. 
            in_channels = out_channels
            stride = 1

class Conv2ResidualSubBlock (nn.Module):
    def __init__ (self, in_channels, out_channels, stride):
        assert out_channels % 4 == 0

        super().__init__()

        self.conv1 = nn.Conv2d (in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d (out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = PadLayer(lambda x: F.pad(x[:, :, ::2, ::2], 
                (0, 0, 0, 0, out_channels//4, out_channels//4), "constant", 0))
        else:
            self.shortcut = None

    def forward (self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(input) if self.shortcut else input
        x = self.relu2(x)
        return x


class ResnetNN(nn.Module):
    def __init__ (self):
        super().__init__()

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.layer1 = Conv2ResidualBlock(16, 16, stride=1, repeat=5)
        self.layer2 = Conv2ResidualBlock(16, 32, stride=2, repeat=5)
        self.layer3 = Conv2ResidualBlock(32, 64, stride=2, repeat=5)

        self.avg_pool = nn.AvgPool2d(8)

        self.linear = nn.Linear(64, 10)

        def init_weights(module):
            if isinstance (module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight)
        self.apply(init_weights)

    def forward (self, x):
        x = self.conv_encoder(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    @contextmanager
    def eval_only (self):
        try:
            self.eval()
            with torch.no_grad():
                yield
        finally:
            self.train()

def train_once (model, train_loader, test_loader, optimizer, loss_function, report):
    for images, labels in report.track_train_batches(train_loader):            
        images = images.view(-1, 3, 32, 32)
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_function(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.max(outputs, 1)[1].to(device)
        correct = (predictions == labels).sum().float()

        report.add_batch_train_loss(loss.item())
        report.add_batch_train_accuracy(correct / len(labels))

        if report.is_report_step:
            avg_accuracy, _ = test (model, test_loader, loss_function, report)

def train(train_set, test_set):
    train_loader = FastDataLoader(train_set, pin_memory=True, batch_size=128, num_workers=2, shuffle=True)
    test_loader = FastDataLoader(test_set, pin_memory=True, batch_size=128, num_workers=2, shuffle=False) 
 
    model = ResnetNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    #optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        milestones=[100, 150])

    report = ImageClassificatioanReport(model, 
        label_names  = CifarImages.get_label_names(), 
        report_every = "epoch") 
    report.add_text("Model Topology", model)
    report.add_text("Transform", CifarImages.train_image_transform)

    NUM_EPOCHS = 200
    for epoch in report.track_epochs(range (1, NUM_EPOCHS+1)):
        print ("lr used:", lr_scheduler.get_last_lr())
        train_once (model, train_loader, test_loader, optimizer, loss_function, report)
        lr_scheduler.step()

def test (model, test_loader, loss_function, report):
    with model.eval_only():
        for images, labels in report.track_test_batches(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = loss_function(outputs, labels)

            predictions = torch.max(outputs, 1)[1].to(device)
            correct_flags  = (predictions == labels)
            num_correct = correct_flags.sum().float()

            report.add_batch_test_loss(loss.item())
            report.add_batch_test_accuracy(num_correct / len(labels))
            report.add_batch_test_confusion_data (predictions, labels)

        return report.test_stats.accuracy.average, report.test_stats.loss.average        

class PredictionInfo:
    def __init__ (self, prediction, confidences):
        self.prediction = prediction
        self.confidences = confidences

    def print_results(self):
        for label, confidence in zip(CifarImages.get_label_names(), self.confidences):
            print (f"{label}: {confidence*100:.3f}%")
        print ("Guess is:", CifarImages.get_label_names()[self.prediction])

def predict (model_filename, image_filename):
    saved_data = torch.load(model_filename)
    accuracy = saved_data["accuracy"]
    print (f"Loading model with test accuracy of {accuracy:.3f}%")
    model = ResnetNN()
    model.load_state_dict(saved_data["state_dict"])

    import cv2
    image = cv2.imread(image_filename)
    resizedImage = cv2.resize(image, (32, 32))
    transformedImage = CifarImages.test_image_transform(resizedImage)
    cv2.imshow('Original Image', image)
    #cv2.imshow('Resized', resizedImage)
    cv2.waitKey()

    with model.eval_only():
        transformedImage = transformedImage.view(-1, 3, 32, 32)
        output = model(transformedImage)
        confidences = F.softmax(output, 1)
        prediction = torch.max(confidences, 1)[1]

    return PredictionInfo(prediction, confidences[0])
