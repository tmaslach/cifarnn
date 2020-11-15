import sys
import os
import pickle

import torch
from torch import nn
from torch import optim
from torch.nn.modules import loss
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.utils.data

sys.path.append(os.path.dirname(__file__))
from torchhacks import FastDataLoader

import torchvision
import torchvision.transforms as transforms

import numpy as np

from tensorboardreport import TenserBoardReport

class Conv3ResidualBlock (nn.Sequential):
    def __init__ (self, in_channels, out_channels, stride, repeat):
        assert out_channels % 4 == 0

        super().__init__()

        self._downsample = None
        self._out_channels = out_channels
        if stride != 1 or in_channels != out_channels:
            self._downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1 , stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        intermediate = out_channels // 4
        self.resblocks = nn.ModuleList()
        for _ in range(repeat):
            self.resblocks.append (nn.Sequential(
                nn.Conv2d (in_channels, intermediate, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(intermediate),
                nn.ReLU(),
                nn.Conv2d (intermediate, intermediate, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(intermediate),
                nn.ReLU(),
                nn.Conv2d (intermediate, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
            ))

            # Only first layer uses in_channels defined by user.  Others follow the last out
            in_channels = out_channels
            # Same with stride - only first first block in 'repeat'
            stride = 1

        self.relu = nn.ReLU()

    def forward (self, x):
        identity = self._downsample(x) if self._downsample else x

        for resblock in self.resblocks:
            x = resblock(x)
            x += identity
            x = self.relu(x) 
        return x


class ObjectNN(nn.Module):
    def __init__ (self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.resnet_layer1 = Conv3ResidualBlock(64, 256, stride=1, repeat=3)
        self.resnet_layer2 = Conv3ResidualBlock(256, 512, stride=2, repeat=4)
        self.resnet_layer3 = Conv3ResidualBlock(512, 1024, stride=2, repeat=6)
        self.resnet_layer4 = Conv3ResidualBlock(1024, 2048, stride=2, repeat=3)

        #self.avg_pool = nn.AvgPool2d(7, stride=2)
        #self.dropout = nn.Dropout2d(p=.5, inplace=True)

        self.linear = nn.Linear(2048, 10)

    def forward (self, x):
        x = self.conv1(x)
        x = self.resnet_layer1(x)
        x = self.resnet_layer2(x)
        x = self.resnet_layer3(x)
        x = self.resnet_layer4(x)
        #x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def train (num_epochs, train_loader, test_loader, optimizer, loss_function, report):
    for epoch in report.track_epochs(range (1, num_epochs+1)):
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

            report.set_batch_train_loss(loss.item())
            report.set_batch_train_accuracy(correct / len(labels))

            if report.is_print_step():
                test (test_loader, loss_function, report)

def test (test_loader, loss_function, report):
    with torch.no_grad():
        for images, labels in report.track_test_batches(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = loss_function(outputs, labels)

            predictions = torch.max(outputs, 1)[1].to(device)
            correct_flags  = (predictions == labels)
            num_correct = correct_flags.sum().float()

            report.set_batch_test_loss(loss.item())
            report.set_batch_test_accuracy(num_correct / len(labels))
            report.add_batch_test_confusion_data (predictions, labels)
            report.add_incorrect_test_images (correct_flags, images)

if __name__ == "__main__": 
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu:0")
    print (f"Running on {device}...")

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, 5))                                 
    ])

    train_set = torchvision.datasets.CIFAR10(
        root      = "./data",
        train     = True,
        transform = image_transform,
        download  = True,
    )

    test_set = torchvision.datasets.CIFAR10(
        root      = "./data",
        train     = False,
        transform = image_transform,
        download  = True,
    )
    with open ("./data/cifar-10-batches-py/batches.meta", 'rb') as file:
        label_names = pickle.load(file, encoding="utf-8")["label_names"]
 
    train_loader = FastDataLoader(train_set, pin_memory=True, batch_size=100, num_workers=2)
    test_loader = FastDataLoader(test_set, pin_memory=True, batch_size=100, num_workers=2)

    model = ObjectNN().to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00838)

    report = TenserBoardReport(label_names=label_names, report_every=50)
    report.add_text("Model Topology", model)
    report.add_text("Transform", image_transform)
    
    train (20, train_loader, test_loader, optimizer, loss_function, report)

