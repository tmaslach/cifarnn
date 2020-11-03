import sys
import os
import pickle

import torch
from torch import nn
from torch import optim
from torch.nn.modules import loss
import torch.utils.data

sys.path.append(os.path.dirname(__file__))
from torchhacks import FastDataLoader

import torchvision
import torchvision.transforms as transforms

import numpy as np

from tensorboardreport import TenserBoardReport

class ObjectNN(nn.Module):
    def __init__ (self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d (3, 32, 4),
            nn.BatchNorm2d (32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d (32, 64, 4),
            nn.BatchNorm2d (64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.linear = nn.Sequential(
            nn.Linear(64*5*5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward (self, x):
        x = self.conv(x)
        x = x.view (x.size(0), -1)
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
 
    train_loader = FastDataLoader(train_set, pin_memory=True, batch_size=200, num_workers=2)
    test_loader = FastDataLoader(test_set, pin_memory=True, batch_size=200, num_workers=2)

    model = ObjectNN().to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    report = TenserBoardReport(label_names=label_names, report_every=50)
    report.add_text("Model Topology", model)
    report.add_text("Transform", image_transform)
    
    train (10, train_loader, test_loader, optimizer, loss_function, report)

