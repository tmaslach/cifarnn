import sys
import os
import time
import pickle

import torch
from torch import nn
from torch import optim
from torch.nn.modules import loss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import optimizer
import torch.utils.data

sys.path.append(os.path.dirname(__file__))
from torchhacks import FastDataLoader

import torchvision
import torchvision.transforms as transforms

import numpy as np

from tqdm import tqdm
from tensorboardX import SummaryWriter

class ObjectClassifier(nn.Module):
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

def test (test_loader, loss_function, writer=None, counter=None):
    total = 0
    correct = 0
    accumulated_loss = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 3, 32, 32)
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            predictions = torch.max(outputs, 1)[1]

            if total == 0: # First time?
                log_images (writer, images, labels, predictions, counter)

            predictions = predictions.to(device)
            correct += (predictions == labels).sum().float()
            total += len(labels)
            accumulated_loss += loss_function(outputs, labels).item()

    accuracy = (correct / total) * 100
    loss     = accumulated_loss / len(test_loader)

    if writer:
        writer.add_scalar ("Acc/test", accuracy, counter)
        writer.add_scalar ("Loss/test", loss, counter)
    
    return accuracy, loss 

def log_images (writer, images, labels, predictions, counter):
    num_items = int(len(labels) * .1)
    for index, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
        prediction = labelIndexToName[int(prediction.cpu())]
        label = labelIndexToName[int(label.cpu())]
        writer.add_image(f"Image #{index}_{counter} - Guess: {prediction}, Actual: {label}", 
            image, counter)
        if index > num_items:
            break

class NNReport:
    optimizer_class = None
    optimizer_class_default_args = None
    loss_function_class = None
    train_data = None
    test_data = None

    def __init__ (self, network):
        self.network = network

    def train (self, num_epochs, train_data, test_data, optimizer, loss_function, print_every=50, 
                     add_texts=[]):
        """Add_texts is a tuple of tuples (or lists of lists): (("Tag", "description")) 
        """
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.loss_function = loss_function

        with self._track_run () as run_tracker:
            for tag, text in add_texts:
                self._add_text(run_tracker.writer, tag, text)

            for epoch in range(1, num_epochs+1):
                with self._track_epochs(num_epochs) as epoch_tracker:
                    self._train_epoch(epoch, )                    
                    
    def _track_run (self):
        writer = SummaryWriter()
        self._add_text(writer, "Model", self.network)
        class RunTracker:
            start_time = time.time()
            writer = writer
            epoch_count = 0
        tracker = RunTracker()

        yield tracker

        print ("Time Taken:", time.time() - tracker.start_time)
        tracker.writer.close()
        
    def _track_epoch (self):
        class EpochTracker:
            start_time = time.time()
            total_epochs = 0
            loss = 0
            correct = 0
        tracker = EpochTracker()

        yield tracker



    def train_ (num_epochs, print_every):
        counter = 0

        for epoch
        for epoch_step, (images, labels) in enumerate(tqdm(train_loader, desc = f"Epoch {epoch}", leave=False), start=1):
            images = images.view(-1, 3, 32, 32)
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            counter += 1
            if counter % print_every == 0 or epoch_step == len(train_loader):
                # Calculate Output for Training data
                predictions = torch.max(outputs, 1)[1].to(device)
                correct = (predictions == labels).sum().float()
                total = len(labels)
                train_accuracy = correct / total * 100
                writer.add_scalar ("Acc/train", train_accuracy, counter)
                writer.add_scalar ("Loss/train", loss.item(), counter)
                test (test_loader, loss_function, writer=writer, counter=counter)

        counter+= 1
        test_accuracy, test_loss = test(test_loader, loss_function, writer=writer, counter=counter)

        duration = time.time() - start
        writer.add_scalar ("Epoch Durations", duration, epoch)
        print (f"Epoch {epoch}: Complete: trainAccuracy = {train_accuracy:.2f}%,",
                                        f"trainLoss = {epoch_loss/len(train_loader):.3f},",
                                        f"testAccuracy = {test_accuracy:.2f}%,",
                                        f"testLoss = {test_loss:.3f},",
                                        f"duration = {duration:.2f}s")

    def _add_text(self, writer, tag, text):
        text = self._convert_to_markdown(str(text))
        writer.add_text(tag, text)
        
    def _convert_to_markdown(self, text):
        return text.replace("\n", "   \n")

    def _epoch_range (self, num_epochs):
        self.
if __name__ == "__main__": 
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu:0")
    print (f"Running on {device}...")

    image_transform = transforms.Compose([
        transforms.ToTensor()                                 
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
        labelIndexToName = pickle.load(file, encoding="bytes")[b"label_names"]
        labelIndexToName = {i:str(labelIndexToName[i],encoding="ascii") for i in range(len(labelIndexToName))}
 
    train_loader = FastDataLoader(train_set, pin_memory=True, batch_size=200, num_workers=2)
    test_loader = FastDataLoader(test_set, pin_memory=True, batch_size=200, num_workers=2)

    model = ObjectClassifier().to(device)

    # OptimizeGenerator can be created - it's focus is just what RunBuilder does
    optimizer = optim.Adam(self.network.get_parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    nn = NNRunner(model)
    nn.train(1, train_loader, test_loader, optimizer, loss_function,
        add_texts=(("Transform", image_transform),))