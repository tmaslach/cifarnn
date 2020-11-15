import sys
import os
import pickle

import torch
from torch import nn, optim
import torch.utils.data

import torchvision
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(__file__))
from torchhacks import FastDataLoader
from tensorboardreport import TenserBoardReport

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
            from torch.nn import functional as F
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

def train (num_epochs, model, train_loader, test_loader, optimizer, loss_function, report, lr_scheduler):
    avg_accuracy = 0
    best_avg_accuracy = 0
    for epoch in report.track_epochs(range (1, num_epochs+1)):
        print ("lr used:", lr_scheduler.get_last_lr())
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
                avg_accuracy, avg_loss = test (model, test_loader, loss_function, report)

        if avg_accuracy > best_avg_accuracy:
            best_avg_accuracy = avg_accuracy
            data_to_save = dict(
                epoch = epoch,
                state_dict = model.state_dict(),
                best_accuracy = best_avg_accuracy)
            # Should be in report class!
            torch.save(data_to_save, os.path.join(report.writer.logdir, f"epoch_{epoch}.pkl"))

        lr_scheduler.step()
    return avg_accuracy

def test (model, test_loader, loss_function, report):
    accum_loss = 0
    accum_accuracy = 0
    with torch.no_grad():
        for images, labels in report.track_test_batches(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = loss_function(outputs, labels)

            predictions = torch.max(outputs, 1)[1].to(device)
            correct_flags  = (predictions == labels)
            num_correct = correct_flags.sum().float()

            accuracy = num_correct / len(labels)
            report.set_batch_test_loss(loss.item())
            report.set_batch_test_accuracy(accuracy)
            report.add_batch_test_confusion_data (predictions, labels)
            report.add_incorrect_test_images (correct_flags, images)

            accum_loss += loss.item()
            accum_accuracy += accuracy
        
        return accum_accuracy / len(test_loader) * 100, accum_loss / len(test_loader)

def train_cifar(train_set, test_set):
    train_loader = FastDataLoader(train_set, pin_memory=True, batch_size=128, num_workers=2, shuffle=True)
    test_loader = FastDataLoader(test_set, pin_memory=True, batch_size=128, num_workers=2, shuffle=False)

    model = ResnetNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    #optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        milestones=[100, 150])
    report = TenserBoardReport(label_names=label_names, report_every=5000)
    report.add_text("Model Topology", model)
    report.add_text("Transform", image_transform)
    
    train (200, model, train_loader, test_loader, optimizer, loss_function, report, lr_scheduler)

if __name__ == "__main__": 
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu:0")
    print (f"Running on {device}...")

    image_transform = [
        transforms.ToTensor(),
        #transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, 5))
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]

    train_set = torchvision.datasets.CIFAR10(
        root      = "./data",
        train     = True,
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4)] + 
            image_transform),
        download  = True,
    )

    test_set = torchvision.datasets.CIFAR10(
        root      = "./data",
        train     = False,
        transform = transforms.Compose(image_transform),
        download  = True,
    )

    #num_train_subset = int(len(train_set) * .8)
    #num_val_subset = len(train_set) - num_train_subset
    #train_set, val_set = torch.utils.data.random_split(train_set, 
    #    [num_train_subset, num_val_subset])

    with open ("./data/cifar-10-batches-py/batches.meta", 'rb') as file:
        label_names = pickle.load(file, encoding="utf-8")["label_names"]
 
    train_cifar(train_set, test_set)