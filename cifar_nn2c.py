import sys
import os
import pickle

import torch
from torch import nn
from torch import optim
from torch.nn.modules import loss
import torch.utils.data

import optuna

sys.path.append(os.path.dirname(__file__))
from torchhacks import FastDataLoader

import torchvision
import torchvision.transforms as transforms

import numpy as np

class ObjectNN(nn.Module):
    def __init__ (self, num_outputs_linear1, num_outputs_linear2):
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
            nn.Linear(64*5*5, num_outputs_linear1),
            nn.ReLU(),
            nn.Linear(num_outputs_linear1, num_outputs_linear2),
            nn.ReLU(),
            nn.Linear(num_outputs_linear2, 10)
        )

    def forward (self, x):
        x = self.conv(x)
        x = x.view (x.size(0), -1)
        x = self.linear(x)
        return x

def train (model, train_loader, test_loader, optimizer, loss_function, print_every=50):
    for images, labels in train_loader:            
        images = images.view(-1, 3, 32, 32)
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_function(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.max(outputs, 1)[1].to(device)
        correct = (predictions == labels).sum().float()

def test (model, test_loader, loss_function):
    with torch.no_grad():
        num_correct = 0
        loss = 0
        loss_steps = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss += loss_function(outputs, labels)

            predictions = torch.max(outputs, 1)[1].to(device)
            num_correct += (predictions == labels).sum().float()
            loss_steps += 1

        return loss / loss_steps, num_correct / len(test_loader.dataset)


def optimize_cifar(trial, train_set):
    num_epochs = 10

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    num_outputs_linear1 = trial.suggest_int('num_output_linear1', 256, 1024)
    num_outputs_linear2 = trial.suggest_int('num_output_linear2', 64, 512)
    batch_size = trial.suggest_int('batch_size', 1, 200)

    model = ObjectNN(num_outputs_linear1, num_outputs_linear2).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = FastDataLoader(train_set, pin_memory=True, batch_size=batch_size, num_workers=2)
    val_loader = FastDataLoader(val_set, pin_memory=True, batch_size=batch_size, num_workers=2)

    for epoch in range (1, num_epochs+1):
        train (model, train_loader, val_loader, optimizer, loss_function)

        val_loss, val_accuracy = test (model, val_loader, loss_function)
        trial.report (val_accuracy, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_accuracy

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
    )

    num_train_subset = int(len(train_set) * .9)
    num_val_subset = len(train_set) - num_train_subset
    train_set, val_set = torch.utils.data.random_split(train_set, 
        [num_train_subset, num_val_subset])

    with open ("./data/cifar-10-batches-py/batches.meta", 'rb') as file:
        label_names = pickle.load(file, encoding="utf-8")["label_names"]
 
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optimize_cifar(trial, train_set), 
        n_trials=1000, n_jobs=1, show_progress_bar=True, gc_after_trial=True)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))