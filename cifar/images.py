import pickle

import torch
import torchvision
import torchvision.transforms as transforms

class CifarImages:
    """Syntactic Sugar for accessing Cifar image. Can easily create a chunk of validation
       images, and have one place for anything Cifar image related.
       train_set, test_set, and optionally validation_set contain images"""

    train_image_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4), 
        transforms.ToTensor(),
        #transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, 5))
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_image_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, 5))
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def __init__ (self, validation_percentage=0):
        self.train_set = torchvision.datasets.CIFAR10(
            root      = "./data",
            train     = True,
            transform = self.train_image_transform,
            download  = True,
        )

        self.test_set = torchvision.datasets.CIFAR10(
            root      = "./data",
            train     = False,
            transform = self.test_image_transform,
            download  = True,
        )

        self.validation_set = None
        if validation_percentage > 0:
            num_train_subset = int(len(self.train_set) * (1-validation_percentage))
            num_val_subset = len(self.train_set) - num_train_subset
            self.train_set, self.validation_set = torch.utils.data.random_split(
                self.train_set, [num_train_subset, num_val_subset])

    @staticmethod        
    def get_label_names():
        with open ("./data/cifar-10-batches-py/batches.meta", 'rb') as file:
            return pickle.load(file, encoding="utf-8")["label_names"]
