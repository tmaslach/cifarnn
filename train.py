import sys, os
import argparse

this_folder = os.path.dirname(__file__)
sys.path.append(this_folder)
from cifar import nn

if __name__ == "__main__":
    print (f"Running on {nn.device}...")
    cifar = nn.CifarImages()
    nn.train(cifar.train_set, cifar.test_set)