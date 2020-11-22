import sys, os
import argparse

this_folder = os.path.dirname(__file__)
sys.path.append(this_folder)
from cifar import nn

parser = argparse.ArgumentParser(
    description="Predict type of object from airplane, automobile, bird, "
                "cat, deer, dog, frog, horse, ship or truck"
)
parser.add_argument('--nnfile', help="Pytorch NN state to use. By default, internal one is used")
parser.add_argument('image', help="Image to be classified")

args = parser.parse_args()

nnfile = args.nnfile
if not nnfile:
    nnfile = os.path.join(this_folder, "default_nn.pt")

nn.predict(nnfile, args.image).print_results()