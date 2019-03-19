from __future__ import print_function
import sys, os, argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

print(sys.path[0])
cuda = torch.cuda.is_available()
