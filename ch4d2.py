import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

EPOCH = 1 #train the train N time ,to save
BATCH_SIZE = 64  #amount sample
TIME_STEP = 28   #image height rnn time Step
TIME_SIZE = 28   # RNN input size (image width)
LR = 0.01
DOWNLOAD_MNIST = False


train_data  = dsets.MNIST(root='./mnist/',train = True ,
                transforms = transforms.ToTensor(),download = False)
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                batch_size= BATCH_SIZE,shuffle=True)
