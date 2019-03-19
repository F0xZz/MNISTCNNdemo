import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.linspace(-5,5,200)
# x data (tensor),shape=(100,1)
x=Variable(x)
x_np= x.data.numpy()
#x.data is tensor
y_relu= F.relu(x).data.numpy()
y_sigmod=F.torch.sigmoid(x).data.numpy()
y_tanh=F.torch.tanh(x).data.numpy()
y_softplus =F.softplus(x).data.numpy()

plt.figure(1,figsize=(8,6))
plt.subplot(221)
plt.plot(x_np,y_relu,c='red',label='relu')
plt.ylim(-1,5)
plt.legend(loc='best')
plt.show()
