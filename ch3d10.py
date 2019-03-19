import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt



x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
# x data (tensor),shape = (100,1) 100 -1,1
y = x.pow(2)+0.2*torch.rand(x.size())
# noisy y data (tensor),shape
#x^2+noisy rand value
x,y = Variable(x),Variable(y)

#plt.scatter(x.data.numpy(),y.data.numpy())
#plt.show()
class Net(torch.nn.Module):#offical method to use
    def __init__(self, n_features , n_hidden , n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
        pass
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
plt.ion()
plt.show()

net = Net(1,30,1)
#input source =1 hidden layer num=10 output layer =1
print(net)
optimizer = torch.optim.SGD(net.parameters(),lr=0.3)
loss_func = torch.nn.MSELoss()
#real time print function
for t in range(100):
    prediction = net(x)

    loss = loss_func(prediction,y)#prediction is first
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t%5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        #real randon value scatter point
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-')
        #print linear in the prediction way
        plt.text(0.5,0,'Loss=%.4f'% loss.data[0],fontdict={'size':30})
        plt.pause(0.1)
plt.ioff()
plt.show()
