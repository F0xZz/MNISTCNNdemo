import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100) #
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1),0).type(torch.LongTensor)


x,y = Variable(x),Variable(y)

# another way method
net2 = torch.nn.Sequential{
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2),
}
print(net2)

optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
#loss_func = torch.nn.MSELoss() #outputvalue [0,1][1,0]
#use MSELoss for regression problem
loss_func = torch.nn.CrossEntropyLoss()#outputvalue is transform to this [0.1,0.2,0.7]


#real time print function
for t in range(100):
    out = net(x)

    loss = loss_func(out,y)#prediction is first
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t%1 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out),1)[1]
        #out is a number and use softmax will be the possibilities
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt. scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pred_y,s=100,lw=0)
        accuracy = sum(pred_y==target_y)/200
        plt.text(1.5, -4, 'accuracy=%.2f' %accuracy,fontdict={'size':20})
        plt.pause(0.5)
plt.ioff()
plt.show()
