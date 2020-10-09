import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.core.quant import QuantType

# Load dataset

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class QuantLeNet(Module):
    def __init__(self):
        super(QuantLeNet, self).__init__()
        self.conv1 = qnn.QuantConv2d(3, 6, 5,
                                     weight_quant_type=QuantType.INT,
                                     weight_bit_width=8)
        self.relu1 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)
        self.conv2 = qnn.QuantConv2d(6, 16, 5,
                                     weight_quant_type=QuantType.INT,
                                     weight_bit_width=8)
        self.relu2 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)
        self.fc1   = qnn.QuantLinear(16*5*5, 120, bias=True,
                                     weight_quant_type=QuantType.INT,
                                     weight_bit_width=8)
        self.relu3 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)
        self.fc2   = qnn.QuantLinear(120, 84, bias=True,
                                     weight_quant_type=QuantType.INT,
                                     weight_bit_width=8)
        self.relu4 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)
        self.fc3   = qnn.QuantLinear(84, 10, bias=False,
                                     weight_quant_type=QuantType.INT,
                                     weight_bit_width=8)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out

net = QuantLeNet()

# Define device as GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load Net onto GPU

net.to(device)

# Define Loss and Gradient

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train Net on GPU

for epoch in range(2):  # loop over the dataset multiple times
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data[0].to(device), data[1].to(device)
		# zero the parameter gradients
		optimizer.zero_grad()
		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		# print statistics
		running_loss += loss.item()
		if i % 2000 == 1999:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print('Finished Training on GPU')

# Test Net on GPU

correct = 0
total = 0
with torch.no_grad():
	for data in testloader:
		inputs, labels = data[0].to(device), data[1].to(device)
		outputs = net(inputs)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

#  Show Class Performance

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
	for data in testloader:
		inputs, labels = data[0].to(device), data[1].to(device)
		outputs = net(inputs)
		_, predicted = torch.max(outputs, 1)
		c = (predicted == labels).squeeze()
		for i in range(4):
			label = labels[i]
			class_correct[label] += c[i].item()
			class_total[label] += 1
