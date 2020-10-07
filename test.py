import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import brevitas.nn as qnn
from brevitas.core.quant import QuantType

batch_size=32
img_dimensions = 224

# Normalize to the ImageNet mean and standard deviation
# Could calculate it for the cats/dogs data set, but the ImageNet
# values give acceptable results here.
img_transforms = transforms.Compose([
    transforms.Resize((img_dimensions, img_dimensions)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
    ])
img_test_transforms = transforms.Compose([
    transforms.Resize((img_dimensions,img_dimensions)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
    ])

num_workers = 6

def train(model, optimizer, loss_fn, train_loader, epochs=5, device="cpu"):
    entry = {
            "epochnum" : None,
            "loss" : None
            }
    losses = list()
    for epoch in range(epochs):
        training_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)
        entry.update(epochnum = epoch, loss = training_loss)
        losses.append(dict(entry))
    for loss in losses:
        print("Training Loss for epoch " + str(loss.get("epochnum")) + ": " + str(loss.get("loss")))

def test_model(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Test performance:    correct: {:d}  total: {:d}   accuracy: {:f}'.format(correct, total, correct / total))

def check_image(path):
    try:
        img = Image.open(path)
        return True
    except:
        return False

print("Which dataset should be loaded?")
print("1 = Cats vs Dogs")
print("2 = CIFAR10")

choice = input()

choices = ["1", "2"]

for x in choices:
    if x == "1":
        train_data_path = "/home/lucia/projdata/catsVdogs/train/"
        test_data_path = "/home/lucia/projdata/catsVdogs/test/"
        train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=img_transforms, is_valid_file=check_image)
        test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=img_test_transforms, is_valid_file=check_image)
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        num_classes = 2
    
    elif x == "2":
        trainset = torchvision.datasets.CIFAR10(root='/home/lucia/projdata/CIFAR10', train=True, download=True, transform=img_transforms)
        train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        testset = torchvision.datasets.CIFAR10(root='/home/lucia/projdata/CIFAR10', train=False, download=True, transform=img_transforms)
        test_data_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = 10

    else:
        print("invalid choice")
    
    for y in range(2,3):

        weightQuantType = QuantType.BINARY
        weightBitWidth = 1
        activationQuantType = QuantType.BINARY
        activationBitWidth = 1

        class Net(nn.Module):
            def __init__(self, num_classes = num_classes, weight_quant_type=weightQuantType, weight_bit_width=weightBitWidth, 
                        quant_type = activationQuantType, bit_width = activationBitWidth):
                super(Net, self).__init__()
                self.features = nn.Sequential(
                    qnn.QuantConv2d(3, 96, kernel_size=11, stride=4, padding=0,
                        weight_quant_type=weightQuantType, weight_bit_width=weightBitWidth),
                    qnn.QuantTanh(bit_width=activationBitWidth, quant_type=activationQuantType),
                    nn.BatchNorm2d(96),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    qnn.QuantConv2d(96, 256, kernel_size=5, padding=2,
                        weight_quant_type=weightQuantType, weight_bit_width=weightBitWidth),
                    qnn.QuantTanh(bit_width=activationBitWidth, quant_type=activationQuantType),
                    nn.BatchNorm2d(256),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    qnn.QuantConv2d(256, 384, kernel_size=3, padding=1,
                        weight_quant_type=weightQuantType, weight_bit_width=weightBitWidth),
                    qnn.QuantTanh(bit_width=activationBitWidth, quant_type=activationQuantType),
                    qnn.QuantConv2d(384, 384, kernel_size=3, padding=1,
                        weight_quant_type=weightQuantType, weight_bit_width=weightBitWidth),
                    qnn.QuantTanh(bit_width=activationBitWidth, quant_type=activationQuantType),
                    qnn.QuantConv2d(384, 256, kernel_size=3, padding=1,
                        weight_quant_type=weightQuantType, weight_bit_width=weightBitWidth),
                    qnn.QuantTanh(bit_width=activationBitWidth, quant_type=activationQuantType),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    )
                self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
                self.classifier = nn.Sequential(
                    qnn.QuantLinear(256 * 6 * 6, 4096, bias=True, 
                        weight_quant_type=weightQuantType, weight_bit_width=weightBitWidth),
                    qnn.QuantTanh(bit_width=activationBitWidth, quant_type=activationQuantType),
                    nn.Dropout(),
                    qnn.QuantLinear(4096, 4096, bias=False,
                        weight_quant_type=weightQuantType, weight_bit_width=weightBitWidth),
                    qnn.QuantTanh(bit_width=activationBitWidth, quant_type=activationQuantType),
                    nn.Dropout(),
                    qnn.QuantLinear(4096, num_classes, bias=True,
                        weight_quant_type=weightQuantType, weight_bit_width=weightBitWidth),
                    )
            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

        net = Net()
            
        if torch.cuda.is_available():
            device = torch.device("cuda") 
        else:
            device = torch.device("cpu")
            
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            
        train(net, optimizer, criterion, train_data_loader, epochs=2, device=device)
        test_model(net)


