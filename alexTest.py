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

from BrevitasModNets.self_alexnet import AlexNet as n1

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

num_workers = 6

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=5, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
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

        num_correct = 0 
        num_examples = 0
        model.eval()
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets) 
            valid_loss += loss.data.item() * inputs.size(0)
                            
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
            valid_loss /= len(val_loader.dataset)
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(epoch, training_loss,
            valid_loss, num_correct / num_examples))

def test_model(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('correct: {:d}  total: {:d}'.format(correct, total))
    print('accuracy = {:f}'.format(correct / total))

def check_image(path):
    try:
        img = Image.open(path)
        return True
    except:
        return False

train_data_path = "/export/users/wucm/datasets/catsVdogs/train/"
train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=img_transforms, is_valid_file=check_image)
validation_data_path = "/export/users/wucm/datasets/catsVdogs/validation/"
validation_data = torchvision.datasets.ImageFolder(root=validation_data_path,transform=img_transforms, is_valid_file=check_image)
test_data_path = "/export/users/wucm/datasets/catsVdogs/test/"
test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=img_transforms, is_valid_file=check_image)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
num_classes=2

weight_quant_type = QuantType.BINARY
quant_type = QuantType.BINARY
weight_bit_width = 1
bit_width = 1

net = n1(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width)

if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")

net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)

train(net, optimizer, criterion, train_data_loader, validation_data_loader , 5, device)
test_model(net)

torch.save(net.state_dict(), "/export/users/wucm/Hiwi/alexTest.pth")