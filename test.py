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
from BrevitasModNets.torchvision_alexnet_noBN import AlexNet as n2
from BrevitasModNets.torchvision_alexnet_BN_PreA import AlexNet as n3
from BrevitasModNets.torchvision_alexnet_BN_PostA import AlexNet as n4
from BrevitasModNets.torchvision_googlenet import GoogLeNet as n5
from BrevitasModNets.torchvision_resnet import resnet18 as n6

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

def train(model, modelname, datasetName, optimizer, loss_fn, train_loader, val_loader, epochs=5, device="cpu"):
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
            if modelname == "torchvision_googlenet":
                output = output.logits
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)

        if datasetName == "CatsVSDogs":
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
            
        elif datasetName == "CIFAR10":
            print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {}, accuracy = {}'.format(epoch, training_loss,
                None, None))

def test_model(model, modelname, datasetName):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            output = model(images)
            if modelname == "torchvision_googlenet" and datasetName == "CIFAR10":
                output = output.logits
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

#######################################################################################################################

print("Testing...")

quantizations = [QuantType.BINARY, QuantType.INT, QuantType.FP]
bitWidths = [2, 3, 4, 5, 6, 7, 8]
bitWidths_alt = [8]
choices = ["CatsVSDogs", "CIFAR10"]

for choice in choices:
    if choice == "CatsVSDogs":
        train_data_path = "/home/lucia/datasets/catsVdogs/train/"
        train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=img_transforms, is_valid_file=check_image)
        validation_data_path = "/home/lucia/datasets/catsVdogs/validation/"
        validation_data = torchvision.datasets.ImageFolder(root=validation_data_path,transform=img_transforms, is_valid_file=check_image)
        test_data_path = "/home/lucia/datasets/catsVdogs//test/"
        test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=img_transforms, is_valid_file=check_image)
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        num_classes=2
    
    elif choice == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root='/home/lucia/datasets/CIFAR10', train=True, download=True, transform=img_transforms)
        testset = torchvision.datasets.CIFAR10(root='/home/lucia/datasets/CIFAR10', train=False, download=True, transform=img_transforms)
        train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_data_loader = None
        test_data_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = 10


    for quantization in quantizations:
        weight_quant_type = quantization
        quant_type = quantization
            
        if quantization == QuantType.BINARY:
            weight_bit_width = 1
            bit_width = 1
            index = 0

            Nets = [n1(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width), 
                    n2(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width), 
                    n3(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width), 
                    n4(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width),
                    n5(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width),
                    n6(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width)
                    ]

            netStr = [  "self_alexnet", 
                        "torchvision_alexnet_noBN", 
                        "torchvision_alexnet_BN_PreA",
                        "torchvision_alexnet_BN_PostA",
                        "torchvision_googlenet",
                        "torchvision_resnet" 
                    ]

            for net in Nets:

                if torch.cuda.is_available():
                    device = torch.device("cuda") 
                else:
                    device = torch.device("cpu")
                
                print("######################################################################################")
                print(" ")
                print("Testing net " + netStr[index] + " with following parameters on Dataset " + choice + ":")
                print("quantization: " + str(quantization) + " || bit width: " + str(bit_width))
                print(" ")
                
                net.to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)
                
                train(net, netStr[index], choice, optimizer, criterion, train_data_loader, validation_data_loader, 1, device)
                test_model(net, netStr[index], choice)
                
                index+=1

        elif quantization == QuantType.INT:
            for bitWidth in bitWidths:
                weight_bit_width = bitWidth
                bit_width = bitWidth
                index = 0
            
                Nets = [n1(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width), 
                        n2(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width), 
                        n3(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width), 
                        n4(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width),
                        n5(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width),
                        n6(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width)
                        ]

                netStr = [  "self_alexnet", 
                            "torchvision_alexnet_noBN", 
                            "torchvision_alexnet_BN_PreA",
                            "torchvision_alexnet_BN_PostA",
                            "torchvision_googlenet",
                            "torchvision_resnet" 
                        ]
                    
                for net in Nets:

                    if torch.cuda.is_available():
                        device = torch.device("cuda") 
                    else:
                        device = torch.device("cpu")
                
                    print("######################################################################################")
                    print(" ")
                    print("Testing net " + netStr[index] + " with following parameters on Dataset " + choice + ":")
                    print("quantization: " + str(quantization) + " || bit width: " + str(bit_width))
                    print(" ")
                    
                    net.to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
                
                    train(net, netStr[index], choice, optimizer, criterion, train_data_loader, validation_data_loader, 1, device)
                    test_model(net, netStr[index], choice)
                    
                    index+=1

        elif quantization == QuantType.FP:
                weight_bit_width = 32
                bit_width = 32
                index = 0
                                
                Nets = [n1(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width), 
                        n2(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width), 
                        n3(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width), 
                        n4(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width),
                        n5(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width),
                        n6(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width)
                        ]

                netStr = [  "self_alexnet", 
                            "torchvision_alexnet_noBN", 
                            "torchvision_alexnet_BN_PreA",
                            "torchvision_alexnet_BN_PostA",
                            "torchvision_googlenet",
                            "torchvision_resnet" 
                        ]
                    
                for net in Nets:

                    if torch.cuda.is_available():
                        device = torch.device("cuda") 
                    else:
                        device = torch.device("cpu")
                
                    print("######################################################################################")
                    print(" ")
                    print("Testing net " + netStr[index] + " with following parameters on Dataset " + choice + ":")
                    print("quantization: " + str(quantization) + " || bit width: " + str(bit_width))
                    print(" ")
                    
                    net.to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
                
                    train(net, netStr[index], choice, optimizer, criterion, train_data_loader, validation_data_loader, 1, device)
                    test_model(net, netStr[index], choice)
                    
                    index+=1