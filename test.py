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

def trainCatsVSDogs(model, modelname, optimizer, loss_fn, epochs=5, device="cpu"):
    itr = 1
    p_itr = 200
    model.train()
    total_loss = 0
    loss_list = []
    acc_list = []
    for epoch in range(epochs):
        for inputs, targets  in train_data_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if itr%p_itr == 0:
                pred = torch.argmax(output, dim=1)
                correct = pred.eq(targets)
                acc = torch.mean(correct.float())
                print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, itr, total_loss/p_itr, acc))
                loss_list.append(total_loss/p_itr)
                acc_list.append(acc)
                total_loss = 0
            
            itr += 1

def trainCIFAR10(model, modelname, optimizer, loss_fn, epochs=5, device="cpu"):
    entry = {
            "epochnum" : None,
            "loss" : None
            }
    losses = list()
    for epoch in range(epochs):
        training_loss = 0.0
        model.train()
        for batch in train_data_loader:
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
        training_loss /= len(train_data_loader.dataset)
        entry.update(epochnum = epoch, loss = training_loss)
        losses.append(dict(entry))
    for loss in losses:
        print("Training Loss for epoch " + str(loss.get("epochnum")) + ": " + str(loss.get("loss")))

def test_model(model, modelname):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_data_loader:
            input, target = data
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            if modelname == "torchvision_googlenet":
                _, predicted = torch.max(output.logits, 1)
            else:
                _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Test performance:    correct: {:d}  total: {:d}   accuracy: {:f}'.format(correct, total, correct / total))

def check_image(path):
    try:
        img = Image.open(path)
        return True
    except:
        return False

#######################################################################################################################

print("Testing...")

quantizations = [QuantType.BINARY, QuantType.INT]
bitWidths = [2, 3, 4, 5, 6, 7, 8]
bitWidths_alt = [8]
choices = ["CatsVSDogs", "CIFAR10"]

for x in choices:
    if x == "CatsVSDogs":
        train_data_path = "/home/lucia/projdata/catsVdogs/train/"
        test_data_path = "/home/lucia/projdata/catsVdogs/test/"
        train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=img_transforms, is_valid_file=check_image)
        test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=img_transforms, is_valid_file=check_image)
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        num_classes = 2
    
    elif x == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root='/home/lucia/projdata/CIFAR10', train=True, download=True, transform=img_transforms)
        testset = torchvision.datasets.CIFAR10(root='/home/lucia/projdata/CIFAR10', train=False, download=True, transform=img_transforms)
        train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_data_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = 10

    for quantization in quantizations:
        weight_quant_type = quantization
        quant_type = quantization
            
        if quantization == QuantType.BINARY:
            weight_bit_width = 1
            bit_width = 1
            Nets = [#n5(weight_quant_type, weight_bit_width, quant_type, bit_width, num_classes),
                    n1(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width), 
                    #n2(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width), 
                    #n3(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width), 
                    #n4(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width)
                    ]

            netStr = [#"torchvision_googlenet", 
                    "self_alexnet", 
                    #"torchvision_alexnet_noBN", 
                    #"torchvision_alexnet_BN_PreA",
                    #"torchvision_alexnet_BN_PostA"
                    ]
            index = 0

            for net in Nets:
                if torch.cuda.is_available():
                    device = torch.device("cuda") 
                else:
                    device = torch.device("cpu")
                
                print("######################################################################################")
                print(" ")
                print("Testing net " + netStr[index] + " with following parameters on Dataset " + x + ":")
                print("quantization: " + str(quantization) + " || bit width: " + str(bit_width))
                print(" ")

                net.to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)

                trainCatsVSDogs(net, netStr[index], optimizer, criterion, epochs=2, device=device)
                test_model(net, netStr[index])

                index+=1

        else:
            for bitWidth in bitWidths_alt:
                weight_bit_width = bitWidth
                bit_width = bitWidth
            
                Nets = [#n5(weight_quant_type, weight_bit_width, quant_type, bit_width, num_classes),
                    n1(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width), 
                    #n2(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width), 
                    #n3(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width), 
                    #n4(num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width)
                    ]

                netStr = [#"torchvision_googlenet", 
                    "self_alexnet", 
                    #"torchvision_alexnet_noBN", 
                    #"torchvision_alexnet_BN_PreA",
                    #"torchvision_alexnet_BN_PostA"
                    ]

                index = 0
            
                for net in Nets:
                    if torch.cuda.is_available():
                        device = torch.device("cuda") 
                    else:
                        device = torch.device("cpu")
                
                    print("######################################################################################")
                    print(" ")
                    print("Testing net " + netStr[index] + " with following parameters on Dataset " + x + ":")
                    print("quantization: " + str(quantization) + " || bit width: " + str(bit_width))
                    print(" ")

                    net.to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
                
                    trainCIFAR10(net, netStr[index], optimizer, criterion, epochs=2, device=device)
                    test_model(net, netStr[index])

                    index+=1


