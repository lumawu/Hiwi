# Following has been removed during forward pass between features and
# classifiers:
# x = self.avgpool(x)
# x = torch.flatten(x, 1)

# BatchNorm2d before activation: How Thomas told me to do it
# BatchNorm2d after activation: How the prototxt told me to do it
# Test both!

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.core.quant import QuantType

class activation(nn.Module):
        def __init__(self, bit_width, quant_type):
            super(activation, self).__init__()

        def __call__(self, bit_width, quant_type):
            if quant_type == QuantType.BINARY:
                return qnn.QuantTanh(bit_width=bit_width, quant_type=quant_type)
            else: 
                return qnn.QuantReLU(bit_width=bit_width, quant_type=quant_type)

class AlexNet(nn.Module):
            def __init__(self, num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width):
                super(AlexNet, self).__init__()
                a = activation(bit_width, quant_type)
                self.features = nn.Sequential(
                    qnn.QuantConv2d(3, 96, kernel_size=11, stride=4, padding=0,
                        weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
                    a(bit_width, quant_type),      
                    nn.BatchNorm2d(96),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    qnn.QuantConv2d(96, 256, kernel_size=5, padding=2,
                        weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
                    a(bit_width, quant_type),  
                    nn.BatchNorm2d(256),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    qnn.QuantConv2d(256, 384, kernel_size=3, padding=1,
                        weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
                    a(bit_width, quant_type),  
                    qnn.QuantConv2d(384, 384, kernel_size=3, padding=1,
                        weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
                    a(bit_width, quant_type),  
                    qnn.QuantConv2d(384, 256, kernel_size=3, padding=1,
                        weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
                    a(bit_width, quant_type),
                    nn.BatchNorm2d(256), 
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    )
                self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
                self.classifier = nn.Sequential(
                    qnn.QuantLinear(256 * 6 * 6, 4096, bias=True, 
                        weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
                    a(bit_width, quant_type),  
                    nn.Dropout(),
                    qnn.QuantLinear(4096, 4096, bias=False,
                        weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
                    a(bit_width, quant_type),  
                    nn.Dropout(),
                    qnn.QuantLinear(4096, num_classes, bias=True,
                        weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
                    )
            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x
