import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.core.quant import QuantType


class AlexNet(nn.Module):

    def __init__(self, num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            qnn.QuantConv2d(3, 64, kernel_size=11, stride=4, padding=2,
               weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
            activation(bit_width, quant_type),
            nn.MaxPool2d(kernel_size=3, stride=2),
            qnn.QuantConv2d(64, 192, kernel_size=5, padding=2,
               weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
            activation(bit_width, quant_type),
            nn.MaxPool2d(kernel_size=3, stride=2),
            qnn.QuantConv2d(192, 384, kernel_size=3, padding=1,
               weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
            activation(bit_width, quant_type),
            qnn.QuantConv2d(384, 256, kernel_size=3, padding=1,
               weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
            activation(bit_width, quant_type),
            qnn.QuantConv2d(256, 256, kernel_size=3, padding=1,
               weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
            activation(bit_width, quant_type),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            qnn.QuantLinear(256 * 6 * 6, 4096, bias=True,
                  weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
            activation(bit_width, quant_type),
            nn.Dropout(),
            qnn.QuantLinear(4096, 4096, bias=True,
                  weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
            activation(bit_width, quant_type),
            qnn.QuantLinear(4096, num_classes, bias=True,
                  weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class activation(nn.Module):
        def __init__(self, bit_width, quant_type):
            self.bit_width = bit_width
            self.quant_type = quant_type

        def __call__(self):
            if self.quant_type == QuantType.BINARY:
                return qnn.QuantTanh(self.bit_width, self.quant_type)
            else: 
                return qnn.QuantReLU(self.bit_width, self.quant_type)