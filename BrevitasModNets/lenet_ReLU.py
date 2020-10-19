# Following has been removed during forward pass between features and
# classifiers:
# x = self.avgpool(x)
# x = torch.flatten(x, 1)

import torch.nn as nn
import brevitas.nn as qnn
from brevitas.core.quant import QuantType

class LeNet(nn.Module):
            def __init__(self, num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width):
                super(LeNet, self).__init__()
                self.features = nn.Sequential(
                    qnn.QuantConv2d(1, 20, kernel_size=5, stride=1, padding=0,
                        weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),                   
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    qnn.QuantConv2d(20, 50, kernel_size=5, padding=0,
                        weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    )
                self.classifier = nn.Sequential(
                    qnn.QuantLinear(64 * 50 * 4 * 4, 64 * 500, bias=True, 
                        weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
                    qnn.QuantReLU(quant_type, bit_width),                    
                    qnn.QuantLinear(64 * 500, 64 * 10, bias=False,
                        weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
                    )
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x