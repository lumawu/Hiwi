import torch
import torch.nn as nn
from .utils import load_state_dict_from_url
import brevitas.nn as qnn


class AlexNet(nn.Module):

    def __init__(self, num_classes, weight_quant_type, weight_bit_width, quant_type, bit_width):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            qnn.QuantConv2d(3, 64, kernel_size=11, stride=4, padding=2,
               weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
            qnn.QuantTanh(inplace=True, bit_width, quant_type),
            qnn.BatchNorm2dToQuantScaleBias(64, weight_quant_type, weight_bit_width),
            nn.MaxPool2d(kernel_size=3, stride=2),
            qnn.QuantConv2d(64, 192, kernel_size=5, padding=2,
               weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
            qnn.QuantTanh(inplace=True, bit_width, quant_type),
            qnn.BatchNorm2dToQuantScaleBias(192, weight_quant_type, weight_bit_width),
            nn.MaxPool2d(kernel_size=3, stride=2),
            qnn.QuantConv2d(192, 384, kernel_size=3, padding=1,
               weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
            qnn.QuantTanh(inplace=True, bit_width, quant_type),
            qnn.BatchNorm2dToQuantScaleBias(384, weight_quant_type, weight_bit_width),
            qnn.QuantConv2d(384, 256, kernel_size=3, padding=1,
               weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
            qnn.QuantTanh(inplace=True, bit_width, quant_type),
            qnn.BatchNorm2dToQuantScaleBias(96, weight_quant_type, weight_bit_width),
            qnn.QuantConv2d(256, 256, kernel_size=3, padding=1,
               wweight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
            qnn.QuantTanh(inplace=True, bit_width, quant_type),
            qnn.BatchNorm2dToQuantScaleBias(96, weight_quant_type, weight_bit_width),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            qnn.QuantLinear(256 * 6 * 6, 4096, 
                  weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
            qnn.QuantTanh(inplace=True, bit_width, quant_type),
            qnn.BatchNorm2dToQuantScaleBias(96, weight_quant_type, weight_bit_width),
            nn.Dropout(),
            qnn.QuantLinear(4096, 4096,
                  weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
            qnn.QuantTanh(inplace=True, bit_width, quant_type),
            qnn.BatchNorm2dToQuantScaleBias(96, weight_quant_type, weight_bit_width),
            qnn.QuantLinear(4096, num_classes,
                  weight_quant_type = weight_quant_type, weight_bit_width = weight_bit_width),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x