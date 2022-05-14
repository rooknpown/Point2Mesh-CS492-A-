import torch
import torch.nn as nn
from torch.nn import init


def init_weight(model, init_weights, init_type):
    def init_w(model):
        classname = model.__class__.__name__
        if hasattr(model, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal_':
                init.normal_(model.weight.data, 0.0, init_weights)
            elif init_type == 'uniform_':
                init.uniform_(model.weight.data, -init_weights, init_weights)
                init.uniform_(model.bias.data, -init_weights, init_weights)
            elif init_type == "conv_xavier_normal_":
                for m in model.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_normal_(m.weight)
                        nn.init.constant_(m.bias, init_weights)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    model.apply(init_w)




