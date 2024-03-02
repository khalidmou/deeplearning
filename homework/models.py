import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn


class CNNClassifier(torch.nn.Module):
    def __init__(self, layers=[16,  64, 128], n_input_channels=3, n_output_channels=6, kernel_size=5):
        super().__init__()

        self.Dropout = torch.nn.Dropout(0.2)
        L = []
        c = n_input_channels
        for l in layers:
            L.append(torch.nn.Conv2d(c, l, kernel_size, stride=2, padding=kernel_size//2 , bias = False))
            L.append(torch.nn.BatchNorm2d(l))
            L.append(torch.nn.ReLU())
            L.append(torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, n_output_channels)

    def forward(self, x):
        return self.classifier(self.network(x).mean(dim=[2, 3]))




def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


class FCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, dropout_rate=0.5):
        super(FCN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


        self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)  # Concatenate with skip connection
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.ConvTranspose2d(32 , num_classes, kernel_size=3, stride=1,
                               padding=1)  # Concatenate with skip connection

    def forward(self, x):

        x1 = self.relu(self.conv1(x))
        if x.size(2) > 2 and x.size(3) > 2:
            x2 = self.maxpool(x1)
        else :
            x2 = x1

        x3 = self.relu(self.conv2(x2))
        x5 = self.dec1(x3)
        skip_connection1 = torch.cat((x2,x5),dim=1)
        skip_connection1_dec1 = self.dec1(skip_connection1)
        if x.size(2) > 2 and x.size(3) > 2:
            x6 = self.upsample(skip_connection1_dec1)
        else:
            x6 = x5
        skip_connection2 = torch.cat((x1, x6), dim=1)
        x7 = self.dec1(skip_connection2)
        return self.dec2(x7)


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
