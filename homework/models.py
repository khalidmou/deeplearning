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


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FCN(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=5, features=[64, 128],
    ):
        super(FCN, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.small_size = nn.Conv2d(3, 5, 1, 1)

        for feature in features:
            self.downs.append(Block(in_channels, feature))
            in_channels = feature


        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(Block(feature*2, feature))

        self.bottleneck = Block(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        if x.size(2) > 8 and x.size(3) > 8:
            for down in self.downs:
                    x = down(x)
                    skip_connections.append(x)
                    x = self.pool(x)

            x = self.bottleneck(x)
            skip_connections = skip_connections[::-1]

            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                skip_connection = skip_connections[idx//2]

                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = self.ups[idx+1](concat_skip)

            return self.final_conv(x)
        return self.small_size(x)




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
