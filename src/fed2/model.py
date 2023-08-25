import torch.nn as nn


def CnnModel_0(num_classes=1000):
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32,
                  kernel_size=5, stride=1, padding='same'),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64,
                  kernel_size=5, stride=1, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(3136, 2048),
        nn.ReLU(),
        nn.Linear(2048, num_classes, bias=False))

CnnModelWeightNames_0 = (
    'model/0.cnn.weight', 'model/0.cnn.bias', 'model/3.cnn.weight', 'model/3.cnn.bias', 
    'model/7.Linear.weight', 'model/7.Linear.bias', 'model/7.Linear.weight')

def CnnModel_1(num_classes=1000):
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32,
                  kernel_size=3, stride=1, padding='same'),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64,
                  kernel_size=5, stride=1, padding='same'),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64,
                  kernel_size=3, stride=1, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(3136, 1024),
        nn.ReLU(),
        nn.Linear(1024, num_classes, bias=False))

CnnModelWeightNames_1 = (
    'model/0.cnn.weight', 'model/0.cnn.bias', 'model/3.cnn.weight', 'model/3.cnn.bias', 'model/5.cnn.weight', 
    'model/5.cnn.bias', 'model/9.Linear.weight', 'model/9.Linear.bias', 'model/11.Linear.weight')
