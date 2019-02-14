from image_processing import *
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from collections import OrderedDict

def create_model(model_arch, device, learning_rate, hidden_units):
    
    if model_arch == "VGG":
        model = models.vgg19(pretrained=True)
        num_features = model.classifier[0].in_features
    else:
        model = models.densenet121(pretrained=True)
        num_features = model.classifier.in_features
    print(model)

    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict ([('fc1', nn.Linear(num_features, hidden_units)),
                                       ('relu', nn.ReLU()),
                                       ('fc2', nn.Linear(hidden_units, 102)),
                                       ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier    

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    model.to(device)
    
    return model, criterion, optimizer, classifier

