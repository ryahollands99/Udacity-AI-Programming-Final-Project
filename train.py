import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import OrderedDict
import json
import argparse
import os
from image_processing import transform_image
from model import create_model


def get_args():

    """
        Get arguments from command line
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_directory", action="store", type=str, default="/home/workspace/aipnd-project/flowers", help="data directory containing training and testing data")
    parser.add_argument("model_arch", type=str, default='Densenet121', help="Choose a model: VGG19 or Densenet121")
    parser.add_argument("Learning_rate", type=float, default = 0.001)
    parser.add_argument("hidden_units", type = int, default = 500)
    parser.add_argument("epochs", type=int, default=3)
    parser.add_argument("device", type=str, default = "GPU", help="Use CPU or GPU")
    
    args = parser.parse_args()
    return args



def main():
    input = get_args()
    data_dir = input.data_directory
    trainloader, testloader, validloader, traindata = transform_image(data_dir)
    if input.device == "GPU":
        device = "cuda"
    else:
        device = "cpu"
    
    model, criterion, optimizer, classifier = create_model(input.model_arch, device, input.Learning_rate, input.hidden_units)
    
    
    train_model = do_deep_learning(model, trainloader, validloader, input.epochs, 40, criterion, optimizer, device)
    validate(train_model, testloader, device, criterion)
    save_checkpoint(train_model, classifier, optimizer, traindata)
    print('Completed!')






def do_deep_learning(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0
    
    train_losses, valid_losses = [], []

    # change to cuda
    model.to(device)

    for e in range(epochs): 
        running_loss = 0
        for images, labels in iter(trainloader):
            steps += 1
            
    
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
                
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                correct = 0
    
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model.forward(inputs)
                        batch_loss = criterion(outputs, labels)
                        valid_loss += batch_loss.item()
                        prediction = torch.exp(outputs)
                        top_p, top_class = prediction.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        correct += torch.mean(equals.type(torch.FloatTensor)).item()
            
                train_losses.append(running_loss/len(trainloader))
                valid_losses.append(valid_loss/len(validloader))
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Train loss: {:.4f}... ".format(running_loss/print_every),
                      "Validation loss: {:4f}...".format(valid_loss/len(validloader)),
                      "Validation accuracy: {:.4f}".format(correct/len(validloader)) )
                running_loss = 0
                model.train()
                
    return model

def validate(model, testloader, device, criterion):

    model.eval()
    correct = 0
    test_loss = 0
    model.to(device)
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            batch_loss = criterion(outputs, labels)
            test_loss += batch_loss.item()
            prediction = torch.exp(outputs)
            top_p, top_class = prediction.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            correct += torch.mean(equals.type(torch.FloatTensor)).item()
    
    print("Testing Loss: {:.3f}".format(test_loss/len(testloader)),"Testing Accuracy: {:.3f}".format(correct/len(testloader)))
              
                
def save_checkpoint(model, classifier, optimizer, train_data):
    checkpoint = {'arch': 'densenet121',
                'classifier': classifier,
                #'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'mapping': train_data.class_to_idx}
    torch.save(checkpoint, '/home/workspace/aipnd-project/checkpoint.pth')
    print("Saved")
    
    
if __name__ == '__main__':
       main()
