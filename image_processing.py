import json
from PIL import Image
from torchvision import datasets, transforms, models
import torch
import numpy as np
import matplotlib.pyplot as plt

def transform_image(data_dir):
    

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms =  transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])


    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_dir)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 8)
    testloader = torch.utils.data.DataLoader(valid_data, batch_size= 8)
    
    return trainloader, validloader, testloader, train_data

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)
    
    size=256
    width, height = image.size
    shortest_side = min(width, height)
                
    image_resize = image.resize((int((width/shortest_side)*size),int((height/shortest_side)*size)) )
    
    center = width/4, height/4
    left, top, right, bottom = center[0]-(224/2), center[1]-(224/2), center[0]+(224/2), center[1]+(224/2)
    cropped_image = image_resize.crop((left, top, right, bottom))
    np_image = np.array(cropped_image) / 255.
                         
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])     
    image_array = (np_image - mean) / std
    image_array = np_image.transpose((2, 0, 1))                     

    
    return image_array


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
   
    #PyTorch tensors assume the color channel is the first dimension
    #but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


