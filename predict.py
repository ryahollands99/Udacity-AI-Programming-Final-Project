import argparse
import random, os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from image_processing import *
import matplotlib.pyplot as plt

def get_args():

    """
        Get arguments from command line
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_directory", action="store", type=str, default="/home/workspace/aipnd-project/flowers", help="data directory")
    parser.add_argument("top_k", type=int, default='5', help="Print out the top K c1asses")
    parser.add_argument("json_file", type=str, default = '/home/workspace/aipnd-project/cat_to_name.json')
    parser.add_argument("device", type=str, default = "GPU", help="Use CPU or GPU")
    
    args = parser.parse_args()
    return args

    
def main():
    
    # get arguments from command line
    input = get_args()
    data_dir = input.data_directory
    file_name = random_image(data_dir + "/test")
    
    # load category names file
    with open(input.json_file, 'r') as f:
        cat_to_name = json.load(f)
        
    # load trained model
    model = load_checkpoint()
    
    # Set device to either CUDA or CPU
    if input.device == "GPU":
        device = "cuda"
    else:
        device = "cpu"
    
    # Process images, predict classes, and display results
    predict(model, file_name, cat_to_name, input.top_k, device)
    

def load_checkpoint():
    
    checkpoint = torch.load('checkpoint.pth')
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    model.classifier = checkpoint['classifier']      
    optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['mapping']
 
    return model

def random_image(test_dir):
    dir_name = random.choice(os.listdir(test_dir)) #change dir name to whatever
    file_path = test_dir + '/'+ str(dir_name)
    file_name = test_dir + '/'+ str(dir_name) + '/' + str(random.choice(os.listdir(file_path)))
    return file_name

def predict(model, file_name, cat_to_name, top_k, device):
    
    file_path = file_name.split('/')[-2]
    print(file_path)
    image_dir = file_name
    flower_name = cat_to_name[file_path]
    print(flower_name)

    model.eval()
    model.to(device)
    img = process_image(file_name)
    img_torch = torch.from_numpy(img)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    probability = F.softmax(output.data,dim=1)
    probability = probability.topk(5)

    n = 0

    prob, labels = probability
    labels = np.array(labels[0])

    prob = np.array(prob)
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [index_to_class[label] for label in labels]

    class_labels = [cat_to_name[label] for label in top_classes]
    
    print(prob)
    print(class_labels)
    #imshow((process_image(file_name)))
        
# Run the program           
if __name__ == "__main__":
    main()