# Dependencies
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import OrderedDict
import time
from PIL import Image
import matplotlib
import json

def preprocessData(data_dir):

    #The above function takes in the model architecture, the number of hidden units, and the category to 
    # name mapping. 

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = [transforms.Compose([
                                       transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(30),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])]


    # TODO: Load the datasets with ImageFolder
    image_datasets = [datasets.ImageFolder(train_dir,transform = data_transforms[0]),
                  datasets.ImageFolder(valid_dir,transform = data_transforms[1]),
                  datasets.ImageFolder(test_dir,transform = data_transforms[2])]

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = [DataLoader(image_datasets[0],batch_size=128,shuffle=True),
               DataLoader(image_datasets[1],batch_size=32,shuffle=True),
               DataLoader(image_datasets[2],batch_size=32,shuffle=True)]
    
    return image_datasets,dataloaders
    

def build_model(arch, hidden_units,cat_to_name):
    
    #This function takes in a pretrained model, a number of hidden units, and a
    #dictionary of categories to names. It then freezes the parameters of the pretrained model, builds a
    #classifier with the specified number of hidden units, and replaces the classifier of the pretrained
    #model with the new classifier
    
    
    if arch.lower() == "vgg13":
        model = models.vgg13(pretrained=True)
    else:
        model = models.resnet50(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False # Freeze parameters so we don't backprop through them
        
    if arch.lower() == "vgg13":
        classifier = nn.Sequential(OrderedDict([
                            ('dropout1', nn.Dropout(0.1)),
                            ('fc1', nn.Linear(25088, hidden_units)), # 25088 must match
                            ('relu1', nn.ReLU()),
                            ('dropout2', nn.Dropout(0.1)),
                            ('fc2', nn.Linear(hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
        
        model.classifier = classifier
    
    else:
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(model.fc.in_features, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout1',nn.Dropout(0.1)),
                          ('fc3', nn.Linear(hidden_units,len(cat_to_name))),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
        model.fc = classifier
    
    print(f"\n\nModel built from {arch} and {hidden_units} hidden units.")

    return model

def train_model(model,dataloaders,learning_rate, epochs, gpu):
    
    if gpu:
        print(f"\n\nGPU Activated\n")
    
    print(f"Model Training Started.\n\n")
    
    criterion = nn.NLLLoss()
        
    optimizer = optim.Adam(model.fc.parameters(),lr=learning_rate)
    
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device);
    
    steps = 0
    running_loss = 0
    print_every = 5
    
    
    for epoch in range(epochs):
        for inputs, labels in dataloaders[0]:
            steps += 1
            # Move input and label tensors to the default device
            if gpu:
                inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders[1]:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        test_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps).data
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(dataloaders[1]):.3f}.. "
                  f"Test accuracy: {accuracy/len(dataloaders[1])*100:.3f}")
                running_loss = 0
            
            
                model.train()
    
    return model, optimizer, criterion

def test_model(model,dataloaders,gpu):
    test_loss = 0   
    accuracy = 0
    print_every = 5
    model.eval()
    
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device);
    
    with torch.no_grad():
            for inputs, labels in dataloaders[2]:
                    if gpu:
                        inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Test loss: {test_loss/len(dataloaders[2]):.3f}.. "
                  f"Test accuracy: {accuracy/len(dataloaders[2])*100:.3f}")
          
    model.train()
    

parser = argparse.ArgumentParser()


parser.add_argument('data_directory', action='store',
                    default = 'flowers',
                    help='Set directory to load training data, e.g., "flowers"')


parser.add_argument('--save_dir', action='store',
                    default = '.',
                    dest='save_dir',
                    help='Set directory to save checkpoints, e.g., "assets"')

# Choose architecture: python train.py data_dir --arch "vgg13"
parser.add_argument('--arch', action='store',
                    default = 'resnet50',
                    dest='arch',
                    help='Choose architecture, e.g., "vgg13"')

# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
parser.add_argument('--learning_rate', action='store',
                    default = 0.001,
                    dest='learning_rate',
                    help='Choose architecture learning rate, e.g., 0.01')

parser.add_argument('--hidden_units', action='store',
                    default = 1024,
                    dest='hidden_units',
                    help='Choose architecture hidden units, e.g., 512')

parser.add_argument('--epochs', action='store',
                    default = 5,
                    dest='epochs',
                    help='Choose architecture number of epochs, e.g., 20')

# Use GPU for training: python train.py data_dir --gpu
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Use GPU for training, set a switch to true')

parse_results = parser.parse_args()


data_dir = parse_results.data_directory
save_dir = parse_results.save_dir
arch = parse_results.arch
learning_rate = float(parse_results.learning_rate)
hidden_units = int(parse_results.hidden_units)
epochs = int(parse_results.epochs)
gpu = parse_results.gpu


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Load and preprocess data
image_datasets, dataloaders = preprocessData(data_dir)

# Building and training the classifier
model_init = build_model(arch, hidden_units,cat_to_name)
model, optimizer, criterion = train_model(model_init,dataloaders,learning_rate, epochs, gpu)

# Save the checkpoint 
model.to('cpu')

# TODO: Save the checkpoint 
model.class_to_idx = image_datasets[0].class_to_idx

checkpoint = {
    'arch': model,
    'hidden_dim': hidden_units,
    'output_dim': len(cat_to_name),
    'class_to_idx': model.class_to_idx,
    'state_dict': model.state_dict()
}

torch.save(checkpoint, save_dir + '/checkpoint.pth')

if save_dir == ".":
    save_dir_name = "current folder"
else:
    save_dir_name = save_dir + " folder"

print(f'\n\nCheckpoint saved to {save_dir_name}.')