#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse


#TODO: Import dependencies for Debugging andd Profiling
try:
    import smdebug.pytorch as smd
except:
    pass

def test(model, test_loader, criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    
    # set the hook to evaluation mode
    if hook:
        hook.set_mode(smd.modes.EVAL)


    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    print(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")
#------------------------------------------------------------------------------------------

def train(model, train_loaders, epochs, criterion, optimizer, device, hook):
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            running_loss = 0
            running_correct = 0
        
            if phase == 'train':
                model.train()
                if hook:
                    hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                if hook:
                    hook.set_mode(smd.modes.EVAL)
        
            for data, target in train_loaders[phase]:
                data = data.to(device)
                target = target.to(device)
            
                outputs = model(data)
                loss = criterion(outputs, target)
            
                _, preds = torch.max(outputs, 1)
            
                running_loss += loss.item() * data.size(0)
                with torch.no_grad():
                    running_correct += torch.sum(preds == target).item()
            
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
            epoch_loss = running_loss / len(train_loaders[phase].dataset)
            epoch_acc = running_correct / len(train_loaders[phase].dataset)
        
            print(f'Epoch : {epoch}-{phase}, epoch loss = {epoch_loss}, epoch_acc = {epoch_acc}')

    return model
    
def net(device):
    num_classes = 133
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_inputs = model.fc.in_features
    model.fc = nn.Linear(num_inputs, num_classes)
    model = model.to(device)
    return model

#-----------------------------------------------------------------------------------------------------------
def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model
#------------------------------------------------------------------------------------------------------------

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    # creating the dataloaders
    dataloaders = {
        split : torch.utils.data.DataLoader(data[split], batch_size, shuffle=True)
        for split in ['train', 'valid', 'test']
    }

    return dataloaders

def main(args):


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net(device)
    # create the debugging profiling hook
    hook = smd.Hook.create_from_json_file()
    # register the model for debugging and saving tensors
    hook.register_hook(model)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    # register the model loss for debugging
    hook.register_loss(loss_criterion)

    optimizer = optim.Adam(model.fc.parameters(), args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    # loading the data
    # declaring the data_transforms for train, validation and test datasets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # loading the datasets
    image_datasets = {
        split : datasets.ImageFolder(os.path.join(args.data_dir, split), data_transforms[split])
        for split in ['train', 'valid', 'test']
    }

    dataloaders = create_data_loaders(image_datasets , args.batch_size)
    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']
    test_loader = dataloaders['test']

    train_loaders = {
        'train' : train_loader,
        'valid' : valid_loader
    }

    model=train(model, train_loaders, args.epochs, loss_criterion, optimizer, device, hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device, hook)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, metavar="E", help="number of epochs to train (default: 1)")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_DATA"])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    args = parser.parse_args()

    main(args)