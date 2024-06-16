# Import necessary dependencies
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.debugger import Rule, DebuggerHookConfig, ProfilerConfig, rule_configs

# Import dependencies for Debugging and Profiling
from smdebug.pytorch import get_hook


#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader, criterion, hook):
    model.eval()
    hook.set_mode(sagemaker.debugger.modes.EVAL)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}%")

def train(model, train_loader, criterion, optimizer, hook):
    model.train()
    hook.set_mode(sagemaker.debugger.modes.TRAIN)
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    
def net():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(2048, 133)  # Assuming 133 classes for dog breed classification
    return model

def create_data_loaders(data, batch_size):
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
    trainset = torchvision.datasets.ImageFolder(root=data+'/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = torchvision.datasets.ImageFolder(root=data+'/test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def main(args):
    model = net()
    hook = get_hook(create_if_not_exists=True)
    hook.register_module(model)
    
    train_loader, test_loader = create_data_loaders(args.data, args.batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    train(model, train_loader, criterion, optimizer, hook)
    test(model, test_loader, criterion, hook)
    
    torch.save(model.state_dict(), args.model_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/path/to/data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--model_path', type=str, default='/path/to/save/model.pth')
    args = parser.parse_args()
    main(args)
