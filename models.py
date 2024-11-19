import torch
from torch import nn


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 10)
    
    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = self.fc1(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size= 3), 
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(32, 64, kernel_size= 3),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(64, 128, kernel_size= 3),
            nn.ReLU(),
            nn.Flatten())
        
        self.linear =  nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 10))


            #nn.Linear(, ),#
            #nn.Linearr)


    def forward(self, x):
        ### To do 4
        x = self.conv(x)
        x = self.linear(x)
        return x 



class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        ## To do 7
        self.resnet.fc = nn.Linear(512, 10)  #on change un attribut de la classe resnet 

    def forward(self, x):
        return self.resnet(x)



