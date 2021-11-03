from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models,transforms

dictlabel = ['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']

def getImage(path):
    img = Image.open(path)
    return img

def preProcessImage(img):
    mytransform = transforms.Compose(
        [
            transforms.Resize((200,200)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ]
    )
    img = mytransform(img)
    img=img.view(1,3,200,200)
    return img

class SoilTypeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,1)
        self.conv2 = nn.Conv2d(16,32,3,1)
        self.conv3 = nn.Conv2d(32,64,3,1)
        self.conv4 = nn.Conv2d(64,64,3,1)
        self.conv5 = nn.Conv2d(64,64,3,1)
        self.fc1 = nn.Linear(1024,128)
        self.fc2 = nn.Linear(128,5)
    
    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X,2,2)
        
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X,2,2)
        
        X = F.relu(self.conv5(X))
        X = F.max_pool2d(X,2,2)
        
        X = X.view(-1,1024)
        
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return F.log_softmax(X,dim=1)


def model_initialize():
    model = SoilTypeModel()
    return model

def inference(img_path):
    img = getImage(img_path)
    img = preProcessImage(img)
    Model = model_initialize()
    Model.load_state_dict(torch.load('SoilDetect2.pth'))
    Model.eval()
    with torch.no_grad():
        y = Model(img)
    predicted = torch.max(y.data,1)[1]
    return dictlabel[predicted]