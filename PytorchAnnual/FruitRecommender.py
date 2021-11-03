import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

def getDictlabel(path):
    df = pd.read_csv("Training_set.csv")
    # print(df.head)
    df['label'] = pd.Categorical(df['label'])
    dictlabel = df['label'].cat.categories
    return dictlabel

def getImage(path):
    img = Image.open(path)
    return img

def preProcessImg(img):
    mytransform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    img = mytransform(img)
    img = img.view(1,3,224,224)
    return img

class fruitRegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,5,1)
        self.conv2 = nn.Conv2d(16,64,3,1)
        self.conv3 = nn.Conv2d(64,128,3,1)
        self.fc1 = nn.Linear(86528,256)
        self.fc2 = nn.Linear(256,131)
    
    def forward(self,X):
        
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        
        X = X.view(-1, 86528)
        
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)

def model_initialize():
    model = fruitRegModel()
    return model

def inference(img_path,lbl_path,weight_path):
    dictlabel = getDictlabel(lbl_path)
    img = getImage(img_path)
    img = preProcessImg(img)
    Model = model_initialize()
    Model.load_state_dict(torch.load(weight_path))
    Model.eval()
    with torch.no_grad():
        y = Model(img)
    predicted = torch.max(y.data,1)[1]
    return dictlabel[predicted]