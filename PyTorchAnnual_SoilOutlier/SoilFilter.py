from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models,transforms

dictlabel = ['Invalid Image', 'Valid Image']

def getImage(path):
    img = Image.open(path)
    return img

def preProcessImage(img):
    mytransform = transforms.Compose(
        [
            transforms.Resize((227,227)),
            #transforms.FiveCrop(227),
            #transforms.RandomHorizontalFlip(0.5),
            #transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ]
    )
    img = mytransform(img)
    img=img.view(1,3,227,227)
    return img

def model_initialize():
    model = models.alexnet(pretrained=False)
    model.classifier[6] = nn.Linear(4096,2)
    return model

def inference(img_path):
    Img = getImage(img_path)
    Img = preProcessImage(Img)
    Model = model_initialize()
    Model.load_state_dict(torch.load('Soil-Net.pth',map_location=torch.device('cpu')))
    Model.eval()
    with torch.no_grad():
        y = Model(Img)
    y = F.log_softmax(y,dim=1)
    predicted = torch.max(y.data,1)[1]
    return dictlabel[predicted]