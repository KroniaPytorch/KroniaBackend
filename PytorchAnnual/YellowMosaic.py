from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

dictlabel=['diseased okra leaf', 'fresh okra leaf']

def getImg(path):
    img = Image.open(path)
    return img

def preProcessImg(img):
    mytransform = transforms.Compose(
            [
              transforms.Resize((224,224)),
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
    )
    img = mytransform(img)
    img = img.view(1,3,224,224)
    return img

class YVMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,5,1)
        self.conv2 = nn.Conv2d(16,64,3,1)
        self.conv3 = nn.Conv2d(64,128,3,1)
        self.conv4 = nn.Conv2d(128,128,3,1)
        self.fc1 = nn.Linear(18432,256)
        self.fc2 = nn.Linear(256,2)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,2,2)
        
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x,2,2)
        
        x = x.view(-1,18432)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

def model_initialize():
    model = YVMModel()
    return model

def inference(img_path):
    img = getImg(img_path)
    img = preProcessImg(img)
    Model = model_initialize()
    Model.load_state_dict(torch.load('YellowMosiac.pth',map_location=torch.device('cpu')))
    Model.eval()
    with torch.no_grad():
        y = Model(img)
    predicted = torch.max(y.data,1)[1]
    return dictlabel[predicted]