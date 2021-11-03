from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

dictlabel = [
    'Apple Apple_scab',
    'Apple Black_rot',
    'Apple Cedar_apple_rust',
    'Apple healthy',
    'Blueberry healthy',
    'Cherry(including_sour) Powdery_mildew',
    'Cherry_(including_sour) healthy',
    'Corn_(maize) Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize) Common_rust_',
    'Corn_(maize) Northern_Leaf_Blight',
    'Corn_(maize) healthy',
    'Grape Black_rot',
    'Grape Esca_(Black_Measles)',
    'Grape Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape healthy',
    'Orange Haunglongbing_(Citrus_greening)',
    'Peach Bacterial_spot',
    'Peach healthy',
    'Pepper,_bell Bacterial_spot',
    'Pepper,_bell healthy',
    'Potato Early_blight',
    'Potato Late_blight',
    'Potato healthy',
    'Raspberry healthy',
    'Soybean healthy',
    'Squash Powdery_mildew',
    'Strawberry Leaf_scorch',
    'Strawberry healthy',
    'Tomato Bacterial_spot',
    'Tomato Early_blight',
    'Tomato Late_blight',
    'Tomato Leaf_Mold',
    'Tomato Septoria_leaf_spot',
    'Tomato Spider_mites Two-spotted_spider_mite',
    'Tomato Target_Spot',
    'Tomato Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato Tomato_mosaic_virus',
    'Tomato healthy']

def getImg(path):
    img = Image.open(path)
    return img

def preprocessImg(img):
    mytransform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])
    img = mytransform(img)
    img = img.view(1,3,256,256)
    return img

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)  
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True) 
        self.conv4 = ConvBlock(256, 512, pool=True) 
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))
        
    def forward(self, xb): 
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

def model_initialize(in_channels,num_diseases):
    model = ResNet9(in_channels,num_diseases)
    return model

def inference(img_path):
    img = getImg(img_path)
    img = preprocessImg(img=img)
    Model = model_initialize(3,len(dictlabel))
    Model.load_state_dict(torch.load('Leaf-Diseases.pth',map_location=torch.device('cpu')))
    Model.eval()
    with torch.no_grad():
        y = Model(img)

    _,pred = torch.max(y,dim=1)
    return dictlabel[pred[0].item()]