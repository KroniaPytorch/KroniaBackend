import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms, utils, datasets
from PIL import Image


class RpsClassifier(nn.Module):
    def __init__(self):
        super(RpsClassifier, self).__init__()
        self.block1 = self.conv_block(c_in=3, c_out=256, dropout=0.1, kernel_size=5, stride=1, padding=2)
        self.block2 = self.conv_block(c_in=256, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=75, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = self.lastcnn(x)
        return x
    def conv_block(self, c_in, c_out, dropout, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        return seq_block


def predict(req_data):
    model2 = RpsClassifier()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model2.to(device)
    model2.load_state_dict(torch.load('wheatDisease.pth',map_location=torch.device('cpu')))

    model2.eval()

    image_transforms = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    image=Image.open(req_data)

    image=image_transforms(image)

    image = image.unsqueeze(0)

    dict={0:'Healthy',1:'Leaf Rust',2:'Stem Rust'}

    predictions=model2(image)

    _, y_pred_tag = torch.max(predictions, dim = 1)

    val=y_pred_tag[0][0].cpu().numpy()

    return dict[val[0]]
