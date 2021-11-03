import torch
import torch.nn as nn
import numpy as np

class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x


def predict(req_data):
  crops={20:'rice', 11:'maize', 3:'chickpea',9:'kidneybeans', 18:'pigeonpeas',
       13:'mothbeans', 14:'mungbean', 2:'blackgram', 10:'lentil', 19:'pomegranate',
       1:'banana', 12:'mango', 7:'grapes', 21:'watermelon', 15:'muskmelon', 0:'apple',
       16:'orange', 17:'papaya', 4:'coconut', 6:'cotton', 8:'jute', 5:'coffee'}
  model2 = MulticlassClassification(num_feature = 7, num_class=22)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model2.to(device)
  model2.load_state_dict(torch.load('cropRecommender.pth',map_location=torch.device('cpu')))

  model2.eval()
  data=[]
  for i in req_data:
    if i=='nitro' or i=='potash' or i=='phosp':
      data+=[float(req_data[i])*4.535]
    else:
      data+=[req_data[i]]
      
  # print(data)

  
  # data = np.array([[83, 45, 60, 28, 70.3, 7.0, 150.9]])
  data = np.array([data])

  data = data.astype(np.float32)
  data=torch.tensor(data)

  with torch.no_grad():
    prediction = model2(data)
    _, y_pred_tags = torch.topk(prediction,6,dim=1)
    val=y_pred_tags.cpu().numpy()

  dict={}
  dict['recommended']=crops[val[0][0]]
  dict['similar']=[]
  for i in val[0]:
    if dict['recommended']!=str(crops[i]):
      dict['similar']+=[str(crops[i])]


  return dict