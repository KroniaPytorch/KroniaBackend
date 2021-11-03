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
  ferti={0:{'fertilizer':'10-26-26','price':round(158/74.93,0),'shop':'https://www.amazon.in/Bharath-Agencies-Soluble-10-26-26-Gardening/dp/B07MXV321M/ref=sr_1_5?crid=33H7L00MKMLG5&dchild=1&keywords=10+26+26+npk+fertilizer&qid=1635699333&qsid=257-4806340-5008705&sprefix=10+26+2%2Caps%2C342&sr=8-5&sres=B07MXV321M%2CB07PRSDDBR%2CB08PPJCK81%2CB07Q2BN6Q8%2CB07MXVGRCF%2CB08PP1MKBJ%2CB07MXWB5BF%2CB07PN4YM1C%2CB07MXVM58V%2CB08KH94GPL%2CB07MXVTL8J%2CB08SQPML7S%2CB084TMWFLP%2CB082Q2H3KP%2CB084TM9LY5%2CB07MBZ25Y2%2CB07ZWZ4S3R%2CB08N47965C%2CB07PBR7W8Z%2CB08KDMMTNZ&srpt=FERTILIZER'},1:{'fertilizer':'14-35-14','price':round(349/74.93,0),'shop':'https://www.amazon.in/Soluble-14-35-14-Gardening-Bharath-Agencies/dp/B07L5TT47N/ref=sr_1_5?dchild=1&keywords=14+35+14+fertilizer&qid=1635699084&qsid=257-4806340-5008705&sr=8-5&sres=B07L5TT47N%2CB07L6DQS36%2CB07L5T4B5J%2CB08Y5W3SDH%2CB08Y2N768Q%2CB08Y5C3665%2CB08Y5Q6ZHQ%2CB08Y5C368T%2CB08Y2VK9QZ%2CB08Y5M9WD5%2CB08Y5L47S4%2CB08Y2VXT9W%2CB08Y5PD96H%2CB08Y5RTBSQ%2CB08Y2D6Y9P%2CB08Y5NVXGT%2CB00R03724C%2CB07ZWZ4S3R%2CB07PRXV7MJ%2CB08BJPBTFT&srpt=FERTILIZER'},2:{'fertilizer':'17-17-17','price':round(189/74.93,0),'shop':'https://www.amazon.in/enterprises%C2%AE-Specialty-Crystalline-Fertilizer-Nutrition/dp/B09C2ZZ5N2/ref=sr_1_3_sspa?dchild=1&keywords=17+17+fertilizer&qid=1635698956&sr=8-3-spons&psc=1&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUExMEhCTU9QSjBEMUwmZW5jcnlwdGVkSWQ9QTA3ODIwMDMySUhIVEM2TVpMWlkyJmVuY3J5cHRlZEFkSWQ9QTA4MzkzNzkxWFZPM05UTlhLN1hZJndpZGdldE5hbWU9c3BfYXRmJmFjdGlvbj1jbGlja1JlZGlyZWN0JmRvTm90TG9nQ2xpY2s9dHJ1ZQ=='},3:{'fertilizer':'20-20','price':round(169/74.93,0),'shop':'https://www.amazon.in/JD-FRESH-NPK-Crystalline-fertilizers/dp/B09K929D8Q/ref=sr_1_1_sspa?crid=3K9KBE4XK890M&dchild=1&keywords=20+20+fertilizer&qid=1635698916&sprefix=20+20+fert%2Clawngarden%2C314&sr=8-1-spons&psc=1&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUEyOEdMR0VXMVFST05IJmVuY3J5cHRlZElkPUEwOTAzNTI3MkxaQkdIV0xRS1BBNiZlbmNyeXB0ZWRBZElkPUEwOTc0OTAyQkk5NkdNSDhBVE5EJndpZGdldE5hbWU9c3BfYXRmJmFjdGlvbj1jbGlja1JlZGlyZWN0JmRvTm90TG9nQ2xpY2s9dHJ1ZQ=='},4:{'fertilizer':'28-28','price':round(450/74.93,0),'shop':'https://www.amazon.in/Growth-More-Enterprises-NPK-Fertilizer/dp/B08XQQQ3T8/ref=sr_1_4?dchild=1&keywords=28+28+fertilizer&qid=1635698798&qsid=257-4806340-5008705&s=garden&sr=1-4&sres=B08XQQQ3T8%2CB08XQQ6NMZ%2CB08XQQF3DM%2CB07FKF5DMG%2CB07FKFJFHF%2CB07FKFKNNY%2CB08XQQWMC3%2CB08XQQS18S%2CB08XQQPCGG%2CB08XQRLP48%2CB08XQRH612%2CB08XQPYDV8%2CB08XQRFPNQ%2CB08XQPQ8B6%2CB08XQQL768%2CB08XQQC6NW%2CB08XQQTGY2%2CB07T1TXSTH%2CB08519MCN4%2CB082WGG2QS&srpt=FERTILIZER'},5:{'fertilizer':'DAP','price':round(179/74.93,0),'shop':'https://www.amazon.in/Go-Garden-Purpose-Fertilizer-Gardening/dp/B071P39FJV/ref=asc_df_B071P39FJV/?tag=googleshopdes-21&linkCode=df0&hvadid=396987265806&hvpos=&hvnetw=g&hvrand=13545032045620602687&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9302138&hvtargid=pla-404914910802&psc=1&ext_vrnc=hi'},6:{'fertilizer':'Urea','price':round(129/74.93,0),'shop':'https://www.amazon.in/Shehri-Kisaan-Fertilizer-Essential-Fertilizers/dp/B07TMGCFD2/ref=sr_1_4?dchild=1&keywords=urea&qid=1635698686&qsid=257-4806340-5008705&s=garden&sr=1-4&sres=B07TMGCFD2%2CB08H8FXNJ9%2CB07YCM3ZZR%2CB082575GT4%2CB072JMDJ9S%2CB01MTCCDQF%2CB07PGG5LBH%2CB082WGZ3Q5%2CB07PF62TBB%2CB07Q3N3PFP%2CB0868CZ9QW%2CB07PF61T2T%2CB07YHP5CY7%2CB07P52FHRN%2CB07NZJ9CXD%2CB07YHSNNDQ%2CB08KZC8G53%2CB07YHRBF9S%2CB097K2CTKY%2CB07PH8LWYQ&srpt=FERTILIZER'}}

  model2 = MulticlassClassification(num_feature = 7, num_class=7)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model2.to(device)
  model2.load_state_dict(torch.load('fertilizer.pth',map_location=torch.device('cpu')))

  model2.eval()
  data=[]
  data+=[req_data['temp']]
  data+=[req_data['humid']]
  data+=[req_data['moisture']]
  data+=[req_data['soil_type']]
  data+=[req_data['nitro']]
  data+=[req_data['pota']]
  data+=[req_data['phosp']]








  # for i in req_data:
  #   data+=[req_data[i]]
  
  # print(data)
 # data = np.array([[26, 52, 38, 4, 37, 0, 0]])
#   data = data.astype(np.float32)
#   data=torch.tensor(data)
  data = np.array([data])

  data = data.astype(np.float32)
  data=torch.tensor(data)

  with torch.no_grad():
    prediction = model2(data)
    _, y_pred_tags = torch.topk(prediction,4,dim=1)
    val=y_pred_tags.cpu().numpy()

  dict={}
  dict['recommended']=ferti[val[0][0]]
  dict['similar']=[]
  for i in val[0]:
    if dict['recommended']['fertilizer']!=str(ferti[i]['fertilizer']):
      dict['similar']+=[str(ferti[i])]


  return dict