import torch
from torch import nn
from transformers import ViTModel
from PIL import Image
import numpy as np

class CompNet(nn.Module):
    def __init__(self,
                 img_dim=768,
                 hidden_dim=16,
                 class_num=2,
                 layer_num=2,
                 vit_version='google/vit-base-patch16-224'):
        super().__init__()
        self.vitmodel = ViTModel.from_pretrained(vit_version)
        for param in self.vitmodel.parameters():
            param.requires_grad = False
        
        mlp = []
        #for i in range(layer_num):
        #    input_dim=2*hidden_dim+18 if i==0 else hidden_dim
        #    output_dim=hidden_dim if i!=layer_num-1 else class_num
        #    mlp.append(nn.Linear(input_dim, output_dim))
        #    if i!=layer_num-1:
        #        mlp.append(nn.ReLU())
        mlp.append(nn.Linear(18, 18))
        mlp.append(nn.ReLU())
        mlp.append(nn.Linear(18, 18))
        mlp.append(nn.ReLU())
        mlp.append(nn.Linear(18, 18))
        mlp.append(nn.ReLU())
        mlp.append(nn.Linear(18, 2))

        self.mlp = nn.Sequential(*mlp)
        self.img_layer=nn.Linear(img_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        profile_image=x["profile_image"]
        profile_banner=x["profile_banner"]
        theme=x["theme"]
        #followers_count=x["followers_count"]
        #friends_count=x["friends_count"]
        #listed_count=x["listed_count"]
        #statuses_count=x["statuses_count"]
        #favourites_count=x["favourites_count"]
        #profile_image = self.vitmodel(profile_image).last_hidden_state[:,0,:]
        #profile_banner = self.vitmodel(profile_banner).last_hidden_state[:,0,:]
        #x = torch.cat((profile_image,profile_banner,theme,followers_count,friends_count,listed_count,statuses_count,favourites_count),dim=1)
        #x = torch.cat((x['bool_features'],x['num_features']),dim=1)
        x = torch.cat((x['bool_features'],x['num_features']),dim=1)
        #x1=self.img_layer(profile_image)
        #x2=self.img_layer(profile_banner)
        #x=torch.cat((x,x1,x2),dim=1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x
