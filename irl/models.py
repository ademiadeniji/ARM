"""
Define MLP layers on top of CLIP
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import clip
from PIL import Image
import requests
from glob import glob
from natsort import natsorted
import torch.nn as nn 
import copy
import random 
from torch.optim import Adam
from einops import rearrange, repeat 
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
try:
    from torchvision.transforms import InterpolationMode 
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class RewardMLP(nn.Module):
    def __init__(self, clip_model, hidden_layers=[256,512], scale_logit=False, predict_logit=False):
        super().__init__()
        self.predict_logit = predict_logit
        self.scale_logit = scale_logit
        if predict_logit:
            print('Predicting logits with hidden sizes {} and output dim 1'.format(
                hidden_layers))
            linears = [nn.Linear(768*2, hidden_layers[0])]
            for i in range(1, len(hidden_layers)):
                linears.extend([
                    nn.ReLU(), 
                    nn.Linear(hidden_layers[i-1], hidden_layers[i])
                ])
            
            linears += [
                nn.ReLU(), 
                nn.Linear(hidden_layers[-1], 1),
                nn.Tanh()
                ]
            self.mlp = nn.Sequential(*linears)
        else:
            linears = [nn.Linear(768, hidden_layers[0])]
            for i in range(1, len(hidden_layers)):
                linears.extend([
                    nn.ReLU(), 
                    nn.Linear(hidden_layers[i-1], hidden_layers[i]), 
                ])
            self.img_mlp = nn.Sequential(*linears)
            self.text_mlp = nn.Sequential(*copy.deepcopy(linears)) 
            if self.scale_logit:
                self.logit_scale = copy.deepcopy(clip_model.logit_scale)
                self.logit_scale.requires_grad = True 
        self.clip_model = clip_model 
 
        for p in self.clip_model.parameters():
            p.requires_grad = False
    
    def forward(self, image, text):
 
        with torch.no_grad():
            clip_img = self.clip_model.encode_image(image).detach().to(torch.float32)
            clip_text = self.clip_model.encode_text(text).detach().to(torch.float32) #  self.text_mlp(clip_text.to(torch.float32))
         
        if self.predict_logit:
            clip_text = repeat(clip_text, '1 d -> bb d', bb=clip_img.shape[0])  
            logits = self.mlp(torch.cat((clip_img, clip_text), dim=-1)).squeeze(-1)
 
        else:
            image_features = self.img_mlp(clip_img)
            text_features = self.text_mlp(clip_text)
            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp() if self.scale_logit else 1.0
            logits = logit_scale * image_features @ text_features.t()
             
 
        return logits 
    
