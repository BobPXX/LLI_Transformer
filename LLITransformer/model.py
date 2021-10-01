import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from modules import Transformer, separated_conv

class LLI_Transformer(nn.Module):
    def __init__(self, num_classes=1000, img_size=224, in_chans=3, patch_size=16, patchify_inner_channels=16, hidden_size=768, heads=12, encoder_num=12, mlp_dim=3072):
        super(LLI_Transformer, self).__init__()
        self.num_classes=num_classes
        self.img_size=img_size
        self.in_chans=in_chans
        self.patch_size=patch_size
        self.patchify_inner_channels=patchify_inner_channels
        self.hidden_size=hidden_size
        self.heads=heads
        self.patchify_output_size=self.img_size//self.patch_size
        self.encoder_num=encoder_num
        self.mlp_dim=mlp_dim

        self.Patchify = separated_conv(input_size=self.img_size, patch_size=self.patch_size, in_channels=self.in_chans, inner_channels=self.patchify_inner_channels,out_channels=self.hidden_size)
        self.Transformer = Transformer(hidden_size=self.hidden_size, heads=self.heads, patchify_output_size=self.patchify_output_size, encoder_num=self.encoder_num,mlp_dim=self.mlp_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(self.hidden_size, mlp_dim),
            #nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            #nn.Linear(mlp_dim, mlp_dim),
            #nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(mlp_dim, num_classes)
        )
    def forward(self, x):
        x= self.Patchify(x)
        x= self.Transformer(x)
        x = self.mlp_head(x)
        return x