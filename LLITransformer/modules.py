import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class separated_conv(nn.Module):
    def __init__(self, input_size,patch_size,in_channels,inner_channels,out_channels):
        super(separated_conv, self).__init__()
        self.input_size=input_size
        self.patch_size = patch_size
        self.in_channels=in_channels
        self.inner_channels=inner_channels
        self.out_channels=out_channels

        self.conv_h1=nn.Conv2d(in_channels=self.in_channels, out_channels=self.inner_channels, kernel_size=(self.patch_size, 1),stride=(self.patch_size, 1),padding=(0,0),bias=False)
        self.conv_w1=nn.Conv2d(in_channels=self.inner_channels, out_channels=self.out_channels, kernel_size=(1,self.patch_size), stride=(1,self.patch_size),
                      padding=(0, 0),bias=False)
        self.conv_w2=nn.Conv2d(in_channels=self.in_channels, out_channels=self.inner_channels, kernel_size=(1,self.patch_size),stride=(1,self.patch_size),padding=(0,0),bias=False)
        self.conv_h2=nn.Conv2d(in_channels=self.inner_channels, out_channels=self.out_channels, kernel_size=(self.patch_size, 1), stride=(self.patch_size, 1),
                      padding=(0, 0),bias=False)

    def forward(self, x):
        x1=self.conv_h1(x)
        x1=self.conv_w1(x1)
        x2=self.conv_w2(x)
        x2=self.conv_h2(x2)
        return x1+x2

#AxialAttention module refer to: https://github.com/csrhddlam/axial-deeplab
class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups, kernel_size,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = nn.Conv1d(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)

        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1 #(14,14)
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # (N, W, C, H)
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size) #(128,14,14)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,kij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

class Encoder_Block(nn.Module):
    def __init__(self, hidden_size,heads,patchify_output_size,mlp_dim):
        super(Encoder_Block, self).__init__()
        self.hidden_size = hidden_size
        self.heads=heads
        self.patchify_output_size=patchify_output_size
        self.mlp_dim=mlp_dim

        self.height_attn = AxialAttention(in_planes=self.hidden_size, out_planes=self.hidden_size, groups=self.heads, kernel_size=self.patchify_output_size,width=False)
        self.width_attn = AxialAttention(in_planes=self.hidden_size, out_planes=self.hidden_size, groups=self.heads, kernel_size=self.patchify_output_size,width=True)
        self.BN= nn.BatchNorm2d(self.hidden_size)
        self.act=nn.GELU()
        self.conv1x1_up=nn.Conv2d(self.hidden_size, self.mlp_dim, kernel_size=1, stride=1, bias=False)
        self.conv1x1_down = nn.Conv2d(self.mlp_dim, self.hidden_size, kernel_size=1, stride=1, bias=False)
        self.dropout=nn.Dropout(0.1)
    def forward(self, x):
        h = x
        x = self.BN(x)
        x = self.height_attn(x)
        x = self.width_attn(x)
        x = x + h

        h = x
        x = self.BN(x)
        x = self.conv1x1_up(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv1x1_down(x)
        x = self.dropout(x)
        x = x + h
        return x

class Transformer(nn.Module):
    def __init__(self, hidden_size, heads,patchify_output_size,encoder_num,mlp_dim):
        super(Transformer, self).__init__()
        self.hidden_size = hidden_size
        self.heads=heads
        self.patchify_output_size=patchify_output_size
        self.encoder_num=encoder_num
        self.mlp_dim=mlp_dim

        self.Encoder_Block = Encoder_Block(hidden_size=self.hidden_size, heads=self.heads, patchify_output_size=self.patchify_output_size,mlp_dim=self.mlp_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        for _ in range(self.encoder_num):
            x= self.Encoder_Block(x)
        x=self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
