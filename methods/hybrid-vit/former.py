import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SequencePooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention_weights = nn.Linear(dim, 1)

    def forward(self, x):
        weights = F.softmax(self.attention_weights(x), dim=1)
        pooled = torch.sum(weights * x, dim=1) 
        return pooled

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class LocalAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class HybridVisionFormer(nn.Module):
    def __init__(self, num_classes=7, embed_dim=512):
        super().__init__()
        
        # 1. UPGRADE: Dùng ResNet50 với trọng số V2 mạnh nhất hiện tại
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # 2. ResNet50 xuất ra 2048 channels. Cần 1x1 Conv nén về 512 channels
        self.proj = nn.Conv2d(2048, embed_dim, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dim)
        
        # 3. Local Attention
        self.attention = LocalAttentionBlock(embed_dim)

        # 4. Transformer Global
        self.num_patches = 49 
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        
        self.pos_drop = nn.Dropout(p=0.2) # Tăng nhẹ dropout chống overfit cho model lớn

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=1024, batch_first=True, dropout=0.3
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 5. Sequence Pooling
        self.seq_pool = SequencePooling(embed_dim)
        
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5), 
            nn.Linear(512, num_classes)
        )

    def forward(self, image):
        # Trích xuất CNN Features: -> (Batch, 2048, 7, 7)
        x = self.backbone(image) 
        
        # Nén xuống 512: -> (Batch, 512, 7, 7)
        x = F.relu(self.proj_bn(self.proj(x)))
        
        # Local Attention
        x = self.attention(x)    

        # Biến đổi thành Token sequence: -> (Batch, 49, 512)
        x = x.flatten(2).transpose(1, 2)
        
        # Transformer Global Modeling
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.encoder(x)
        x = self.norm(x)
        
        final_rep = self.seq_pool(x)
        return self.mlp_head(final_rep)