"""
Disentangled Encoder - 语义解耦编码器
目标：将panel分解为shape, size, color等独立语义属性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeEncoder(nn.Module):
    """单个属性编码器（shape/size/color共享架构）"""
    def __init__(self, in_channels=1, attr_dim=32):
        super().__init__()
        
        # 轻量级CNN
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),  # 160->80
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 80->40
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 40->20
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 投影到属性空间
        self.proj = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, attr_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, 160, 160)
        Returns:
            attr: (B, attr_dim)
        """
        feat = self.conv(x)  # (B, 64, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (B, 64)
        attr = self.proj(feat)  # (B, attr_dim)
        return attr


class DisentangledEncoder(nn.Module):
    """
    解耦编码器 - 核心创新
    
    设计理念：
    - 不同属性用不同encoder（不共享权重）
    - 通过对比学习强制解耦
    - 输出可解释的语义向量
    """
    def __init__(self, attr_dim=32):
        super().__init__()
        
        self.attr_dim = attr_dim
        
        # 三个独立的属性编码器
        self.shape_encoder = AttributeEncoder(attr_dim=attr_dim)
        self.size_encoder = AttributeEncoder(attr_dim=attr_dim)
        self.color_encoder = AttributeEncoder(attr_dim=attr_dim)
        
        # 对比学习用的投影头
        self.shape_proj = nn.Sequential(
            nn.Linear(attr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.size_proj = nn.Sequential(
            nn.Linear(attr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.color_proj = nn.Sequential(
            nn.Linear(attr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def forward(self, panel):
        """
        Args:
            panel: (B, 1, 160, 160)
        
        Returns:
            semantics: dict {
                'shape': (B, attr_dim),
                'size': (B, attr_dim),
                'color': (B, attr_dim),
                'combined': (B, attr_dim*3)
            }
        """
        # 编码三个属性
        shape = self.shape_encoder(panel)
        size = self.size_encoder(panel)
        color = self.color_encoder(panel)
        
        # 组合
        combined = torch.cat([shape, size, color], dim=1)  # (B, attr_dim*3)
        
        return {
            'shape': shape,
            'size': size,
            'color': color,
            'combined': combined
        }
    
    def get_contrastive_features(self, panel):
        """获取对比学习特征（用于训练）"""
        semantics = self.forward(panel)
        
        return {
            'shape': self.shape_proj(semantics['shape']),
            'size': self.size_proj(semantics['size']),
            'color': self.color_proj(semantics['color'])
        }


def contrastive_loss(z1, z2, temperature=0.5):
    """
    对比学习损失（SimCLR风格）
    
    正样本：同一panel的增强版本
    负样本：batch内其他panels
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # 计算相似度矩阵
    sim_matrix = torch.matmul(z1, z2.T) / temperature
    
    # 对角线是正样本
    batch_size = z1.size(0)
    labels = torch.arange(batch_size).to(z1.device)
    
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss


if __name__ == "__main__":
    print("Testing Disentangled Encoder...")
    
    encoder = DisentangledEncoder(attr_dim=32)
    
    # 测试
    panel = torch.randn(4, 1, 160, 160)
    semantics = encoder(panel)
    
    print(f"✅ Input: {panel.shape}")
    print(f"✅ Shape features: {semantics['shape'].shape}")
    print(f"✅ Size features: {semantics['size'].shape}")
    print(f"✅ Color features: {semantics['color'].shape}")
    print(f"✅ Combined: {semantics['combined'].shape}")
    print(f"✅ Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print("\n✅ Disentangled Encoder测试通过!")