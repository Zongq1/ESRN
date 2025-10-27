"""
ESRN Continuous - 不使用VQ-VAE的简化版本
用于baseline对比和调试
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """简单的CNN编码器，输出连续特征"""
    def __init__(self, in_channels=1, hidden_dims=[64, 128, 256], embed_dim=64):
        super().__init__()
        
        layers = []
        prev_dim = in_channels
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(prev_dim, h_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(inplace=True)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        self.proj = nn.Conv2d(prev_dim, embed_dim, 1)
    
    def forward(self, x):
        features = self.encoder(x)
        return self.proj(features)


class SimpleReasoning(nn.Module):
    """简化的推理模块"""
    def __init__(self, embed_dim=64, hidden_dim=128):
        super().__init__()
        
        self.context_aggregator = nn.Sequential(
            nn.Conv2d(embed_dim * 8, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, embed_dim, 1)
        )
        
        self.reasoning = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, embed_dim, 3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, context_features):
        # context_features: (B, 8, D, H, W)
        B, N, D, H, W = context_features.shape
        
        # 拼接所有context
        context_cat = context_features.view(B, N * D, H, W)
        
        # 聚合
        aggregated = self.context_aggregator(context_cat)
        
        # 推理
        reasoning_state = self.reasoning(aggregated)
        
        return reasoning_state


class ESRNContinuous(nn.Module):
    """
    简化的连续特征版ESRN
    用于baseline和调试
    """
    def __init__(
        self,
        in_channels=1,
        hidden_dims=[64, 128, 256],
        embed_dim=64,
        reasoning_hidden_dim=128,
        num_classes=8
    ):
        super().__init__()
        
        self.encoder = SimpleCNN(in_channels, hidden_dims, embed_dim)
        self.reasoning = SimpleReasoning(embed_dim, reasoning_hidden_dim)
        
        # 全局池化 + 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, context_panels, answer_panels=None):
        """
        Args:
            context_panels: (B, 8, C, H, W)
            answer_panels: (B, 8, C, H, W) [unused in this version]
        
        Returns:
            logits: (B, num_classes)
        """
        B = context_panels.size(0)
        
        # 编码所有context panels
        context_flat = context_panels.view(B * 8, *context_panels.shape[2:])
        context_features = self.encoder(context_flat)
        
        # Reshape: (B, 8, D, H, W)
        _, D, H, W = context_features.shape
        context_features = context_features.view(B, 8, D, H, W)
        
        # 推理
        reasoning_state = self.reasoning(context_features)
        
        # 分类
        logits = self.classifier(reasoning_state)
        
        return logits


if __name__ == "__main__":
    print("Testing ESRN Continuous...")
    
    model = ESRNContinuous(
        in_channels=1,
        hidden_dims=[64, 128, 256],
        embed_dim=64,
        reasoning_hidden_dim=128,
        num_classes=8
    )
    
    # 测试
    context = torch.randn(4, 8, 1, 160, 160)
    logits = model(context)
    
    print(f"Input: {context.shape}")
    print(f"Output: {logits.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ Test passed!")