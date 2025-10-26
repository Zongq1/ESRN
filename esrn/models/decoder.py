"""
Decoder Module
Decodes symbolic reasoning states to final predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymbolicDecoder(nn.Module):
    """
    Decodes symbolic states to answer predictions
    """
    def __init__(
        self,
        embed_dim=64,
        hidden_dim=256,
        num_classes=8,
        spatial_size=(40, 40)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.spatial_size = spatial_size
        
        # Spatial feature aggregation
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Global reasoning head
        self.global_reasoning = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Answer classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Auxiliary reconstruction head (for training)
        self.reconstruction = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, symbols, return_recon=False):
        """
        Args:
            symbols: (B, D, H, W) - symbolic states from rule learner
            return_recon: whether to return reconstruction
        Returns:
            logits: (B, num_classes) - answer logits
            recon: (B, 1, H', W') - reconstructed image (if return_recon)
        """
        B, D, H, W = symbols.shape
        
        # Global pooling for classification
        pooled = self.spatial_pool(symbols).squeeze(-1).squeeze(-1)  # (B, D)
        
        # Global reasoning
        features = self.global_reasoning(pooled)  # (B, hidden_dim)
        
        # Classification
        logits = self.classifier(features)  # (B, num_classes)
        
        if return_recon:
            # Reconstruction for auxiliary loss
            recon = self.reconstruction(symbols)  # (B, 1, H'*8, W'*8)
            return logits, recon
        
        return logits


class MultiHeadDecoder(nn.Module):
    """
    Multi-head decoder for different reasoning tasks
    """
    def __init__(
        self,
        embed_dim=64,
        hidden_dim=256,
        num_classes=8,
        num_heads=4
    ):
        super().__init__()
        self.num_heads = num_heads
        
        # Multiple decoding heads
        self.heads = nn.ModuleList([
            SymbolicDecoder(embed_dim, hidden_dim, num_classes)
            for _ in range(num_heads)
        ])
        
        # Head selection/fusion
        self.head_fusion = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_heads),
            nn.Softmax(dim=-1)
        )
        
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, symbols):
        """
        Args:
            symbols: (B, D, H, W) - symbolic states
        Returns:
            logits: (B, num_classes) - fused predictions
            head_weights: (B, num_heads) - head attention weights
        """
        B = symbols.size(0)
        
        # Get predictions from all heads
        head_logits = []
        for head in self.heads:
            logits = head(symbols)
            head_logits.append(logits)
        
        head_logits = torch.stack(head_logits, dim=1)  # (B, num_heads, num_classes)
        
        # Calculate head weights
        pooled = self.spatial_pool(symbols).squeeze(-1).squeeze(-1)  # (B, D)
        head_weights = self.head_fusion(pooled)  # (B, num_heads)
        
        # Fuse predictions
        head_weights = head_weights.unsqueeze(-1)  # (B, num_heads, 1)
        logits = (head_logits * head_weights).sum(dim=1)  # (B, num_classes)
        
        return logits, head_weights.squeeze(-1)


class ContrastiveDecoder(nn.Module):
    """
    Decoder with contrastive learning for better answer discrimination
    """
    def __init__(
        self,
        embed_dim=64,
        hidden_dim=256,
        num_classes=8,
        temperature=0.07
    ):
        super().__init__()
        self.temperature = temperature
        
        # Context encoder (for 8 context panels)
        self.context_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Answer encoder (for each candidate answer)
        self.answer_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, context_symbols, answer_symbols=None):
        """
        Args:
            context_symbols: (B, D, H, W) - context reasoning state
            answer_symbols: (B, 8, D, H, W) - candidate answer symbols
        Returns:
            logits: (B, num_classes)
            similarity: (B, 8) - context-answer similarity (if answers provided)
        """
        # Encode context
        context_feat = self.context_encoder(context_symbols)  # (B, hidden_dim)
        
        # Classification
        logits = self.classifier(context_feat)  # (B, num_classes)
        
        if answer_symbols is not None:
            B, num_answers, D, H, W = answer_symbols.shape
            
            # Encode all candidate answers
            answer_symbols_flat = answer_symbols.view(B * num_answers, D, H, W)
            answer_feat = self.answer_encoder(answer_symbols_flat)  # (B*8, hidden_dim)
            answer_feat = answer_feat.view(B, num_answers, -1)  # (B, 8, hidden_dim)
            
            # Compute similarity for contrastive learning
            context_proj = self.projection(context_feat)  # (B, 128)
            answer_proj = self.projection(answer_feat.view(B * num_answers, -1))  # (B*8, 128)
            answer_proj = answer_proj.view(B, num_answers, -1)  # (B, 8, 128)
            
            # Cosine similarity
            context_proj = F.normalize(context_proj, dim=-1)
            answer_proj = F.normalize(answer_proj, dim=-1)
            
            similarity = torch.einsum('bd,bnd->bn', context_proj, answer_proj)  # (B, 8)
            similarity = similarity / self.temperature
            
            return logits, similarity
        
        return logits


if __name__ == "__main__":
    print("Testing Decoder Modules...\n")
    
    # Test SymbolicDecoder
    print("=" * 50)
    print("1. Testing SymbolicDecoder")
    print("=" * 50)
    decoder = SymbolicDecoder(
        embed_dim=64,
        hidden_dim=256,
        num_classes=8
    )
    
    symbols = torch.randn(4, 64, 40, 40)
    logits, recon = decoder(symbols, return_recon=True)
    
    print(f"Input symbols: {symbols.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Reconstruction: {recon.shape}")
    print(f"Predicted classes: {logits.argmax(dim=-1).tolist()}")
    
    # Test MultiHeadDecoder
    print("\n" + "=" * 50)
    print("2. Testing MultiHeadDecoder")
    print("=" * 50)
    multi_decoder = MultiHeadDecoder(
        embed_dim=64,
        hidden_dim=256,
        num_classes=8,
        num_heads=4
    )
    
    logits, head_weights = multi_decoder(symbols)
    print(f"Output logits: {logits.shape}")
    print(f"Head weights: {head_weights.shape}")
    print(f"Head weight distribution (sample): {head_weights[0].tolist()}")
    
    # Test ContrastiveDecoder
    print("\n" + "=" * 50)
    print("3. Testing ContrastiveDecoder")
    print("=" * 50)
    contrast_decoder = ContrastiveDecoder(
        embed_dim=64,
        hidden_dim=256,
        num_classes=8
    )
    
    context_symbols = torch.randn(4, 64, 40, 40)
    answer_symbols = torch.randn(4, 8, 64, 40, 40)  # 8 candidate answers
    
    logits, similarity = contrast_decoder(context_symbols, answer_symbols)
    print(f"Context symbols: {context_symbols.shape}")
    print(f"Answer symbols: {answer_symbols.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Similarity scores: {similarity.shape}")
    print(f"Similarity sample: {similarity[0].tolist()}")
    
    print("\n✅ All decoder modules working correctly!")