"""
ESRN Fixed - 正确处理candidate answers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .encoder import SymbolicEncoder
    from .rule_learner import RuleLearner
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from esrn.models.encoder import SymbolicEncoder
    from esrn.models.rule_learner import RuleLearner


class ESRNFixed(nn.Module):
    """
    修复版ESRN - 正确评估8个candidate answers
    """
    def __init__(
        self,
        in_channels=1,
        encoder_hidden_dims=[64, 128, 256],
        vocab_size=32,
        embed_dim=64,
        num_rules=8,
        rule_hidden_dim=128,
        max_steps=5,
        commitment_cost=0.05
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Encoder
        self.encoder = SymbolicEncoder(
            in_channels=in_channels,
            hidden_dims=encoder_hidden_dims,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            commitment_cost=commitment_cost
        )
        
        # Rule Learner
        self.rule_learner = RuleLearner(
            embed_dim=embed_dim,
            num_rules=num_rules,
            hidden_dim=rule_hidden_dim,
            max_steps=max_steps
        )
        
        # Scoring network: 评估 reasoning_state 和 candidate 的匹配度
        self.scorer = nn.Sequential(
            nn.Conv2d(embed_dim * 2, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, context_panels, answer_panels, return_details=False):
        """
        Args:
            context_panels: (B, 8, C, H, W)
            answer_panels: (B, 8, C, H, W)
        
        Returns:
            logits: (B, 8) - scores for each candidate
        """
        B = context_panels.size(0)
        
        # Encode context
        context_flat = context_panels.view(B * 8, *context_panels.shape[2:])
        context_symbols, vq_loss, perplexity = self.encoder(context_flat)
        
        _, D, H, W = context_symbols.shape
        context_symbols = context_symbols.view(B, 8, D, H, W)
        
        # Aggregate and reason
        aggregated = context_symbols.mean(dim=1)  # (B, D, H, W)
        reasoning_state, step_info = self.rule_learner(aggregated)
        
        # Encode all candidates
        answer_flat = answer_panels.view(B * 8, *answer_panels.shape[2:])
        answer_symbols, _, _ = self.encoder(answer_flat)
        answer_symbols = answer_symbols.view(B, 8, D, H, W)
        
        # Score each candidate
        scores = []
        for i in range(8):
            candidate = answer_symbols[:, i]  # (B, D, H, W)
            
            # Concatenate reasoning state with candidate
            combined = torch.cat([reasoning_state, candidate], dim=1)  # (B, 2*D, H, W)
            
            # Score this combination
            score = self.scorer(combined)  # (B, 1)
            scores.append(score)
        
        logits = torch.cat(scores, dim=1)  # (B, 8)
        
        if return_details:
            details = {
                'context_symbols': context_symbols,
                'reasoning_state': reasoning_state,
                'answer_symbols': answer_symbols,
                'vq_loss': vq_loss,
                'perplexity': perplexity
            }
            return logits, details
        
        return logits


if __name__ == "__main__":
    print("Testing ESRN Fixed...")
    
    model = ESRNFixed()
    
    context = torch.randn(4, 8, 1, 160, 160)
    answers = torch.randn(4, 8, 1, 160, 160)
    
    logits = model(context, answers)
    
    print(f"✅ Input: context={context.shape}, answers={answers.shape}")
    print(f"✅ Output logits: {logits.shape}")
    print(f"✅ Predictions: {logits.argmax(dim=1).tolist()}")
    print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n✅ Test passed!")