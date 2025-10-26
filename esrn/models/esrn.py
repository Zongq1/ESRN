"""
ESRN - Emergent Symbolic Reasoning Network
Complete model integrating encoder, rule learner, and decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Relative imports (for package use)
    from .encoder import SymbolicEncoder
    from .rule_learner import RuleLearner
    from .decoder import SymbolicDecoder, MultiHeadDecoder, ContrastiveDecoder
except ImportError:
    # Absolute imports (for direct execution)
    from encoder import SymbolicEncoder
    from rule_learner import RuleLearner
    from decoder import SymbolicDecoder, MultiHeadDecoder, ContrastiveDecoder


class ESRN(nn.Module):
    """
    Complete ESRN model for visual reasoning
    
    Architecture:
        Input (9 panels) -> Encoder -> Symbolic States
                         -> Rule Learner -> Reasoning States
                         -> Decoder -> Answer Prediction
    """
    def __init__(
        self,
        # Encoder config
        in_channels=1,
        encoder_hidden_dims=[64, 128, 256],
        vocab_size=512,
        embed_dim=64,
        
        # Rule Learner config
        num_rules=32,
        rule_hidden_dim=128,
        max_steps=5,
        
        # Decoder config
        decoder_type='basic',  # 'basic', 'multi_head', 'contrastive'
        decoder_hidden_dim=256,
        num_classes=8,
        num_decoder_heads=4,
        
        # Training config
        commitment_cost=0.25,
        adaptive_steps=True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.decoder_type = decoder_type
        self.adaptive_steps = adaptive_steps
        
        # Symbolic Encoder
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
        
        # Decoder
        if decoder_type == 'basic':
            self.decoder = SymbolicDecoder(
                embed_dim=embed_dim,
                hidden_dim=decoder_hidden_dim,
                num_classes=num_classes
            )
        elif decoder_type == 'multi_head':
            self.decoder = MultiHeadDecoder(
                embed_dim=embed_dim,
                hidden_dim=decoder_hidden_dim,
                num_classes=num_classes,
                num_heads=num_decoder_heads
            )
        elif decoder_type == 'contrastive':
            self.decoder = ContrastiveDecoder(
                embed_dim=embed_dim,
                hidden_dim=decoder_hidden_dim,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
    
    def forward(self, context_panels, answer_panels=None, return_details=False):
        """
        Forward pass through complete ESRN
        
        Args:
            context_panels: (B, 8, C, H, W) - 8 context panels
            answer_panels: (B, 8, C, H, W) - 8 candidate answer panels (optional)
            return_details: whether to return intermediate states
        
        Returns:
            logits: (B, num_classes) - answer predictions
            details: dict with intermediate states (if return_details=True)
        """
        B = context_panels.size(0)
        
        # Encode all context panels
        context_flat = context_panels.view(B * 8, *context_panels.shape[2:])  # (B*8, C, H, W)
        context_symbols, vq_loss, perplexity = self.encoder(context_flat)
        
        # Reshape back: (B, 8, D, H', W')
        _, D, H, W = context_symbols.shape
        context_symbols = context_symbols.view(B, 8, D, H, W)
        
        # Aggregate context (e.g., mean pooling across 8 panels)
        aggregated_context = context_symbols.mean(dim=1)  # (B, D, H', W')
        
        # Apply rule-based reasoning
        reasoning_state, step_info = self.rule_learner(
            aggregated_context, 
            adaptive=self.adaptive_steps
        )
        
        # Decode to answer prediction
        if self.decoder_type == 'contrastive' and answer_panels is not None:
            # Encode answer panels
            answer_flat = answer_panels.view(B * 8, *answer_panels.shape[2:])
            answer_symbols, _, _ = self.encoder(answer_flat)
            answer_symbols = answer_symbols.view(B, 8, D, H, W)
            
            logits, similarity = self.decoder(reasoning_state, answer_symbols)
        
        elif self.decoder_type == 'multi_head':
            logits, head_weights = self.decoder(reasoning_state)
            if not return_details:
                return logits
        
        else:
            logits = self.decoder(reasoning_state)
        
        if not return_details:
            return logits
        
        # Return detailed information
        details = {
            'context_symbols': context_symbols,
            'reasoning_state': reasoning_state,
            'vq_loss': vq_loss,
            'perplexity': perplexity,
            'num_reasoning_steps': step_info['num_steps'],
            'rule_confidences': step_info['confidences'],
            'rule_selections': step_info['rule_selections']
        }
        
        if self.decoder_type == 'contrastive' and answer_panels is not None:
            details['answer_similarity'] = similarity
        
        if self.decoder_type == 'multi_head':
            details['decoder_head_weights'] = head_weights
        
        return logits, details
    
    def get_symbolic_representation(self, panels):
        """
        Get symbolic representation of input panels
        Useful for analysis and visualization
        """
        B = panels.size(0) if panels.dim() == 5 else 1
        if panels.dim() == 4:
            panels = panels.unsqueeze(0)
        
        panels_flat = panels.view(-1, *panels.shape[2:])
        symbols, _, _ = self.encoder(panels_flat)
        
        return symbols
    
    def analyze_reasoning_process(self, context_panels):
        """
        Detailed analysis of the reasoning process
        """
        _, details = self.forward(context_panels, return_details=True)
        
        analysis = {
            'encoding': {
                'perplexity': details['perplexity'].item(),
                'vq_loss': details['vq_loss'].item()
            },
            'reasoning': {
                'num_steps': details['num_reasoning_steps'],
                'confidences': [c.item() for c in details['rule_confidences']],
                'avg_confidence': sum(c.item() for c in details['rule_confidences']) / len(details['rule_confidences'])
            }
        }
        
        # Rule usage analysis
        if details['rule_selections']:
            rule_usage = torch.stack(details['rule_selections']).mean(dim=0)  # Average across steps
            top_rules = torch.topk(rule_usage[0], k=5)
            analysis['reasoning']['top_rules'] = {
                'indices': top_rules.indices.tolist(),
                'weights': top_rules.values.tolist()
            }
        
        return analysis


class ESRNLightning(nn.Module):
    """
    PyTorch Lightning wrapper for ESRN
    (Placeholder for future Lightning integration)
    """
    def __init__(self, esrn_config):
        super().__init__()
        self.model = ESRN(**esrn_config)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


if __name__ == "__main__":
    print("Testing Complete ESRN Model...\n")
    print("=" * 60)
    
    # Create ESRN model
    model = ESRN(
        in_channels=1,
        encoder_hidden_dims=[64, 128],
        vocab_size=512,
        embed_dim=64,
        num_rules=32,
        rule_hidden_dim=128,
        max_steps=5,
        decoder_type='multi_head',
        decoder_hidden_dim=256,
        num_classes=8,
        adaptive_steps=True
    )
    
    print(f"✅ ESRN Model Created")
    print(f"   - Encoder: VQ-VAE with vocab_size=512")
    print(f"   - Rule Learner: 32 rules, max 5 steps")
    print(f"   - Decoder: Multi-head (4 heads)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - Total Parameters: {total_params:,}")
    print(f"   - Trainable Parameters: {trainable_params:,}")
    
    print("\n" + "=" * 60)
    print("Testing Forward Pass...")
    print("=" * 60)
    
    # Dummy RAVEN-style input
    batch_size = 4
    context_panels = torch.randn(batch_size, 8, 1, 160, 160)  # 8 context panels
    
    # Forward pass
    logits, details = model(context_panels, return_details=True)
    
    print(f"\n📥 Input:")
    print(f"   - Context panels: {context_panels.shape}")
    
    print(f"\n📤 Output:")
    print(f"   - Logits: {logits.shape}")
    print(f"   - Predictions: {logits.argmax(dim=-1).tolist()}")
    
    print(f"\n🔍 Reasoning Details:")
    print(f"   - VQ Loss: {details['vq_loss'].item():.4f}")
    print(f"   - Perplexity: {details['perplexity'].item():.2f}")
    print(f"   - Reasoning Steps: {details['num_reasoning_steps']}")
    print(f"   - Rule Confidences: {[f'{c.item():.4f}' for c in details['rule_confidences']]}")
    
    if 'decoder_head_weights' in details:
        print(f"   - Decoder Head Weights (sample): {details['decoder_head_weights'][0].tolist()}")
    
    print("\n" + "=" * 60)
    print("Testing Reasoning Analysis...")
    print("=" * 60)
    
    analysis = model.analyze_reasoning_process(context_panels)
    
    print(f"\n📊 Encoding Analysis:")
    print(f"   - Perplexity: {analysis['encoding']['perplexity']:.2f}")
    print(f"   - VQ Loss: {analysis['encoding']['vq_loss']:.4f}")
    
    print(f"\n🧠 Reasoning Analysis:")
    print(f"   - Steps: {analysis['reasoning']['num_steps']}")
    print(f"   - Avg Confidence: {analysis['reasoning']['avg_confidence']:.4f}")
    print(f"   - Top 5 Rules: {analysis['reasoning']['top_rules']['indices']}")
    print(f"   - Rule Weights: {[f'{w:.4f}' for w in analysis['reasoning']['top_rules']['weights']]}")
    
    print("\n" + "=" * 60)
    print("✅ ESRN Model Test Complete!")
    print("=" * 60)