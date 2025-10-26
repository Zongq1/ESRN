"""
Rule Learner Module
Learns local transformation rules for symbolic reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalRuleModule(nn.Module):
    """
    Single local rule module that learns one transformation pattern
    """
    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Rule pattern matcher
        self.pattern_query = nn.Linear(embed_dim, hidden_dim)
        self.pattern_key = nn.Linear(embed_dim, hidden_dim)
        
        # Rule transformation
        self.transform = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Rule confidence predictor
        self.confidence = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, N, D) - symbolic states
        Returns:
            transformed: (B, N, D) - transformed states
            conf: (B, N, 1) - rule confidence
        """
        # Calculate pattern matching score
        query = self.pattern_query(x)  # (B, N, H)
        key = self.pattern_key(x)      # (B, N, H)
        
        # Self-attention for local pattern detection
        attn_scores = torch.matmul(query, key.transpose(-2, -1))  # (B, N, N)
        attn_scores = attn_scores / (self.pattern_query.out_features ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply transformation
        context = torch.matmul(attn_weights, x)  # (B, N, D)
        transformed = self.transform(context)
        
        # Calculate confidence
        conf = self.confidence(x)  # (B, N, 1)
        
        return transformed, conf


class RuleLearner(nn.Module):
    """
    Learns multiple local rules and composes them adaptively
    """
    def __init__(
        self,
        embed_dim=64,
        num_rules=32,
        hidden_dim=128,
        max_steps=5
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_rules = num_rules
        self.max_steps = max_steps
        
        # Multiple rule modules
        self.rules = nn.ModuleList([
            LocalRuleModule(embed_dim, hidden_dim)
            for _ in range(num_rules)
        ])
        
        # Rule selection gate
        self.rule_gate = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_rules)
        )
        
        # Step controller (decides when to stop reasoning)
        self.step_controller = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, symbols, adaptive=True):
        """
        Args:
            symbols: (B, D, H, W) - symbolic states from encoder
            adaptive: whether to use adaptive step control
        Returns:
            final_state: (B, D, H, W) - final reasoning state
            step_info: dict with reasoning trace
        """
        B, D, H, W = symbols.shape
        
        # Reshape to sequence: (B, N, D) where N = H * W
        x = symbols.view(B, D, -1).transpose(1, 2)  # (B, N, D)
        N = x.size(1)
        
        # Reasoning trace
        states = [x]
        rule_selections = []
        confidences = []
        
        for step in range(self.max_steps):
            # Gate: select which rules to apply
            gate_scores = self.rule_gate(x.mean(dim=1))  # (B, num_rules)
            gate_weights = F.softmax(gate_scores, dim=-1)  # (B, num_rules)
            
            # Apply all rules and combine
            rule_outputs = []
            rule_confs = []
            
            for rule in self.rules:
                transformed, conf = rule(x)
                rule_outputs.append(transformed)
                rule_confs.append(conf)
            
            # Stack and weight by gate
            rule_outputs = torch.stack(rule_outputs, dim=1)  # (B, R, N, D)
            rule_confs = torch.stack(rule_confs, dim=1)      # (B, R, N, 1)
            
            # Weighted combination
            gate_weights = gate_weights.view(B, self.num_rules, 1, 1)
            x_new = (rule_outputs * gate_weights).sum(dim=1)  # (B, N, D)
            
            # Average confidence
            avg_conf = (rule_confs * gate_weights).sum(dim=1)  # (B, N, 1)
            
            # Update state with residual connection
            x = x + x_new
            
            states.append(x)
            rule_selections.append(gate_weights.squeeze(-1).squeeze(-1))
            confidences.append(avg_conf.mean())
            
            # Adaptive stopping
            if adaptive:
                stop_prob = self.step_controller(x.mean(dim=1))  # (B, 1)
                if stop_prob.mean() > 0.8:  # High confidence to stop
                    break
        
        # Reshape back to spatial: (B, D, H, W)
        final_state = x.transpose(1, 2).view(B, D, H, W)
        
        step_info = {
            'num_steps': step + 1,
            'states': states,
            'rule_selections': rule_selections,
            'confidences': confidences
        }
        
        return final_state, step_info
    
    def get_rule_usage(self, symbols):
        """
        Analyze which rules are used for given inputs
        """
        B, D, H, W = symbols.shape
        x = symbols.view(B, D, -1).transpose(1, 2)
        
        gate_scores = self.rule_gate(x.mean(dim=1))
        gate_weights = F.softmax(gate_scores, dim=-1)
        
        return gate_weights  # (B, num_rules)


if __name__ == "__main__":
    # Test the rule learner
    rule_learner = RuleLearner(
        embed_dim=64,
        num_rules=32,
        hidden_dim=128,
        max_steps=5
    )
    
    # Dummy symbolic states (from encoder)
    symbols = torch.randn(4, 64, 40, 40)
    
    # Forward pass
    final_state, step_info = rule_learner(symbols, adaptive=True)
    
    print(f"Input symbols shape: {symbols.shape}")
    print(f"Final state shape: {final_state.shape}")
    print(f"Number of reasoning steps: {step_info['num_steps']}")
    print(f"Rule confidences: {[f'{c.item():.4f}' for c in step_info['confidences']]}")
    
    # Analyze rule usage
    rule_usage = rule_learner.get_rule_usage(symbols)
    print(f"\nRule usage distribution (batch mean):")
    print(f"Top 5 rules: {torch.topk(rule_usage.mean(0), 5).indices.tolist()}")
    print(f"Top 5 weights: {torch.topk(rule_usage.mean(0), 5).values.tolist()}")