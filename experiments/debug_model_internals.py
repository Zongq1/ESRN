"""
Debug ESRN internal states
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from esrn.models.esrn import ESRN
from esrn.datasets.raven_dataset import RAVENDataset
from torch.utils.data import DataLoader

# 加载数据
dataset = RAVENDataset('data/RAVEN-10000', 'center_single', 'train')
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# 创建模型
model = ESRN(
    in_channels=1,
    encoder_hidden_dims=[64, 128, 256],
    vocab_size=64,
    embed_dim=64,
    num_rules=32,
    rule_hidden_dim=128,
    max_steps=5,
    decoder_type='basic',
    decoder_hidden_dim=256,
    num_classes=8
).cuda()

# 加载训练好的模型
checkpoint = torch.load('checkpoints/best_model.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 获取一个batch
context, answers, target, _ = next(iter(loader))
context = context.cuda()
answers = answers.cuda()

print("="*70)
print("ESRN Internal States Debugging")
print("="*70)

with torch.no_grad():
    # 1. Encoder输出
    B = context.size(0)
    all_panels = torch.cat([context.view(B*8, 1, 160, 160), 
                            answers.view(B*8, 1, 160, 160)], dim=0)
    
    # 修正：使用正确的属性名 model.encoder
    symbols, vq_loss, perplexity = model.encoder(all_panels)
    print(f"\n1. Symbolic Encoder:")
    print(f"   VQ Loss: {vq_loss.item():.6f}")
    print(f"   Perplexity: {perplexity.item():.2f} / 64")
    print(f"   Codebook usage: {perplexity.item()/64*100:.1f}%")
    
    # 2. RuleLearner输出
    _, D, H, W = symbols.shape
    context_symbols = symbols[:B*8].view(B, 8, D, H, W)
    
    # 聚合上下文（平均池化）
    aggregated_context = context_symbols.mean(dim=1)  # (B, D, H, W)
    
    reasoning_state, step_info = model.rule_learner(aggregated_context)
    
    print(f"\n2. Rule Learner:")
    print(f"   Reasoning state shape: {reasoning_state.shape}")
    if isinstance(step_info, dict):
        if 'num_steps' in step_info:
            print(f"   Number of steps: {step_info['num_steps']}")
        if 'confidences' in step_info:
            confidences = step_info['confidences']
            print(f"   Confidences: {[f'{c.item():.4f}' for c in confidences]}")
        if 'rule_selections' in step_info:
            print(f"   Rule selections available: Yes")
    
    # 3. Decoder输出
    logits = model.decoder(reasoning_state)
    
    print(f"\n3. Decoder:")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"   Logits std: {logits.std().item():.4f}")
    print(f"   Predictions: {logits.argmax(dim=1).tolist()}")
    print(f"   Targets: {target.tolist()}")
    print(f"   Correct: {(logits.argmax(dim=1) == target.cuda()).sum().item()}/{B}")

# 4. 梯度检查
model.train()
context, answers, target, _ = next(iter(loader))
context = context.cuda()
answers = answers.cuda()
target = target.cuda()

logits = model(context, answers)
loss = torch.nn.functional.cross_entropy(logits, target)
loss.backward()

print(f"\n4. Gradient Analysis:")
print(f"   Loss: {loss.item():.6f}")

total_grad_norm = 0
zero_grad_layers = []
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        total_grad_norm += grad_norm
        if grad_norm < 1e-7:
            zero_grad_layers.append(name)

print(f"   Total gradient norm: {total_grad_norm:.6f}")
print(f"   Layers with near-zero gradients: {len(zero_grad_layers)}")
if zero_grad_layers:
    print(f"   Zero-grad layers (first 5):")
    for name in zero_grad_layers[:5]:
        print(f"      - {name}")

# 5. VQ Codebook使用情况
print(f"\n5. VQ Codebook Analysis:")
print(f"   Codebook size: 64")
print(f"   Perplexity: {perplexity.item():.2f}")
print(f"   Effective codes used: ~{int(perplexity.item())}/64")
if perplexity.item() < 10:
    print(f"   ⚠️  WARNING: Low codebook usage! (< 16%)")
elif perplexity.item() < 32:
    print(f"   ⚠️  Moderate codebook usage ({perplexity.item()/64*100:.1f}%)")
else:
    print(f"   ✅ Good codebook usage ({perplexity.item()/64*100:.1f}%)")

print("\n" + "="*70)
print("Debugging Complete!")
print("="*70)