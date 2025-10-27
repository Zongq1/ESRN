"""
训练简化版ESRN（连续特征）
用于快速验证baseline性能
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from esrn.models.esrn_continuous import ESRNContinuous
from esrn.datasets.raven_dataset import RAVENDataset

# 超参数
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据
train_dataset = RAVENDataset('data/RAVEN-10000', 'center_single', 'train')
val_dataset = RAVENDataset('data/RAVEN-10000', 'center_single', 'val')

# Quick test
train_dataset.samples = train_dataset.samples[:1000]
val_dataset.samples = val_dataset.samples[:200]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 模型
model = ESRNContinuous().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# 训练
best_acc = 0
for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for context, answers, target, _ in pbar:
        context = context.to(DEVICE)
        target = target.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(context)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = logits.argmax(dim=1)
        train_correct += (pred == target).sum().item()
        train_total += target.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*train_correct/train_total:.2f}%'})
    
    # Validate
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for context, answers, target, _ in tqdm(val_loader, desc=f'Epoch {epoch} [Val]'):
            context = context.to(DEVICE)
            target = target.to(DEVICE)
            
            logits = model(context)
            loss = criterion(logits, target)
            
            val_loss += loss.item()
            pred = logits.argmax(dim=1)
            val_correct += (pred == target).sum().item()
            val_total += target.size(0)
    
    train_acc = 100. * train_correct / train_total
    val_acc = 100. * val_correct / val_total
    
    print(f"\nEpoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%\n")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'checkpoints/simple_best.pth')
        print(f"✅ New best: {best_acc:.2f}%")

print(f"\n✅ Training complete! Best Val Acc: {best_acc:.2f}%")