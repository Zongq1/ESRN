"""
ESRN Training Script
Train ESRN model on RAVEN dataset
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from esrn.models.esrn_fixed import ESRNFixed
from esrn.datasets.raven_dataset import RAVENDataset


class ESRNTrainer:
    """
    Trainer for ESRN model
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        lr=1e-4,
        save_dir='checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss function (Cross Entropy for 8-way classification)
        self.criterion = nn.CrossEntropyLoss()
        
        # Checkpointing
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.best_val_acc = 0.0
        self.train_history = []
        self.val_history = []
        
        print(f"✅ Trainer initialized")
        print(f"   Device: {device}")
        print(f"   Learning rate: {lr}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, (context, answers, target, meta) in enumerate(pbar):
            # Move to device
            context = context.to(self.device)  # (B, 8, 1, 160, 160)
            answers = answers.to(self.device)  # (B, 8, 1, 160, 160)
            target = target.to(self.device)    # (B,)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(context, answers)  # (B, 8)
            
            # Compute loss
            loss = self.criterion(logits, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for context, answers, target, meta in pbar:
                # Move to device
                context = context.to(self.device)
                answers = answers.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                logits = self.model(context, answers)
                
                # Compute loss
                loss = self.criterion(logits, target)
                
                # Statistics
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs):
        """Train for multiple epochs"""
        print(f"\n{'='*70}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*70}\n")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
            
            # Save history
            self.train_history.append({
                'epoch': epoch,
                'loss': train_loss,
                'accuracy': train_acc
            })
            self.val_history.append({
                'epoch': epoch,
                'loss': val_loss,
                'accuracy': val_acc
            })
            
            # Print epoch summary
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"{'='*70}\n")
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                print(f"✅ New best model! Val Acc: {val_acc:.2f}%")
            
            # Save regular checkpoint every 5 epochs
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print(f"\n{'='*70}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*70}\n")
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint with safety mechanisms"""
        import os
        import tempfile
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        # 安全保存：先保存到临时文件，确认成功后再移动
        if is_best:
            final_path = self.save_dir / 'best_model.pth'
        else:
            final_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        
        # 创建临时文件
        temp_path = self.save_dir / f'temp_checkpoint_{epoch}.pth'
        
        try:
            # 保存到临时文件
            torch.save(checkpoint, temp_path)
            
            # 验证文件大小（应该>1MB）
            file_size = os.path.getsize(temp_path)
            if file_size < 1000:  # 小于1KB表示保存失败
                raise RuntimeError(f"Checkpoint file too small: {file_size} bytes")
            
            # 强制刷新到磁盘
            with open(temp_path, 'rb') as f:
                f.read(1)  # 触发系统缓存刷新
            
            # 重命名为最终文件（原子操作）
            if final_path.exists():
                final_path.unlink()  # 删除旧文件
            temp_path.rename(final_path)
            
            print(f"💾 Safely saved checkpoint to {final_path} ({file_size/1024/1024:.2f} MB)")
            
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise
        
        # Save training history (also with safety)
        history_path = self.save_dir / 'training_history.json'
        temp_history = self.save_dir / 'temp_history.json'
        
        try:
            with open(temp_history, 'w') as f:
                json.dump({
                    'train': self.train_history,
                    'val': self.val_history
                }, f, indent=2)
            
            if history_path.exists():
                history_path.unlink()
            temp_history.rename(history_path)
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save training history: {e}")
            if temp_history.exists():
                temp_history.unlink()


def main():
    parser = argparse.ArgumentParser(description='Train ESRN on RAVEN')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='data/RAVEN-10000',
                        help='Path to RAVEN dataset')
    parser.add_argument('--config', type=str, default='center_single',
                        choices=RAVENDataset.get_all_configs(),
                        help='RAVEN configuration')
    parser.add_argument('--img_size', type=int, default=160,
                        help='Image size')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--num_symbols', type=int, default=64,
                        help='Number of symbols in codebook')
    parser.add_argument('--num_rules', type=int, default=32,
                        help='Number of rules')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test with small data')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"\n{'='*70}")
    print(f"ESRN Training Configuration")
    print(f"{'='*70}")
    print(f"  Dataset: RAVEN-10000 ({args.config})")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Image size: {args.img_size}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num symbols: {args.num_symbols}")
    print(f"  Num rules: {args.num_rules}")
    print(f"{'='*70}\n")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = RAVENDataset(
        data_root=args.data_root,
        config=args.config,
        split='train',
        img_size=args.img_size
    )
    
    val_dataset = RAVENDataset(
        data_root=args.data_root,
        config=args.config,
        split='val',
        img_size=args.img_size
    )
    
    # Quick test mode (use subset)
    if args.quick_test:
        print("⚡ Quick test mode: using 100 train + 50 val samples")
        train_dataset.samples = train_dataset.samples[:100]
        val_dataset.samples = val_dataset.samples[:50]
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"✅ Datasets loaded:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples\n")
    
    # Create model
    print("Creating ESRN model...")
    model = ESRNFixed(
        in_channels=1,
        encoder_hidden_dims=[64, 128, 256],
        vocab_size=args.num_symbols,
        embed_dim=64,
        num_rules=args.num_rules,
        rule_hidden_dim=args.hidden_dim // 4,  # 128
        max_steps=5,
        commitment_cost=0.05
    )
    
    # Create trainer
    trainer = ESRNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        lr=args.lr,
        save_dir=args.save_dir
    )
    
    # Train
    trainer.train(num_epochs=args.epochs)
    
    print("\n✅ Training completed!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == '__main__':
    main()