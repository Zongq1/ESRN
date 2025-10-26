import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Symbol Reconstruction Experiment
Test end-to-end reconstruction: Image → Encoder → Decoder → Reconstructed Image

Week 3-4 Task #3: 符号重建实验
- 测试VQ-VAE编码器质量
- 测试Decoder重建能力
- 可视化原图 vs 重建图
- 分析重建误差和符号质量
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from esrn.models.encoder import SymbolicEncoder
from esrn.models.decoder import SymbolicDecoder
from esrn.datasets.raven_dataset import RAVENDataset


def visualize_reconstruction(
    original_images,
    reconstructed_images,
    save_path=None,
    num_samples=4
):
    """
    Visualize original vs reconstructed images
    
    Args:
        original_images: (B, C, H, W) original images
        reconstructed_images: (B, C, H', W') reconstructed images
        save_path: path to save figure
        num_samples: number of samples to display
    """
    num_samples = min(num_samples, original_images.size(0))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        orig = original_images[i, 0].cpu().numpy()
        axes[i, 0].imshow(orig, cmap='gray', vmin=-1, vmax=1)
        axes[i, 0].set_title(f'Sample {i+1}: Original\n{orig.shape}')
        axes[i, 0].axis('off')
        
        # Reconstructed image
        recon = reconstructed_images[i, 0].cpu().numpy()
        axes[i, 1].imshow(recon, cmap='gray', vmin=-1, vmax=1)
        axes[i, 1].set_title(f'Reconstructed\n{recon.shape}')
        axes[i, 1].axis('off')
        
        # Difference heatmap
        # Resize if needed
        if orig.shape != recon.shape:
            # Interpolate reconstruction to match original size
            recon_resized = torch.nn.functional.interpolate(
                reconstructed_images[i:i+1],
                size=orig.shape,
                mode='bilinear',
                align_corners=False
            )[0, 0].cpu().numpy()
        else:
            recon_resized = recon
        
        diff = np.abs(orig - recon_resized)
        mse = np.mean(diff ** 2)
        
        im = axes[i, 2].imshow(diff, cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Absolute Difference\nMSE: {mse:.4f}')
        axes[i, 2].axis('off')
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization to {save_path}")
    
    plt.show()


def test_reconstruction(
    encoder,
    decoder,
    dataset,
    device='cuda',
    num_samples=8,
    save_dir='results/reconstruction'
):
    """
    Test end-to-end reconstruction
    
    Args:
        encoder: SymbolicEncoder model
        decoder: SymbolicDecoder model
        dataset: RAVENDataset
        device: 'cuda' or 'cpu'
        num_samples: number of samples to test
        save_dir: directory to save results
    """
    encoder.eval()
    decoder.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    all_originals = []
    all_reconstructed = []
    all_vq_losses = []
    all_perplexities = []
    all_mse = []
    
    print(f"\n{'='*70}")
    print(f"Testing End-to-End Reconstruction")
    print(f"{'='*70}")
    print(f"  Encoder: SymbolicEncoder")
    print(f"  Decoder: SymbolicDecoder (reconstruction head)")
    print(f"  Device: {device}")
    print(f"  Num samples: {num_samples}")
    print(f"{'='*70}\n")
    
    with torch.no_grad():
        for idx in indices:
            context, answers, target, meta = dataset[idx]
            
            # Use first context panel
            image = context[0:1].to(device)  # (1, 1, H, W)
            
            # Encode
            symbols, vq_loss, perplexity = encoder(image)
            
            # Reconstruct using decoder
            _, reconstructed = decoder(symbols, return_recon=True)
            
            # Calculate MSE (resize if needed)
            if image.shape != reconstructed.shape:
                recon_resized = torch.nn.functional.interpolate(
                    reconstructed,
                    size=image.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                recon_resized = reconstructed
            
            mse = torch.mean((image - recon_resized) ** 2).item()
            
            all_originals.append(image)
            all_reconstructed.append(reconstructed)
            all_vq_losses.append(vq_loss.item())
            all_perplexities.append(perplexity.item())
            all_mse.append(mse)
            
            print(f"Sample {len(all_mse)}:")
            print(f"  Input: {image.shape} | Symbols: {symbols.shape} | Recon: {reconstructed.shape}")
            print(f"  VQ loss: {vq_loss.item():.6f} | Perplexity: {perplexity.item():.2f} | MSE: {mse:.6f}")
    
    # Stack samples
    all_originals = torch.cat(all_originals, dim=0)
    all_reconstructed = torch.cat(all_reconstructed, dim=0)
    
    # Statistics
    avg_vq_loss = np.mean(all_vq_losses)
    avg_perplexity = np.mean(all_perplexities)
    avg_mse = np.mean(all_mse)
    std_mse = np.std(all_mse)
    
    print(f"\n{'='*70}")
    print("Reconstruction Statistics:")
    print(f"{'='*70}")
    print(f"  Average VQ loss: {avg_vq_loss:.6f}")
    print(f"  Average Perplexity: {avg_perplexity:.2f}")
    print(f"  Average MSE: {avg_mse:.6f} ± {std_mse:.6f}")
    print(f"  Min MSE: {np.min(all_mse):.6f}")
    print(f"  Max MSE: {np.max(all_mse):.6f}")
    print(f"{'='*70}\n")
    
    # Visualize
    save_path = save_dir / 'reconstruction_comparison.png'
    visualize_reconstruction(
        all_originals,
        all_reconstructed,
        save_path=save_path,
        num_samples=min(4, num_samples)
    )
    
    # Plot statistics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # MSE distribution
    axes[0].plot(all_mse, 'o-', linewidth=2, markersize=8, color='blue')
    axes[0].axhline(avg_mse, color='r', linestyle='--', label=f'Mean: {avg_mse:.6f}')
    axes[0].set_xlabel('Sample Index', fontsize=12)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('Reconstruction MSE', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Perplexity
    axes[1].plot(all_perplexities, 'o-', linewidth=2, markersize=8, color='green')
    axes[1].axhline(avg_perplexity, color='r', linestyle='--', label=f'Mean: {avg_perplexity:.2f}')
    axes[1].set_xlabel('Sample Index', fontsize=12)
    axes[1].set_ylabel('Perplexity', fontsize=12)
    axes[1].set_title('VQ Codebook Perplexity', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # VQ Loss
    axes[2].plot(all_vq_losses, 'o-', linewidth=2, markersize=8, color='orange')
    axes[2].axhline(avg_vq_loss, color='r', linestyle='--', label=f'Mean: {avg_vq_loss:.6f}')
    axes[2].set_xlabel('Sample Index', fontsize=12)
    axes[2].set_ylabel('VQ Loss', fontsize=12)
    axes[2].set_title('Vector Quantization Loss', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    stats_path = save_dir / 'reconstruction_statistics.png'
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved statistics to {stats_path}")
    plt.show()
    
    return avg_mse, avg_vq_loss, avg_perplexity


def main():
    parser = argparse.ArgumentParser(description='Symbol Reconstruction Experiment')
    
    parser.add_argument('--data_root', type=str, default='data/RAVEN-10000',
                        help='Path to RAVEN dataset')
    parser.add_argument('--config', type=str, default='center_single',
                        help='RAVEN configuration')
    parser.add_argument('--num_samples', type=int, default=8,
                        help='Number of samples to test')
    parser.add_argument('--vocab_size', type=int, default=512,
                        help='Codebook size')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--decoder_hidden', type=int, default=256,
                        help='Decoder hidden dimension')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='results/reconstruction',
                        help='Results directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"\n{'='*70}")
    print("Symbol Reconstruction Experiment")
    print(f"{'='*70}")
    print(f"  Date: 2025-10-26")
    print(f"  Week 3-4 Task #3: 符号重建实验")
    print(f"  Dataset: {args.config}")
    print(f"  Codebook size: {args.vocab_size}")
    print(f"  Embedding dim: {args.embed_dim}")
    print(f"  Device: {args.device}")
    print(f"  Random seed: {args.seed}")
    print(f"{'='*70}\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = RAVENDataset(
        data_root=args.data_root,
        config=args.config,
        split='train',
        img_size=160
    )
    print(f"✅ Loaded {len(dataset)} samples\n")
    
    # Create encoder
    print("Creating encoder...")
    encoder = SymbolicEncoder(
        in_channels=1,
        hidden_dims=[64, 128, 256],
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        commitment_cost=0.25
    ).to(args.device)
    
    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"✅ Encoder created: {encoder_params:,} parameters\n")
    
    # Create decoder
    print("Creating decoder...")
    decoder = SymbolicDecoder(
        embed_dim=args.embed_dim,
        hidden_dim=args.decoder_hidden,
        num_classes=8
    ).to(args.device)
    
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"✅ Decoder created: {decoder_params:,} parameters\n")
    
    # Test reconstruction
    avg_mse, avg_vq_loss, avg_perplexity = test_reconstruction(
        encoder=encoder,
        decoder=decoder,
        dataset=dataset,
        device=args.device,
        num_samples=args.num_samples,
        save_dir=args.save_dir
    )
    
    # Final summary
    print(f"\n{'='*70}")
    print("✅ Experiment Completed!")
    print(f"{'='*70}")
    print(f"  Average MSE: {avg_mse:.6f}")
    print(f"  Average VQ loss: {avg_vq_loss:.6f}")
    print(f"  Average perplexity: {avg_perplexity:.2f}")
    print(f"  Codebook usage: ~{avg_perplexity/args.vocab_size*100:.2f}%")
    print(f"  Results saved to: {args.save_dir}")
    print(f"{'='*70}\n")
    
    # Quality assessment
    print("Quality Assessment:")
    print(f"{'='*70}")
    
    # MSE assessment
    if avg_mse < 0.01:
        print("  ✅ Reconstruction: Excellent (MSE < 0.01)")
    elif avg_mse < 0.05:
        print("  ✅ Reconstruction: Good (MSE < 0.05)")
    elif avg_mse < 0.1:
        print("  ⚠️  Reconstruction: Moderate (MSE < 0.1)")
    else:
        print("  ❌ Reconstruction: Poor (MSE >= 0.1)")
    
    # VQ loss assessment
    if avg_vq_loss < 0.5:
        print("  ✅ Encoding: Excellent (VQ loss < 0.5)")
    elif avg_vq_loss < 1.0:
        print("  ✅ Encoding: Good (VQ loss < 1.0)")
    else:
        print("  ⚠️  Encoding: Needs improvement")
    
    # Codebook usage
    usage_rate = avg_perplexity / args.vocab_size * 100
    if usage_rate > 80:
        print("  ⚠️  Codebook: High usage (may need larger vocabulary)")
    elif usage_rate > 30:
        print("  ✅ Codebook: Moderate usage (good)")
    else:
        print("  ⚠️  Codebook: Low usage (may be overfitting or underfitting)")
    
    print(f"{'='*70}\n")
    
    print("📊 Week 3-4 Task #3 Status: ✅ COMPLETE")
    print("Next: SSE Technical Report (Task #4)")


if __name__ == '__main__':
    main()