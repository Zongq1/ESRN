"""
RAVEN Dataset Loader
Loads RAVEN (Relational and Analogical Visual rEasoNing) dataset
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from pathlib import Path


class RAVENDataset(Dataset):
    """
    RAVEN Dataset for Abstract Visual Reasoning
    
    Dataset structure:
        data/RAVEN-10000/
            ├── center_single/
            │   ├── train/
            │   ├── val/
            │   └── test/
            ├── distribute_four/
            ├── ... (7 configurations)
            
    Each sample contains:
        - 8 context panels (3x3 grid minus bottom-right)
        - 8 candidate answer panels
        - 1 correct answer index (0-7)
    """
    
    def __init__(
        self,
        data_root='data/RAVEN-10000',
        config='center_single',
        split='train',
        transform=None,
        img_size=160
    ):
        """
        Args:
            data_root: path to RAVEN dataset
            config: configuration name (e.g., 'center_single')
            split: 'train', 'val', or 'test'
            transform: optional image transformations
            img_size: resize images to this size
        """
        self.data_root = Path(data_root)
        self.config = config
        self.split = split
        self.transform = transform
        self.img_size = img_size
        
        # Path to this configuration
        self.config_path = self.data_root / config / split
        
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Dataset path not found: {self.config_path}\n"
                f"Please download RAVEN dataset to {self.data_root}"
            )
        
        # Get all NPZ files
        self.samples = sorted(list(self.config_path.glob('*.npz')))
        
        print(f"Loaded {len(self.samples)} samples from {config}/{split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            context: (8, 1, H, W) - 8 context panels
            answers: (8, 1, H, W) - 8 candidate answers
            target: int - correct answer index (0-7)
            meta: dict - metadata
        """
        # Load NPZ file
        sample_path = self.samples[idx]
        data = np.load(sample_path)
        
        # RAVEN format: (16, 160, 160) - 8 context + 8 answers
        image = data['image']  # (16, 160, 160)
        target = data['target']  # scalar (0-7)
        
        # Split context and answers
        context = image[:8]   # (8, 160, 160)
        answers = image[8:]   # (8, 160, 160)
        
        # Resize if needed
        if self.img_size != 160:
            context = self._resize_panels(context)
            answers = self._resize_panels(answers)
        
        # Convert to tensors and normalize
        context = torch.from_numpy(context).float().unsqueeze(1) / 255.0  # (8, 1, H, W)
        answers = torch.from_numpy(answers).float().unsqueeze(1) / 255.0  # (8, 1, H, W)
        target = torch.tensor(target, dtype=torch.long)
        
        # Apply transforms if provided
        if self.transform:
            context = torch.stack([self.transform(img) for img in context])
            answers = torch.stack([self.transform(img) for img in answers])
        
        meta = {
            'sample_id': sample_path.stem,
            'config': self.config,
            'split': self.split
        }
        
        return context, answers, target, meta
    
    def _resize_panels(self, panels):
        """Resize multiple panels"""
        from torchvision.transforms.functional import resize
        resized = []
        for panel in panels:
            img = Image.fromarray(panel.astype(np.uint8))
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            resized.append(np.array(img))
        return np.stack(resized)
    
    @staticmethod
    def get_all_configs():
        """Return all 7 RAVEN configuration names"""
        return [
            'center_single',
            'distribute_four',
            'distribute_nine',
            'left_center_single_right_center_single',
            'up_center_single_down_center_single',
            'in_center_single_out_center_single',
            'in_distribute_four_out_center_single'
        ]


class RAVENDataModule:
    """
    Data module for easy training setup
    """
    def __init__(
        self,
        data_root='data/RAVEN-10000',
        config='center_single',
        batch_size=32,
        num_workers=4,
        img_size=160
    ):
        self.data_root = data_root
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        
        # Create datasets
        self.train_dataset = RAVENDataset(
            data_root, config, 'train', img_size=img_size
        )
        self.val_dataset = RAVENDataset(
            data_root, config, 'val', img_size=img_size
        )
        self.test_dataset = RAVENDataset(
            data_root, config, 'test', img_size=img_size
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def download_raven_instructions():
    """
    Print instructions for downloading RAVEN dataset
    """
    print("=" * 70)
    print("RAVEN Dataset Download Instructions")
    print("=" * 70)
    print("\n📥 Option 1: Official GitHub (Recommended)")
    print("   Repository: https://github.com/WellyZhang/RAVEN")
    print("   1. Visit the repository")
    print("   2. Download RAVEN-10000.zip")
    print("   3. Extract to: E:/Design/ESRN/data/RAVEN-10000")
    print("\n📥 Option 2: Google Drive")
    print("   Link: Check the GitHub README for Google Drive link")
    print("\n📂 Expected structure:")
    print("   data/RAVEN-10000/")
    print("       ├── center_single/")
    print("       ├── distribute_four/")
    print("       ├── distribute_nine/")
    print("       └── ... (7 configs total)")
    print("\n💾 Dataset size: ~2.5 GB")
    print("=" * 70)


if __name__ == "__main__":
    print("Testing RAVEN Dataset Loader...\n")
    
    # Check if dataset exists
    data_root = Path('data/RAVEN-10000')
    
    if not data_root.exists():
        print("⚠️  RAVEN dataset not found!")
        download_raven_instructions()
        print("\n" + "=" * 70)
        print("Testing with mock data structure check...")
        print("=" * 70)
    else:
        print("✅ Dataset directory found!")
        print(f"   Path: {data_root.absolute()}\n")
        
        # Try to load a dataset
        try:
            print("=" * 70)
            print("Testing RAVENDataset...")
            print("=" * 70)
            
            dataset = RAVENDataset(
                data_root=str(data_root),
                config='center_single',
                split='train'
            )
            
            print(f"\n✅ Dataset loaded successfully!")
            print(f"   - Total samples: {len(dataset)}")
            
            # Test loading one sample
            print("\nLoading sample #0...")
            context, answers, target, meta = dataset[0]
            
            print(f"\n📊 Sample shapes:")
            print(f"   - Context panels: {context.shape}")
            print(f"   - Answer panels: {answers.shape}")
            print(f"   - Target: {target.item()}")
            print(f"   - Metadata: {meta}")
            
            # Test DataLoader
            print("\n" + "=" * 70)
            print("Testing DataLoader...")
            print("=" * 70)
            
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            batch = next(iter(dataloader))
            context_batch, answers_batch, target_batch, meta_batch = batch
            
            print(f"\n✅ DataLoader working!")
            print(f"   - Batch context: {context_batch.shape}")
            print(f"   - Batch answers: {answers_batch.shape}")
            print(f"   - Batch targets: {target_batch.shape}")
            
            # Test all configs
            print("\n" + "=" * 70)
            print("Available RAVEN configurations:")
            print("=" * 70)
            for i, config in enumerate(RAVENDataset.get_all_configs(), 1):
                config_path = data_root / config / 'train'
                exists = "✅" if config_path.exists() else "❌"
                print(f"{i}. {exists} {config}")
            
            print("\n" + "=" * 70)
            print("✅ All tests passed!")
            print("=" * 70)
            
        except Exception as e:
            print(f"\n❌ Error loading dataset: {e}")
            print("\nPlease make sure:")
            print("1. RAVEN dataset is downloaded")
            print("2. Extracted to correct location")
            print("3. Directory structure is correct")

