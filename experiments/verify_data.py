"""
验证RAVEN数据集的标签是否正确
"""
import sys
sys.path.insert(0, '.')

from esrn.datasets.raven_dataset import RAVENDataset
import numpy as np

dataset = RAVENDataset('data/RAVEN-10000', 'center_single', 'train')

print("="*70)
print("RAVEN Dataset Verification")
print("="*70)

# 1. 检查标签分布
targets = [dataset[i][2] for i in range(min(1000, len(dataset)))]
print(f"\n1. Label Distribution (first 1000 samples):")
for i in range(8):
    count = targets.count(i)
    print(f"   Class {i}: {count:4d} ({count/10:.1f}%)")

# 2. 检查是否所有标签都在0-7范围内
if all(0 <= t <= 7 for t in targets):
    print("\n✅ All labels in valid range [0-7]")
else:
    print("\n❌ ERROR: Some labels out of range!")
    
# 3. 检查具体样本
print(f"\n2. Sample Inspection:")
for i in range(5):
    context, answers, target, meta = dataset[i]
    print(f"\n   Sample {i}:")
    print(f"      Context shape: {context.shape}")
    print(f"      Answers shape: {answers.shape}")
    print(f"      Target: {target}")
    print(f"      Meta: {meta}")
    
    # 检查数据范围
    print(f"      Context value range: [{context.min():.3f}, {context.max():.3f}]")
    print(f"      Answers value range: [{answers.min():.3f}, {answers.max():.3f}]")

# 4. 检查数据文件结构
import os
data_path = 'data/RAVEN-10000/center_single'
if os.path.exists(data_path):
    print(f"\n3. Data Directory Structure:")
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_path, split)
        if os.path.exists(split_path):
            npz_files = [f for f in os.listdir(split_path) if f.endswith('.npz')]
            print(f"   {split}: {len(npz_files)} files")
            
            if npz_files:
                # 检查一个文件的内容
                sample_file = os.path.join(split_path, npz_files[0])
                data = np.load(sample_file)
                print(f"      Sample file keys: {list(data.keys())}")
                if 'target' in data:
                    print(f"      Target value: {data['target']}")
else:
    print(f"\n❌ Data directory not found: {data_path}")

print("\n" + "="*70)