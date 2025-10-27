"""
完整可视化RAVEN样本（显示所有8个answers）
"""
import sys
sys.path.insert(0, '.')

from esrn.datasets.raven_dataset import RAVENDataset
import matplotlib.pyplot as plt

dataset = RAVENDataset('data/RAVEN-10000', 'center_single', 'train')

# 可视化第一个样本
context, answers, target, meta = dataset[0]

print(f"Target answer: {target}")
print(f"Meta: {meta}")

# 绘制：3行8列（第1-2行context，第3行answers）
fig, axes = plt.subplots(3, 8, figsize=(20, 8))

# 前两行：8个context panels
for i in range(8):
    row = i // 4
    col = i % 4
    if i < 4:
        axes[0, col].imshow(context[i, 0], cmap='gray')
        axes[0, col].set_title(f'Context {i+1}')
        axes[0, col].axis('off')
    else:
        axes[1, col-4+4].imshow(context[i, 0], cmap='gray')
        axes[1, col-4+4].set_title(f'Context {i+1}')
        axes[1, col-4+4].axis('off')

# 隐藏第一行后4个和第二行前4个
for i in range(4):
    axes[0, 4+i].axis('off')
    axes[1, i].axis('off')

# 第三行：8个candidate answers
for i in range(8):
    axes[2, i].imshow(answers[i, 0], cmap='gray')
    title = f'Answer {i}'
    if i == target:
        title += ' ✓'
        axes[2, i].set_title(title, color='green', fontweight='bold', fontsize=14)
    else:
        axes[2, i].set_title(title, fontsize=12)
    axes[2, i].axis('off')

plt.tight_layout()
plt.savefig('raven_complete_visualization.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved to: raven_complete_visualization.png")
print(f"\n正确答案是 Answer {target}")