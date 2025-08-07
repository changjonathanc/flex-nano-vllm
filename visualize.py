# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///

import matplotlib.pyplot as plt
import numpy as np

# Data
configs = ['50% GPU', '90% GPU', '90% GPU\n(high batch)']
vllm_output = [3020, 3772, 3840]
flex_output = [2146, 2899, 3266]

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(configs))
width = 0.35

bars1 = ax.bar(x - width/2, vllm_output, width, label='vLLM v1', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x + width/2, flex_output, width, label='flex-nano-vllm', color='#ff7f0e', alpha=0.8)

ax.set_title('Output Tokens/s Comparison by Configuration', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Tokens/s', fontsize=14)
ax.set_xlabel('GPU Memory Configuration', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=12)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()

# Save the plot
plt.savefig('tokens_per_second_comparison.png', dpi=300, bbox_inches='tight')
print("Simple comparison saved as 'tokens_per_second_comparison.png'")

plt.show()