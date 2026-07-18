import os
import matplotlib.pyplot as plt
import numpy as np

# Set styling for academic journal quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11

conditions = ['BM25\n(Baseline)', 'FAISS\nOnly', 'Full\nPipeline', 'Full +\nValidator']
hit1_scores = [93.8, 20.5, 76.2, 63.0]
hit3_scores = [94.1, 71.1, 93.0, 93.0]

x = np.arange(len(conditions))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

# Curated academic color palette (Navy and Slate Blue)
rects1 = ax.bar(x - width/2, hit1_scores, width, label='Hit@1', color='#1A365D', edgecolor='#0A1D37')
rects2 = ax.bar(x + width/2, hit3_scores, width, label='Hit@3', color='#4A90E2', edgecolor='#21589C')

# Labels and titles
ax.set_ylabel('Accuracy Score (%)', fontweight='bold', color='#1A365D')
ax.set_title('Recommendation Accuracy Across Pipeline Conditions (n=273)', fontweight='bold', fontsize=12, pad=15, color='#1A365D')
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontweight='bold')
ax.set_ylim(0, 105)
ax.legend(frameon=True, facecolor='white', edgecolor='#CCCCCC')

# Add text labels on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()

# Save image
os.makedirs('outputs', exist_ok=True)
output_path = 'outputs/evaluation_comparison.png'
plt.savefig(output_path, bbox_inches='tight')
print(f"Chart successfully saved to {output_path}")
