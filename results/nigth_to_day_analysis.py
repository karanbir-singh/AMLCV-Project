"""
Analysis script for Night-to-Day Evaluation (img2turbo)
Run this in the same folder as 'night_to_day_results.csv'
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
INPUT_FILE = 'night_to_day_results.csv'
OUTPUT_DIR = 'n2d_analysis_plots'

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load results
print(f"Loading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
Path(OUTPUT_DIR).mkdir(exist_ok=True)


# ============================================================
# 1. QUALITY CLASSIFICATION LOGIC (NIGHT-TO-DAY)
# ============================================================
def classify_quality_n2d(row):
    """
    Classifies Night-to-Day quality.
    Prioritizes: Information Recovery (Visibility) and Exposure Control.
    """
    score = 0

    # 1. Semantic Retention (Baseline: 0-30 pts)
    # Did we keep what was already there?
    if row['object_consistency'] >= 0.9:
        score += 30
    elif row['object_consistency'] >= 0.7:
        score += 20
    elif row['object_consistency'] >= 0.5:
        score += 10

    # 2. Information Recovery (The Goal: 0-30 pts)
    # Did we find MORE objects? (>1.0 is good)
    if row['object_visibility_improvement'] >= 1.5:
        score += 30  # Huge recovery
    elif row['object_visibility_improvement'] >= 1.2:
        score += 25
    elif row['object_visibility_improvement'] >= 1.0:
        score += 15

    # 3. Brightness/Exposure (0-20 pts)
    # Is it bright enough without being washed out?
    if 40 <= row['brightness_increase'] <= 90:
        score += 20
    elif 20 <= row['brightness_increase'] <= 110:
        score += 10

    # 4. Exposure Quality (0-20 pts)
    # Did we avoid bleaching everything white?
    if row['overexposure_score'] >= 0.8:
        score += 20
    elif row['overexposure_score'] >= 0.6:
        score += 10

    if score >= 80:
        return 'Excellent'
    elif score >= 60:
        return 'Good'
    elif score >= 40:
        return 'Fair'
    else:
        return 'Poor'


df['quality_class'] = df.apply(classify_quality_n2d, axis=1)
quality_counts = df['quality_class'].value_counts()

# ============================================================
# 2. GENERATE PLOTS
# ============================================================

# --- Plot 1: Metric Distributions ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Key Night-to-Day Metrics', fontsize=16, fontweight='bold')
#     numeric_cols = ['object_consistency', 'object_visibility_improvement', 'ssim',
#                     'brightness_increase', 'warm_shift', 'saturation_increase',
#                     'edge_preservation', 'shadow_removal', 'overexposure_score', 'lpips']
metrics_to_plot = [
    ('object_visibility_improvement', 'Visibility Improvement (Factor)', (0, 3)),
    ('brightness_increase', 'Brightness Increase', (0, 100)),
    ('overexposure_score', 'Overexposure Score (Higher is Better)', (0, 1)),
    ('saturation_increase', 'Saturation Increase (Neg = Desaturated)', (-150, 50))
]

for idx, (metric, title, xlim) in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    sns.histplot(df[metric], bins=30, kde=True, ax=ax, color='steelblue', edgecolor='black')

    mean_val = df[metric].mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')

    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/n2d_metric_distributions.png', dpi=300)
print("✓ Saved metric distributions")

# --- Plot 2: Quality Distribution Pie Chart ---
fig, ax = plt.subplots(figsize=(8, 8))
colors = {'Excellent': '#2ecc71', 'Good': '#3498db', 'Fair': '#f39c12', 'Poor': '#e74c3c'}
available_colors = [colors.get(q, '#999999') for q in quality_counts.index]

quality_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=available_colors, startangle=90)
ax.set_ylabel('')
ax.set_title('Overall Quality Distribution (Night-to-Day)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/n2d_quality_distribution.png', dpi=300)
print("✓ Saved quality distribution")

# --- Plot 3: Issue Frequency ---
issue_counts = {}
for issues in df['issues']:
    if issues != 'OK':
        for issue in issues.split('; '):
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

if issue_counts:
    fig, ax = plt.subplots(figsize=(12, 6))
    issues_df = pd.DataFrame(list(issue_counts.items()), columns=['Issue', 'Count']).sort_values('Count',
                                                                                                 ascending=True)

    ax.barh(issues_df['Issue'], issues_df['Count'], color='coral', edgecolor='black')
    ax.set_title('Frequency of Detected Issues', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Images')

    # Add percentage labels
    for idx, row in issues_df.iterrows():
        pct = (row['Count'] / len(df)) * 100
        ax.text(row['Count'], idx, f" {pct:.1f}%", va='center')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/n2d_issue_frequency.png', dpi=300)
    print("✓ Saved issue frequency")

print("\nAnalysis Complete!")