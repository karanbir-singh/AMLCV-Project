"""
Comprehensive analysis script for img2turbo evaluation results
This generates visualizations and statistical analysis for your report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load results
print("Loading results...")
df = pd.read_csv('evaluation_results.csv')
summary = pd.read_csv('summary_stats.csv')

print(f"Total images evaluated: {len(df)}")
print(f"Columns: {df.columns.tolist()}\n")

# ============================================================
# 1. OVERALL STATISTICS
# ============================================================
print("=" * 60)
print("OVERALL STATISTICS")
print("=" * 60)

metrics = ['object_consistency', 'ssim', 'brightness_reduction',
           'blue_shift', 'saturation_reduction', 'edge_preservation', 'lpips']

for metric in metrics:
    mean = df[metric].mean()
    std = df[metric].std()
    median = df[metric].median()
    q25 = df[metric].quantile(0.25)
    q75 = df[metric].quantile(0.75)

    print(f"\n{metric.upper()}")
    print(f"  Mean:   {mean:.3f} ± {std:.3f}")
    print(f"  Median: {median:.3f}")
    print(f"  Q1-Q3:  {q25:.3f} - {q75:.3f}")
    print(f"  Range:  {df[metric].min():.3f} - {df[metric].max():.3f}")

# ============================================================
# 2. QUALITY CLASSIFICATION
# ============================================================
print("\n" + "=" * 60)
print("QUALITY CLASSIFICATION")
print("=" * 60)


def classify_quality(row):
    """Classify overall quality based on multiple metrics"""
    score = 0

    # Object consistency (0-30 points)
    if row['object_consistency'] >= 0.7:
        score += 30
    elif row['object_consistency'] >= 0.5:
        score += 20
    elif row['object_consistency'] >= 0.3:
        score += 10

    # SSIM structure preservation (0-25 points)
    if row['ssim'] >= 0.7:
        score += 25
    elif row['ssim'] >= 0.5:
        score += 15
    elif row['ssim'] >= 0.3:
        score += 5

    # Brightness reduction (0-20 points)
    if 30 <= row['brightness_reduction'] <= 80:
        score += 20
    elif 20 <= row['brightness_reduction'] <= 100:
        score += 10

    # Edge preservation (0-15 points)
    if row['edge_preservation'] >= 0.5:
        score += 15
    elif row['edge_preservation'] >= 0.3:
        score += 8
    elif row['edge_preservation'] >= 0.1:
        score += 3

    # LPIPS transformation quality (0-10 points)
    if 0.3 <= row['lpips'] <= 0.8:
        score += 10
    elif 0.2 <= row['lpips'] <= 0.9:
        score += 5

    # Classify
    if score >= 80:
        return 'Excellent'
    elif score >= 60:
        return 'Good'
    elif score >= 40:
        return 'Fair'
    else:
        return 'Poor'


df['quality_class'] = df.apply(classify_quality, axis=1)

quality_counts = df['quality_class'].value_counts()
print("\nQuality Distribution:")
for quality, count in quality_counts.items():
    pct = (count / len(df)) * 100
    print(f"  {quality:12s}: {count:4d} ({pct:5.1f}%)")

# ============================================================
# 3. ISSUE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("ISSUE ANALYSIS")
print("=" * 60)

# Count each issue type
issue_counts = {}
for issues in df['issues']:
    if issues != 'OK':
        for issue in issues.split('; '):
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

print(
    f"\nImages without issues: {len(df[df['issues'] == 'OK'])} ({len(df[df['issues'] == 'OK']) / len(df) * 100:.1f}%)")
print(f"Images with issues:    {len(df[df['issues'] != 'OK'])} ({len(df[df['issues'] != 'OK']) / len(df) * 100:.1f}%)")

print("\nMost Common Issues:")
sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
for issue, count in sorted_issues:
    pct = (count / len(df)) * 100
    print(f"  {issue:25s}: {count:4d} ({pct:5.1f}%)")

# ============================================================
# 4. CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("METRIC CORRELATIONS")
print("=" * 60)

corr_metrics = ['object_consistency', 'ssim', 'edge_preservation', 'lpips']
corr_matrix = df[corr_metrics].corr()

print("\nKey Correlations:")
print(f"SSIM ↔ Edge Preservation: {corr_matrix.loc['ssim', 'edge_preservation']:.3f}")
print(f"SSIM ↔ Object Consistency: {corr_matrix.loc['ssim', 'object_consistency']:.3f}")
print(f"LPIPS ↔ SSIM: {corr_matrix.loc['lpips', 'ssim']:.3f}")

# ============================================================
# 5. BEST AND WORST PERFORMERS
# ============================================================
print("\n" + "=" * 60)
print("BEST AND WORST PERFORMERS")
print("=" * 60)

# Create composite score
df['composite_score'] = (
        df['object_consistency'] * 0.3 +
        df['ssim'] * 0.25 +
        df['edge_preservation'] * 0.25 +
        (1 - abs(df['brightness_reduction'] - 55) / 100) * 0.2  # Optimal ~55
)

print("\nTop 5 Best Transformations:")
best = df.nlargest(5, 'composite_score')[['image_name', 'object_consistency',
                                          'ssim', 'edge_preservation', 'issues']]
for idx, row in best.iterrows():
    print(f"\n  {row['image_name']}")
    print(f"    Obj: {row['object_consistency']:.3f}, SSIM: {row['ssim']:.3f}, "
          f"Edge: {row['edge_preservation']:.3f}")
    print(f"    Issues: {row['issues']}")

print("\n\nBottom 5 Worst Transformations:")
worst = df.nsmallest(5, 'composite_score')[['image_name', 'object_consistency',
                                            'ssim', 'edge_preservation', 'issues']]
for idx, row in worst.iterrows():
    print(f"\n  {row['image_name']}")
    print(f"    Obj: {row['object_consistency']:.3f}, SSIM: {row['ssim']:.3f}, "
          f"Edge: {row['edge_preservation']:.3f}")
    print(f"    Issues: {row['issues']}")

# ============================================================
# 6. VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Create output directory
Path('analysis_plots').mkdir(exist_ok=True)

# Plot 1: Distribution of key metrics
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of Key Evaluation Metrics', fontsize=16, fontweight='bold')

metrics_to_plot = [
    ('object_consistency', 'Object Consistency', (0, 1)),
    ('ssim', 'SSIM (Structure)', (0, 1)),
    ('brightness_reduction', 'Brightness Reduction', (0, 100)),
    ('edge_preservation', 'Edge Preservation', (0, 1)),
    ('lpips', 'LPIPS (Perceptual Distance)', (0, 1)),
    ('saturation_reduction', 'Saturation Reduction', (-150, 50))
]

for idx, (metric, title, xlim) in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    ax.hist(df[metric], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(df[metric].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[metric].mean():.2f}')
    ax.axvline(df[metric].median(), color='green', linestyle='--', linewidth=2,
               label=f'Median: {df[metric].median():.2f}')
    ax.set_xlabel(title)
    ax.set_ylabel('Frequency')
    ax.set_xlim(xlim)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis_plots/metric_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis_plots/metric_distributions.png")
plt.close()

# Plot 2: Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Metric Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis_plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis_plots/correlation_heatmap.png")
plt.close()

# Plot 3: Quality classification pie chart
fig, ax = plt.subplots(figsize=(8, 8))
colors = {'Excellent': '#2ecc71', 'Good': '#3498db', 'Fair': '#f39c12', 'Poor': '#e74c3c'}
quality_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%',
                    colors=[colors[q] for q in quality_counts.index],
                    startangle=90)
ax.set_ylabel('')
ax.set_title('Overall Quality Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis_plots/quality_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis_plots/quality_distribution.png")
plt.close()

# Plot 4: Issue frequency bar chart
if issue_counts:
    fig, ax = plt.subplots(figsize=(12, 6))
    issues_df = pd.DataFrame(list(issue_counts.items()), columns=['Issue', 'Count'])
    issues_df = issues_df.sort_values('Count', ascending=True)

    ax.barh(issues_df['Issue'], issues_df['Count'], color='coral', edgecolor='black')
    ax.set_xlabel('Number of Images', fontsize=12)
    ax.set_title('Frequency of Detected Issues', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add percentage labels
    for idx, row in issues_df.iterrows():
        pct = (row['Count'] / len(df)) * 100
        ax.text(row['Count'], idx, f" {pct:.1f}%", va='center')

    plt.tight_layout()
    plt.savefig('analysis_plots/issue_frequency.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: analysis_plots/issue_frequency.png")
    plt.close()

# Plot 5: Scatter plots for key relationships
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Key Metric Relationships', fontsize=16, fontweight='bold')

# SSIM vs Object Consistency
axes[0].scatter(df['ssim'], df['object_consistency'], alpha=0.5, s=30)
axes[0].set_xlabel('SSIM (Structure Preservation)')
axes[0].set_ylabel('Object Consistency')
axes[0].set_title(f'Correlation: {corr_matrix.loc["ssim", "object_consistency"]:.3f}')
axes[0].grid(True, alpha=0.3)

# SSIM vs Edge Preservation
axes[1].scatter(df['ssim'], df['edge_preservation'], alpha=0.5, s=30, color='green')
axes[1].set_xlabel('SSIM (Structure Preservation)')
axes[1].set_ylabel('Edge Preservation')
axes[1].set_title(f'Correlation: {corr_matrix.loc["ssim", "edge_preservation"]:.3f}')
axes[1].grid(True, alpha=0.3)

# LPIPS vs SSIM
axes[2].scatter(df['lpips'], df['ssim'], alpha=0.5, s=30, color='red')
axes[2].set_xlabel('LPIPS (Perceptual Distance)')
axes[2].set_ylabel('SSIM (Structure Preservation)')
axes[2].set_title(f'Correlation: {corr_matrix.loc["lpips", "ssim"]:.3f}')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis_plots/metric_relationships.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis_plots/metric_relationships.png")
plt.close()

# Plot 6: Box plots comparing metrics across quality classes
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Metrics by Quality Class', fontsize=16, fontweight='bold')

key_metrics = ['object_consistency', 'ssim', 'edge_preservation', 'lpips']
for idx, metric in enumerate(key_metrics):
    ax = axes[idx // 2, idx % 2]
    df.boxplot(column=metric, by='quality_class', ax=ax)
    ax.set_title(metric.replace('_', ' ').title())
    ax.set_xlabel('Quality Class')
    ax.set_ylabel(metric.replace('_', ' ').title())
    plt.sca(ax)
    plt.xticks(rotation=45)

plt.suptitle('Metrics by Quality Class', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis_plots/quality_class_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis_plots/quality_class_comparison.png")
plt.close()

# ============================================================
# 7. EXPORT SUMMARY FOR REPORT
# ============================================================
print("\n" + "=" * 60)
print("EXPORTING REPORT SUMMARY")
print("=" * 60)

with open('analysis_plots/report_summary.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("IMG2TURBO NIGHT-TO-DAY TRANSFORMATION EVALUATION\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Total Images Evaluated: {len(df)}\n\n")

    f.write("QUALITY DISTRIBUTION:\n")
    for quality, count in quality_counts.items():
        pct = (count / len(df)) * 100
        f.write(f"  {quality:12s}: {count:4d} ({pct:5.1f}%)\n")

    f.write("\n" + "=" * 60 + "\n")
    f.write("KEY FINDINGS:\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"1. Object Consistency: {df['object_consistency'].mean():.3f} ± {df['object_consistency'].std():.3f}\n")
    f.write(
        f"   - {len(df[df['object_consistency'] >= 0.7])} images ({len(df[df['object_consistency'] >= 0.7]) / len(df) * 100:.1f}%) achieved excellent consistency (≥0.7)\n")
    f.write(
        f"   - {len(df[df['object_consistency'] < 0.5])} images ({len(df[df['object_consistency'] < 0.5]) / len(df) * 100:.1f}%) lost significant objects (<0.5)\n\n")

    f.write(f"2. Structural Similarity (SSIM): {df['ssim'].mean():.3f} ± {df['ssim'].std():.3f}\n")
    f.write(
        f"   - Average preservation indicates {'good' if df['ssim'].mean() >= 0.6 else 'moderate'} structural consistency\n")
    f.write(
        f"   - {len(df[df['ssim'] < 0.5])} images ({len(df[df['ssim'] < 0.5]) / len(df) * 100:.1f}%) show significant structural changes\n\n")

    f.write(
        f"3. Brightness Reduction: {df['brightness_reduction'].mean():.1f} ± {df['brightness_reduction'].std():.1f}\n")
    f.write(
        f"   - Target range (30-80): {len(df[(df['brightness_reduction'] >= 30) & (df['brightness_reduction'] <= 80)])} images ({len(df[(df['brightness_reduction'] >= 30) & (df['brightness_reduction'] <= 80)]) / len(df) * 100:.1f}%)\n\n")

    f.write(f"4. Edge Preservation: {df['edge_preservation'].mean():.3f} ± {df['edge_preservation'].std():.3f}\n")
    f.write(
        f"   - {len(df[df['edge_preservation'] >= 0.5])} images ({len(df[df['edge_preservation'] >= 0.5]) / len(df) * 100:.1f}%) preserved fine details well\n\n")

    f.write("=" * 60 + "\n")
    f.write("MOST COMMON ISSUES:\n")
    f.write("=" * 60 + "\n\n")
    for issue, count in sorted_issues[:5]:
        pct = (count / len(df)) * 100
        f.write(f"  {issue}: {count} images ({pct:.1f}%)\n")

print("✓ Saved: analysis_plots/report_summary.txt")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
print(f"\nGenerated files:")
print(f"  - analysis_plots/metric_distributions.png")
print(f"  - analysis_plots/correlation_heatmap.png")
print(f"  - analysis_plots/quality_distribution.png")
print(f"  - analysis_plots/issue_frequency.png")
print(f"  - analysis_plots/metric_relationships.png")
print(f"  - analysis_plots/quality_class_comparison.png")
print(f"  - analysis_plots/report_summary.txt")
print(f"\nUse these visualizations and statistics in your report!")