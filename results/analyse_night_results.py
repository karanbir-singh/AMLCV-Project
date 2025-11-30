"""
Comprehensive analysis script for night-to-day evaluation results
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
df = pd.read_csv('night_to_day_results.csv')
summary = pd.read_csv('night_to_day_summary.csv')

print(f"Total images evaluated: {len(df)}")
print(f"Columns: {df.columns.tolist()}\n")

# ============================================================
# 1. OVERALL STATISTICS
# ============================================================
print("=" * 60)
print("OVERALL STATISTICS")
print("=" * 60)

metrics = ['object_consistency', 'object_visibility_improvement', 'ssim', 
           'brightness_increase', 'warm_shift', 'saturation_increase', 
           'edge_preservation', 'shadow_removal', 'overexposure_score', 'lpips']

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
    """Classify overall quality based on multiple metrics for night-to-day transformation"""
    score = 0

    # Object consistency (0-20 points)
    if row['object_consistency'] >= 0.9:
        score += 20
    elif row['object_consistency'] >= 0.7:
        score += 15
    elif row['object_consistency'] >= 0.5:
        score += 8
    elif row['object_consistency'] >= 0.3:
        score += 3

    # Object visibility improvement (0-15 points)
    if row['object_visibility_improvement'] >= 2.0:
        score += 15
    elif row['object_visibility_improvement'] >= 1.5:
        score += 12
    elif row['object_visibility_improvement'] >= 1.0:
        score += 8
    elif row['object_visibility_improvement'] >= 0.5:
        score += 4

    # SSIM structure preservation (0-15 points)
    if row['ssim'] >= 0.45:
        score += 15
    elif row['ssim'] >= 0.4:
        score += 12
    elif row['ssim'] >= 0.35:
        score += 8
    elif row['ssim'] >= 0.3:
        score += 4

    # Brightness increase (0-15 points)
    if 50 <= row['brightness_increase'] <= 75:
        score += 15
    elif 40 <= row['brightness_increase'] <= 85:
        score += 10
    elif 30 <= row['brightness_increase'] <= 95:
        score += 5

    # Warm shift (color temperature) (0-10 points)
    if -2.0 <= row['warm_shift'] <= 0.5:
        score += 10
    elif -3.0 <= row['warm_shift'] <= 1.5:
        score += 6
    elif -4.0 <= row['warm_shift'] <= 2.5:
        score += 3

    # Edge preservation (0-10 points)
    if row['edge_preservation'] >= 0.04:
        score += 10
    elif row['edge_preservation'] >= 0.03:
        score += 7
    elif row['edge_preservation'] >= 0.02:
        score += 4
    elif row['edge_preservation'] >= 0.01:
        score += 2

    # Overexposure control (0-10 points)
    if row['overexposure_score'] >= 0.7:
        score += 10
    elif row['overexposure_score'] >= 0.6:
        score += 7
    elif row['overexposure_score'] >= 0.5:
        score += 4
    elif row['overexposure_score'] >= 0.4:
        score += 2

    # Saturation handling (0-5 points)
    if -70 <= row['saturation_increase'] <= -40:
        score += 5
    elif -85 <= row['saturation_increase'] <= -25:
        score += 3

    # Classify
    if score >= 85:
        return 'Excellent'
    elif score >= 70:
        return 'Good'
    elif score >= 50:
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

corr_metrics = ['object_consistency', 'object_visibility_improvement', 'ssim', 
                'brightness_increase', 'warm_shift', 'saturation_increase',
                'edge_preservation', 'overexposure_score', 'lpips']
corr_matrix = df[corr_metrics].corr()

print("\nKey Correlations:")
print(f"SSIM ↔ Edge Preservation: {corr_matrix.loc['ssim', 'edge_preservation']:.3f}")
print(f"SSIM ↔ Object Consistency: {corr_matrix.loc['ssim', 'object_consistency']:.3f}")
print(f"Object Visibility ↔ Object Consistency: {corr_matrix.loc['object_visibility_improvement', 'object_consistency']:.3f}")
print(f"Brightness Increase ↔ Overexposure: {corr_matrix.loc['brightness_increase', 'overexposure_score']:.3f}")
print(f"Warm Shift ↔ Saturation: {corr_matrix.loc['warm_shift', 'saturation_increase']:.3f}")
print(f"LPIPS ↔ SSIM: {corr_matrix.loc['lpips', 'ssim']:.3f}")

# ============================================================
# 5. BEST AND WORST PERFORMERS
# ============================================================
print("\n" + "=" * 60)
print("BEST AND WORST PERFORMERS")
print("=" * 60)

# Create comprehensive composite score for night-to-day transformation
df['composite_score'] = (
    df['object_consistency'] * 0.20 +
    df['object_visibility_improvement'] * 0.15 +
    df['ssim'] * 0.15 +
    df['edge_preservation'] * 0.12 +
    df['overexposure_score'] * 0.10 +
    (1 - abs(df['brightness_increase'] - 62.5) / 100) * 0.10 +  # Optimal ~62.5
    (1 - abs(df['warm_shift'] + 0.8) / 10) * 0.08 +  # Optimal ~-0.8
    (1 - abs(df['saturation_increase'] + 75) / 200) * 0.05 +  # Optimal ~-75
    (1 - df['lpips']) * 0.05  # Lower LPIPS is better
)

print("\nTop 5 Best Transformations:")
best = df.nlargest(5, 'composite_score')[['night_image_name', 'object_consistency',
                                          'object_visibility_improvement', 'ssim', 
                                          'edge_preservation', 'overexposure_score', 'issues']]
for idx, row in best.iterrows():
    print(f"\n  {row['night_image_name']}")
    print(f"    Obj: {row['object_consistency']:.3f}, Vis: {row['object_visibility_improvement']:.3f}, "
          f"SSIM: {row['ssim']:.3f}, Edge: {row['edge_preservation']:.3f}, Overexp: {row['overexposure_score']:.3f}")
    print(f"    Issues: {row['issues']}")

print("\n\nBottom 5 Worst Transformations:")
worst = df.nsmallest(5, 'composite_score')[['night_image_name', 'object_consistency',
                                            'object_visibility_improvement', 'ssim', 
                                            'edge_preservation', 'overexposure_score', 'issues']]
for idx, row in worst.iterrows():
    print(f"\n  {row['night_image_name']}")
    print(f"    Obj: {row['object_consistency']:.3f}, Vis: {row['object_visibility_improvement']:.3f}, "
          f"SSIM: {row['ssim']:.3f}, Edge: {row['edge_preservation']:.3f}, Overexp: {row['overexposure_score']:.3f}")
    print(f"    Issues: {row['issues']}")

# ============================================================
# 6. VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Create output directory
Path('night_to_day_analysis_plots').mkdir(exist_ok=True)

# Plot 1: Distribution of all metrics
fig, axes = plt.subplots(4, 3, figsize=(18, 15))
fig.suptitle('Distribution of All Evaluation Metrics (Night-to-Day)', fontsize=16, fontweight='bold')

metrics_to_plot = [
    ('object_consistency', 'Object Consistency', (0, 1)),
    ('object_visibility_improvement', 'Object Visibility Improvement', (0, 3)),
    ('ssim', 'SSIM (Structure)', (0.2, 0.6)),
    ('brightness_increase', 'Brightness Increase', (30, 90)),
    ('warm_shift', 'Warm Shift (Color Temp)', (-5, 5)),
    ('saturation_increase', 'Saturation Increase', (-120, 50)),
    ('edge_preservation', 'Edge Preservation', (0, 0.1)),
    ('overexposure_score', 'Overexposure Score', (0, 1)),
    ('lpips', 'LPIPS (Perceptual Distance)', (0.3, 0.7)),
    ('shadow_removal', 'Shadow Removal', (0, 1))  # Assuming binary or scaled
]

for idx, (metric, title, xlim) in enumerate(metrics_to_plot):
    if idx < 12:  # Ensure we don't exceed subplot count
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

# Remove empty subplots if any
for idx in range(len(metrics_to_plot), 12):
    axes[idx // 3, idx % 3].set_visible(False)

plt.tight_layout()
plt.savefig('night_to_day_analysis_plots/metric_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: night_to_day_analysis_plots/metric_distributions.png")
plt.close()

# Plot 2: Comprehensive correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Comprehensive Metric Correlation Matrix (Night-to-Day)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('night_to_day_analysis_plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: night_to_day_analysis_plots/correlation_heatmap.png")
plt.close()

# Plot 3: Quality classification pie chart
fig, ax = plt.subplots(figsize=(8, 8))
colors = {'Excellent': '#2ecc71', 'Good': '#3498db', 'Fair': '#f39c12', 'Poor': '#e74c3c'}
quality_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%',
                    colors=[colors[q] for q in quality_counts.index],
                    startangle=90)
ax.set_ylabel('')
ax.set_title('Overall Quality Distribution (Night-to-Day)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('night_to_day_analysis_plots/quality_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: night_to_day_analysis_plots/quality_distribution.png")
plt.close()

# Plot 4: Issue frequency bar chart
if issue_counts:
    fig, ax = plt.subplots(figsize=(12, 8))
    issues_df = pd.DataFrame(list(issue_counts.items()), columns=['Issue', 'Count'])
    issues_df = issues_df.sort_values('Count', ascending=True)

    ax.barh(issues_df['Issue'], issues_df['Count'], color='coral', edgecolor='black')
    ax.set_xlabel('Number of Images', fontsize=12)
    ax.set_title('Frequency of Detected Issues (Night-to-Day)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add percentage labels
    for idx, row in issues_df.iterrows():
        pct = (row['Count'] / len(df)) * 100
        ax.text(row['Count'], idx, f" {pct:.1f}%", va='center')

    plt.tight_layout()
    plt.savefig('night_to_day_analysis_plots/issue_frequency.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: night_to_day_analysis_plots/issue_frequency.png")
    plt.close()

# Plot 5: Advanced scatter plots for key relationships
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Key Metric Relationships (Night-to-Day)', fontsize=16, fontweight='bold')

# Object Consistency vs Visibility Improvement
axes[0, 0].scatter(df['object_consistency'], df['object_visibility_improvement'], alpha=0.6, s=40)
axes[0, 0].set_xlabel('Object Consistency')
axes[0, 0].set_ylabel('Object Visibility Improvement')
axes[0, 0].set_title(f'Correlation: {corr_matrix.loc["object_consistency", "object_visibility_improvement"]:.3f}')
axes[0, 0].grid(True, alpha=0.3)

# Brightness Increase vs Overexposure Score
axes[0, 1].scatter(df['brightness_increase'], df['overexposure_score'], alpha=0.6, s=40, color='orange')
axes[0, 1].set_xlabel('Brightness Increase')
axes[0, 1].set_ylabel('Overexposure Score')
axes[0, 1].set_title(f'Correlation: {corr_matrix.loc["brightness_increase", "overexposure_score"]:.3f}')
axes[0, 1].grid(True, alpha=0.3)

# Warm Shift vs Saturation Increase
axes[0, 2].scatter(df['warm_shift'], df['saturation_increase'], alpha=0.6, s=40, color='green')
axes[0, 2].set_xlabel('Warm Shift')
axes[0, 2].set_ylabel('Saturation Increase')
axes[0, 2].set_title(f'Correlation: {corr_matrix.loc["warm_shift", "saturation_increase"]:.3f}')
axes[0, 2].grid(True, alpha=0.3)

# SSIM vs Edge Preservation
axes[1, 0].scatter(df['ssim'], df['edge_preservation'], alpha=0.6, s=40, color='red')
axes[1, 0].set_xlabel('SSIM (Structure Preservation)')
axes[1, 0].set_ylabel('Edge Preservation')
axes[1, 0].set_title(f'Correlation: {corr_matrix.loc["ssim", "edge_preservation"]:.3f}')
axes[1, 0].grid(True, alpha=0.3)

# LPIPS vs SSIM
axes[1, 1].scatter(df['lpips'], df['ssim'], alpha=0.6, s=40, color='purple')
axes[1, 1].set_xlabel('LPIPS (Perceptual Distance)')
axes[1, 1].set_ylabel('SSIM (Structure Preservation)')
axes[1, 1].set_title(f'Correlation: {corr_matrix.loc["lpips", "ssim"]:.3f}')
axes[1, 1].grid(True, alpha=0.3)

# Object Consistency vs SSIM
axes[1, 2].scatter(df['object_consistency'], df['ssim'], alpha=0.6, s=40, color='brown')
axes[1, 2].set_xlabel('Object Consistency')
axes[1, 2].set_ylabel('SSIM')
axes[1, 2].set_title(f'Correlation: {corr_matrix.loc["object_consistency", "ssim"]:.3f}')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('night_to_day_analysis_plots/metric_relationships.png', dpi=300, bbox_inches='tight')
print("✓ Saved: night_to_day_analysis_plots/metric_relationships.png")
plt.close()

# Plot 6: Comprehensive box plots comparing all metrics across quality classes
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Metrics by Quality Class (Night-to-Day)', fontsize=16, fontweight='bold')

key_metrics = ['object_consistency', 'object_visibility_improvement', 'ssim', 
               'brightness_increase', 'warm_shift', 'edge_preservation', 
               'overexposure_score', 'lpips', 'saturation_increase']

for idx, metric in enumerate(key_metrics):
    if idx < 9:  # Ensure we don't exceed subplot count
        ax = axes[idx // 3, idx % 3]
        df.boxplot(column=metric, by='quality_class', ax=ax)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Quality Class')
        ax.set_ylabel(metric.replace('_', ' ').title())
        plt.sca(ax)
        plt.xticks(rotation=45)

plt.suptitle('Metrics by Quality Class (Night-to-Day)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('night_to_day_analysis_plots/quality_class_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: night_to_day_analysis_plots/quality_class_comparison.png")
plt.close()

# ============================================================
# 7. EXPORT COMPREHENSIVE SUMMARY FOR REPORT
# ============================================================
print("\n" + "=" * 60)
print("EXPORTING REPORT SUMMARY")
print("=" * 60)

with open('night_to_day_analysis_plots/report_summary.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("NIGHT-TO-DAY TRANSFORMATION EVALUATION - COMPREHENSIVE ANALYSIS\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Total Images Evaluated: {len(df)}\n\n")

    f.write("QUALITY DISTRIBUTION:\n")
    for quality, count in quality_counts.items():
        pct = (count / len(df)) * 100
        f.write(f"  {quality:12s}: {count:4d} ({pct:5.1f}%)\n")

    f.write("\n" + "=" * 60 + "\n")
    f.write("COMPREHENSIVE KEY FINDINGS:\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"1. OBJECT CONSISTENCY: {df['object_consistency'].mean():.3f} ± {df['object_consistency'].std():.3f}\n")
    f.write(
        f"   - {len(df[df['object_consistency'] >= 0.9])} images ({len(df[df['object_consistency'] >= 0.9]) / len(df) * 100:.1f}%) achieved excellent consistency (≥0.9)\n")
    f.write(
        f"   - {len(df[df['object_consistency'] < 0.7])} images ({len(df[df['object_consistency'] < 0.7]) / len(df) * 100:.1f}%) lost significant objects (<0.7)\n\n")

    f.write(f"2. OBJECT VISIBILITY IMPROVEMENT: {df['object_visibility_improvement'].mean():.3f} ± {df['object_visibility_improvement'].std():.3f}\n")
    f.write(
        f"   - {len(df[df['object_visibility_improvement'] >= 2.0])} images ({len(df[df['object_visibility_improvement'] >= 2.0]) / len(df) * 100:.1f}%) achieved excellent visibility improvement (≥2.0)\n")
    f.write(
        f"   - {len(df[df['object_visibility_improvement'] >= 1.5])} images ({len(df[df['object_visibility_improvement'] >= 1.5]) / len(df) * 100:.1f}%) achieved good visibility improvement (≥1.5)\n\n")

    f.write(f"3. STRUCTURAL SIMILARITY (SSIM): {df['ssim'].mean():.3f} ± {df['ssim'].std():.3f}\n")
    f.write(
        f"   - Average preservation indicates {'good' if df['ssim'].mean() >= 0.4 else 'moderate'} structural consistency\n")
    f.write(
        f"   - {len(df[df['ssim'] < 0.35])} images ({len(df[df['ssim'] < 0.35]) / len(df) * 100:.1f}%) show significant structural changes\n\n")

    f.write(f"4. BRIGHTNESS INCREASE: {df['brightness_increase'].mean():.1f} ± {df['brightness_increase'].std():.1f}\n")
    f.write(
        f"   - Optimal range (50-75): {len(df[(df['brightness_increase'] >= 50) & (df['brightness_increase'] <= 75)])} images ({len(df[(df['brightness_increase'] >= 50) & (df['brightness_increase'] <= 75)]) / len(df) * 100:.1f}%)\n")
    f.write(
        f"   - Acceptable range (40-85): {len(df[(df['brightness_increase'] >= 40) & (df['brightness_increase'] <= 85)])} images ({len(df[(df['brightness_increase'] >= 40) & (df['brightness_increase'] <= 85)]) / len(df) * 100:.1f}%)\n\n")

    f.write(f"5. COLOR TEMPERATURE (WARM SHIFT): {df['warm_shift'].mean():.3f} ± {df['warm_shift'].std():.3f}\n")
    f.write(
        f"   - Optimal range (-2.0 to 0.5): {len(df[(df['warm_shift'] >= -2.0) & (df['warm_shift'] <= 0.5)])} images ({len(df[(df['warm_shift'] >= -2.0) & (df['warm_shift'] <= 0.5)]) / len(df) * 100:.1f}%)\n\n")

    f.write(f"6. EDGE PRESERVATION: {df['edge_preservation'].mean():.3f} ± {df['edge_preservation'].std():.3f}\n")
    f.write(
        f"   - {len(df[df['edge_preservation'] >= 0.04])} images ({len(df[df['edge_preservation'] >= 0.04]) / len(df) * 100:.1f}%) preserved fine details excellently\n")
    f.write(
        f"   - {len(df[df['edge_preservation'] >= 0.03])} images ({len(df[df['edge_preservation'] >= 0.03]) / len(df) * 100:.1f}%) preserved fine details well\n\n")

    f.write(f"7. OVEREXPOSURE CONTROL: {df['overexposure_score'].mean():.3f} ± {df['overexposure_score'].std():.3f}\n")
    f.write(
        f"   - {len(df[df['overexposure_score'] >= 0.7])} images ({len(df[df['overexposure_score'] >= 0.7]) / len(df) * 100:.1f}%) maintained excellent exposure balance\n")
    f.write(
        f"   - {len(df[df['overexposure_score'] >= 0.6])} images ({len(df[df['overexposure_score'] >= 0.6]) / len(df) * 100:.1f}%) maintained good exposure balance\n\n")

    f.write(f"8. SATURATION HANDLING: {df['saturation_increase'].mean():.1f} ± {df['saturation_increase'].std():.1f}\n")
    f.write(
        f"   - Optimal range (-70 to -40): {len(df[(df['saturation_increase'] >= -70) & (df['saturation_increase'] <= -40)])} images ({len(df[(df['saturation_increase'] >= -70) & (df['saturation_increase'] <= -40)]) / len(df) * 100:.1f}%)\n\n")

    f.write(f"9. PERCEPTUAL QUALITY (LPIPS): {df['lpips'].mean():.3f} ± {df['lpips'].std():.3f}\n")
    f.write(
        f"   - Lower values indicate better perceptual quality\n")
    f.write(
        f"   - {len(df[df['lpips'] < 0.5])} images ({len(df[df['lpips'] < 0.5]) / len(df) * 100:.1f}%) achieved good perceptual similarity (<0.5)\n\n")

    f.write("=" * 60 + "\n")
    f.write("KEY CORRELATIONS:\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Object Consistency ↔ Visibility Improvement: {corr_matrix.loc['object_consistency', 'object_visibility_improvement']:.3f}\n")
    f.write(f"Brightness Increase ↔ Overexposure: {corr_matrix.loc['brightness_increase', 'overexposure_score']:.3f}\n")
    f.write(f"Warm Shift ↔ Saturation: {corr_matrix.loc['warm_shift', 'saturation_increase']:.3f}\n")
    f.write(f"SSIM ↔ Edge Preservation: {corr_matrix.loc['ssim', 'edge_preservation']:.3f}\n")
    f.write(f"LPIPS ↔ SSIM: {corr_matrix.loc['lpips', 'ssim']:.3f}\n\n")

    f.write("=" * 60 + "\n")
    f.write("MOST COMMON ISSUES:\n")
    f.write("=" * 60 + "\n\n")
    for issue, count in sorted_issues[:8]:
        pct = (count / len(df)) * 100
        f.write(f"  {issue}: {count} images ({pct:.1f}%)\n")

print("✓ Saved: night_to_day_analysis_plots/report_summary.txt")

print("\n" + "=" * 60)
print("COMPREHENSIVE ANALYSIS COMPLETE!")
print("=" * 60)
print(f"\nGenerated files:")
print(f"  - night_to_day_analysis_plots/metric_distributions.png")
print(f"  - night_to_day_analysis_plots/correlation_heatmap.png")
print(f"  - night_to_day_analysis_plots/quality_distribution.png")
print(f"  - night_to_day_analysis_plots/issue_frequency.png")
print(f"  - night_to_day_analysis_plots/metric_relationships.png")
print(f"  - night_to_day_analysis_plots/quality_class_comparison.png")
print(f"  - night_to_day_analysis_plots/report_summary.txt")
print(f"\nUse these comprehensive visualizations and statistics in your report!")