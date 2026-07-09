import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories
os.makedirs("results/EXP010/analysis", exist_ok=True)
os.makedirs("results/EXP010/plots", exist_ok=True)

# Load data
df = pd.read_csv("results/EXP010/semantic_annotations_master.csv")
cat_col = 'ANNOTATION: Primary Category'
feat_col = 'ANNOTATION: Explicit Alternative Feasible?'
ctx_col = 'ANNOTATION: Context Window Needed'

# Clean data (ensure strings)
df[cat_col] = df[cat_col].fillna("Missing").astype(str)
df[feat_col] = df[feat_col].fillna("Unknown").astype(str)
df[ctx_col] = df[ctx_col].fillna("Unknown").astype(str)

sns.set_theme(style="whitegrid")

# 1. Primary Category Distribution
cat_counts = df[cat_col].value_counts().reset_index()
cat_counts.columns = ['Primary Category', 'Frequency']
cat_counts['Percentage'] = (cat_counts['Frequency'] / len(df) * 100).round(2)
cat_counts.to_csv("results/EXP010/analysis/primary_category_distribution.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=cat_counts, y='Primary Category', x='Frequency', palette="viridis", hue='Primary Category', legend=False)
plt.title('Primary Capability Missing (EXP010)')
plt.xlabel('Number of Quotes (Total N=200)')
plt.ylabel('')
plt.tight_layout()
plt.savefig("results/EXP010/plots/primary_category_distribution.png", dpi=300)
plt.close()

# 2. Feasibility Breakdown & Capability Recovery Potential
feas_crosstab = pd.crosstab(df[cat_col], df[feat_col], margins=True)
feas_crosstab.to_csv("results/EXP010/analysis/feasibility_crosstab.csv")

# Create stacked bar chart for feasibility
feas_plot_data = pd.crosstab(df[cat_col], df[feat_col], normalize='index') * 100
feas_plot_data = feas_plot_data.drop("Missing", errors='ignore') # ignore if missing category exists

plt.figure(figsize=(10, 6))
feas_plot_data.plot(kind='barh', stacked=True, colormap='coolwarm', figsize=(10, 6))
plt.title('Explicit Alternative Feasible? by Category')
plt.xlabel('Percentage within Category (%)')
plt.ylabel('Primary Category')
plt.legend(title='Feasible?', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("results/EXP010/plots/feasibility_breakdown.png", dpi=300)
plt.close()

# Capability Recovery Potential Table
recovery_df = df.groupby(cat_col)[feat_col].value_counts(normalize=True).unstack().fillna(0) * 100
recovery_df = recovery_df.round(1)
recovery_df = recovery_df.reset_index()
recovery_df.to_csv("results/EXP010/analysis/capability_recovery_potential.csv", index=False)

# 3. Context Window Heatmap
ctx_crosstab = pd.crosstab(df[cat_col], df[ctx_col])
ctx_crosstab = ctx_crosstab.drop("Missing", errors='ignore')
# Reorder context columns logically if they exist
ctx_order = ['Local', 'Nearby', 'Scene', 'Conversation', 'Unknown']
actual_order = [c for c in ctx_order if c in ctx_crosstab.columns] + [c for c in ctx_crosstab.columns if c not in ctx_order]
ctx_crosstab = ctx_crosstab[actual_order]

ctx_crosstab.to_csv("results/EXP010/analysis/context_window_crosstab.csv")

plt.figure(figsize=(10, 6))
sns.heatmap(ctx_crosstab, annot=True, cmap="YlGnBu", fmt="d", cbar_kws={'label': 'Count'})
plt.title('Context Window Required by Failure Category')
plt.xlabel('Context Window')
plt.ylabel('Primary Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/EXP010/plots/context_window_heatmap.png", dpi=300)
plt.close()

print("Analysis and plotting complete. Results saved in results/EXP010/analysis/ and results/EXP010/plots/")
