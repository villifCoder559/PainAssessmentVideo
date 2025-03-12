import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

# Load data
df = pd.read_csv("tsne_whole/results.csv")

# Define the metrics
sil_cols = ['sil_score_None_None', 'sil_score_mean_(2, 3)', 'sil_score_mean_1']
wcss_cols = ['wcss_score_None_None', 'wcss_score_mean_(2, 3)', 'wcss_score_mean_1']
db_cols = ['db_score_None_None', 'db_score_mean_(2, 3)', 'db_score_mean_1']

# 1. Plot histograms for silhouette scores
plt.figure(figsize=(12, 4))
for i, col in enumerate(sil_cols):
  plt.subplot(1, 3, i+1)
  sns.histplot(df[col], kde=True)
  plt.title(col)
plt.tight_layout()
plt.show()

# 2. Scatter plot for silhouette scores (Original vs. mean_(2,3))
plt.figure(figsize=(6, 6))
sns.scatterplot(x='sil_score_None_None', y='sil_score_mean_(2, 3)', data=df)
plt.xlabel('Silhouette Score (Original)')
plt.ylabel('Silhouette Score (mean_(2,3))')
plt.title('Silhouette Score Comparison')
plt.show()

# 3. Paired t-test for silhouette scores (Original vs. mean_(2,3))
stat, p_val = ttest_rel(df['sil_score_None_None'], df['sil_score_mean_(2, 3)'])
print("Paired t-test for Silhouette Scores (Original vs. mean_(2,3)):")
print(f"t-statistic: {stat:.3f}, p-value: {p_val:.3f}")
