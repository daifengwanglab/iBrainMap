import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from adjustText import adjust_text

warnings.filterwarnings("ignore")
print('Loaded all packages')
#%% Read required data
data = pd.read_csv('./Figures/Figures 4-6/data/ad_vs_control_cci_deg.csv')
data = data.set_index('edges')
log2_fold_change = pd.Series(data['logFC'].values, index=data.index)
p_values_series = pd.Series(data['p_values'], index=data.index)

# Convert p-values to -log10 scale
neg_log10_p_values = -np.log10(p_values_series)

# Remove infinite values caused by p-values of 0
neg_log10_p_values.replace([np.inf, -np.inf], np.nan, inplace=True)

#%% Code for plotting
colors = []
for i in range(len(log2_fold_change)):
    if log2_fold_change[i] < -1 and neg_log10_p_values[i] > 2:
        colors.append('blue')
    elif log2_fold_change[i] >= -1 and log2_fold_change[i] <= 1:
        colors.append('grey')
    elif log2_fold_change[i] < -1 and neg_log10_p_values[i] < 2:
        colors.append('grey')
    elif log2_fold_change[i] > 1 and neg_log10_p_values[i] > 2:
        colors.append('red')
    elif log2_fold_change[i] > 1 and neg_log10_p_values[i] < 2:
        colors.append('grey')   

plt.figure(figsize=(10,7))
plt.scatter(log2_fold_change, neg_log10_p_values, c=colors, alpha=0.5)
plt.xlim(-3, 4.25)

# Highlighting significant points in red
#plt.scatter(log2_fold_change[significant], neg_log10_p_values[significant], color='red')
significant = (abs(log2_fold_change) > 1) & (p_values_series < 0.005)

# Identifying the points to label
#label_left = (log2_fold_change < -1) & significant
#label_high_pval_right = ((neg_log10_p_values > 10)  & significant )| ((log2_fold_change > 1) & significant)

label_left = (log2_fold_change < -1) & (neg_log10_p_values > 4) & significant
label_high_pval_right = ((log2_fold_change > 3) & significant) | ((log2_fold_change > 1) & (neg_log10_p_values > 4) &significant)



texts = []
for gene in log2_fold_change[label_left].index:
    texts.append(plt.text(log2_fold_change[gene], neg_log10_p_values[gene], gene, fontsize=14, ha='center', va='center'))

for gene in log2_fold_change[label_high_pval_right].index:
    texts.append(plt.text(log2_fold_change[gene], neg_log10_p_values[gene], gene, fontsize=14, ha='center', va='center'))

adjust_text(texts, adjust=True)

# Adding labels and title
plt.xlabel('Log2(Fold Change)', fontsize=16)
plt.ylabel('-Log10(p-value)', fontsize=16)
#plt.title('Volcano Plot with Updated Point Colors')
plt.axhline(-np.log10(0.05), color='gray', linestyle='dashed')
plt.axvline(1, color='gray', linestyle='dashed')
plt.axvline(-1, color='gray', linestyle='dashed')
plt.savefig('ad_vs_control_cci_volcano.png', dpi=600, bbox_inches='tight')
plt.show()