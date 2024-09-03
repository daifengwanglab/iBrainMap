import pandas as pd
import matplotlib.pyplot as plt

imp_df = pd.read_csv('./Figures/Figures 4-6/data/ad_vs_scz_celltype_imp_scores.csv')
cf_df = pd.read_csv('./Figures/Figures 4-6/data/ad_vs_scz_celltype_fractions.csv')

imp_df = imp_df.set_index(imp_df.columns[0])
cf_df = cf_df.set_index(cf_df.columns[0])

fig = plt.figure(figsize=(12, 8))

plt.plot(imp_df.index, imp_df['ad_imp_mean'], color='#d54036', linestyle='-', linewidth=2)
plt.plot(cf_df.index, cf_df['ad_cf_mean'], color='#d54036', linestyle='-.', linewidth=2, alpha=0.5)

plt.errorbar(cf_df.index, cf_df['ad_cf_mean'], yerr=cf_df['ad_cf_std'], fmt='^', 
            color='#d54036', label='AD - Celltype Fraction', alpha=0.5, linestyle='-.')

plt.errorbar(imp_df.index, imp_df['ad_imp_mean'], yerr=imp_df['ad_imp_std'], fmt='o', 
            color='#d54036', label='AD - Celltype Importance Score', linestyle='-')
plt.legend(fontsize=15, loc='upper left')
plt.xticks(rotation=90, ha='right', fontsize = 20, color='black')
plt.yticks(fontsize=20, color='black')

fig.savefig('ad_celltype_importance.png', dpi=600)
plt.close(fig)
