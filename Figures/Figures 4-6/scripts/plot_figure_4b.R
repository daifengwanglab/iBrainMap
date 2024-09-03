library(ggplot2)

data = read.csv('./Figures/Figures 4-6/data/correlation_comparison.csv')
data[data$condition=='Sleep_WeightGain_Guilt_Suicide', 'condition'] = 'S_WG_G_S'
data$phen2 =  gsub('MSSM_', '',data$phen2)
data[data$Method=='Gene Expression', 'Method'] = 'Gene\nExpression'
data[data$Method=='Gene Importance Score', 'Method'] = 'Gene\nImportance\nScore'

for (phen_label in unique(data$phen1)) {
  if (phen_label == 'AD_c15x') {
    data_flt = data[data$phen1==phen_label, ]
    # one box per variety
    ggplot(data_flt, aes(x=Method, y=corr)) + 
      geom_boxplot() +
      facet_wrap(~phen2, scale="free", ncol = 5) +
      theme(text = element_text(size = 26)) +
      labs(y= "Correlation")+
      theme(legend.position = "bottom", legend.justification = "center")+
      theme(axis.title.x=element_blank())#, axis.text.x=element_text(angle=90))
    ggsave(paste0(phen_label,'_corr_plt_01sep24.png'), width=16, height=8, dpi=300)
  }
} 
