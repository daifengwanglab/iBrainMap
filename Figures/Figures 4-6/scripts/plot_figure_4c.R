library(dplyr)
library(ComplexHeatmap)
library(circlize)

select.rows.to.plot <- function(edge, edge_p, cutoff=0.98) {
  rownames(edge) = edge$X
  edge = edge[,-1]
  rownames(edge_p) = edge_p$X
  edge_p = edge_p[,-1]
  
  # Selecting edges that is significant in atleast one column
  edge_p = edge_p[(edge_p$att_all_ave<0.05 |edge_p$att_AD_ave<0.05 | edge_p$att_SCZ_ave<0.05 |edge_p$att_dat_ave<0.05), ]
  print(nrow(edge_p))
  edge = edge[rownames(edge)%in%rownames(edge_p), ]
  
  rownames(edge) = gsub(':', ': ', rownames(edge))
  rownames(edge_p) = gsub(':', ': ', rownames(edge_p))
  
  rownames(edge) = gsub('->', '-->', rownames(edge))
  rownames(edge_p) = gsub('->', '-->', rownames(edge_p))
  
  edge_ad = edge[,1:4]
  edge_ctl = edge[,5:8]
  
  edge_ad_p = edge_p
  edge_ctl_p = edge_p
  
  row_variances_ad <- apply(edge_ad,1, var)  # Exclude the first column
  row_variances_ctl <- apply(edge_ctl,1, var)  # Exclude the first column
  
  variance_threshold_ad <- quantile(row_variances_ad, cutoff)
  variance_threshold_ctl <- quantile(row_variances_ctl,cutoff)
  
  selected_rows_ad <- edge_ad[row_variances_ad > variance_threshold_ad, ]
  selected_rows_ctl <- edge_ctl[row_variances_ctl > variance_threshold_ctl, ]
  
  edge = unique(c(rownames(selected_rows_ad), rownames(selected_rows_ctl)))
  selected_rows_ad <- edge_ad[edge, ]
  selected_rows_ad <- selected_rows_ad[complete.cases(selected_rows_ad), ]
  selected_rows_ctl <- edge_ctl[edge, ]
  selected_rows_ctl <- selected_rows_ctl[complete.cases(selected_rows_ctl), ]
  select_rows_ad_p = edge_p[edge, ]
  select_rows_ad_p <- select_rows_ad_p[complete.cases(select_rows_ad_p), ]
  select_rows_ctl_p = edge_p[edge, ]
  select_rows_ctl_p <- select_rows_ctl_p[complete.cases(select_rows_ctl_p), ]
  
  colnames(selected_rows_ad) = colnames(selected_rows_ctl) = c('Combined imp score', 'AD imp score',
                                                               'SCZ imp score', 'Data imp score')
  colnames(select_rows_ad_p) = colnames(select_rows_ctl_p) = c('Combined imp score', 'AD imp score',
                                                               'SCZ imp score', 'Data imp score')
  select_rows_ad_p = select_rows_ad_p[, c('AD imp score', 'SCZ imp score',
                                          'Data imp score', 'Combined imp score')]
  select_rows_ctl_p = select_rows_ctl_p[, c('AD imp score', 'SCZ imp score',
                                            'Data imp score', 'Combined imp score')]
  selected_rows_ad = selected_rows_ad[, c('AD imp score', 'SCZ imp score',
                                          'Data imp score', 'Combined imp score')]
  selected_rows_ctl = selected_rows_ctl[, c('AD imp score', 'SCZ imp score',
                                            'Data imp score', 'Combined imp score')]
  
  return (list("ad_edges"=selected_rows_ad, "ctrl_edges"=selected_rows_ctl, "ad_edges_p"=select_rows_ad_p, "ctrl_edges_p"=select_rows_ctl_p))
}

#---------------- Analysis ----------------------------
thresh = 0.5
ct_edge = read.csv('./Figures/Figures 4-6/data/celltype_specific_GRN_edges_matrix.csv')ct_edge = read.csv('celltype_specific_GRN_edges_matrix.csv')
ct_edge_p = read.csv('./Figures/Figures 4-6/data/celltype_specific_GRN_edges_pval.csv')

foo = data.frame(do.call('rbind', strsplit(as.character(ct_edge$X),':',fixed=TRUE)))
ct_edge = cbind(ct_edge, foo)
ct_edge_p = cbind(ct_edge_p, foo)

ct_edge_flt = ct_edge[ct_edge$X1%in%c('Micro', 'IN_SST'), ]
ct_edge_p_flt = ct_edge_p[ct_edge_p$X1%in%c('Micro', 'IN_SST'), ]

ct_edge_others = ct_edge[!(ct_edge$X1%in%c('Micro', 'IN_SST')), ]
ct_edge_p_others = ct_edge_p[!(ct_edge_p$X1%in%c('Micro', 'IN_SST')), ]

ct_dat_flt = select.rows.to.plot(ct_edge_flt, ct_edge_p_flt, cutoff=thresh)
ct_dat_others = select.rows.to.plot(ct_edge_others, ct_edge_p_others, cutoff=thresh)

selected_rows_ad = rbind(ct_dat_flt$ad_edges, ct_dat_others$ad_edges)
selected_rows_ctl = rbind(ct_dat_flt$ctrl_edges, ct_dat_others$ctrl_edges)
select_rows_ad_p = rbind(ct_dat_flt$ad_edges_p, ct_dat_others$ad_edges_p)
select_rows_ctl_p = rbind(ct_dat_flt$ctrl_edges_p, ct_dat_others$ctrl_edges_p)

edges_to_remove = c('ERG --> ELOVL7', 'Micro: IRF8 --> HK2', 'Micro: IRF8 --> CSF3R', 'SOX5 --> RORA',
                    'Micro: IRF8 --> IL17RA', 'Micro: NFATC2 --> RREB1', 'Micro: NFATC2 --> ARHGAP25',
                    'IN_PVALB: PRDM1 --> ST8SIA4', 'NFIB --> EYA4', 'PRDM1 --> TRPC4', 'REL --> SRGN',
                    'SRRM3 --> LRFN5', 'Oligo: IKZF1 --> DENND3', 'Oligo: ELF1 --> RAD51B',
                    'Oligo: IKZF1 --> NCK2', 'PRDM1 --> PLCH1', 'SOX8 --> OLIG1',
                    'ZEB1 --> COL11A1')

selected_rows_ad = selected_rows_ad[!(rownames(selected_rows_ad)%in%edges_to_remove), ]
selected_rows_ctl = selected_rows_ctl[!(rownames(selected_rows_ctl)%in%edges_to_remove), ]
select_rows_ad_p = select_rows_ad_p[!(rownames(select_rows_ad_p)%in%edges_to_remove), ]
select_rows_ctl_p = select_rows_ctl_p[!(rownames(select_rows_ctl_p)%in%edges_to_remove), ]

# PLot
ad_mat = as.matrix(selected_rows_ad)
ctrl_mat = as.matrix(selected_rows_ctl)
Breaks <- seq(0, max(c(ad_mat, ctrl_mat)+0.05), 0.05)


min_col = min(min(ad_mat), min(ctrl_mat))
max_col = max(max(ad_mat), max(ctrl_mat))
avg_col = (max_col - min_col)/2

ad_mat = ad_mat[, 1:3]
colnames(ad_mat) = c('AD prior', 'SCZ prior', 'Data-driven')

min(ad_mat)
max(ad_mat)

idx1 = which(ad_mat[, 1] > ad_mat[,3])
ad_mat1 = ad_mat[idx1, ]
ad_p_mat = select_rows_ad_p[idx1, ]

#####
ct_dat_flt = select.rows.to.plot(ct_edge_flt, ct_edge_p_flt, cutoff=0.87)
ct_dat_others = select.rows.to.plot(ct_edge_others, ct_edge_p_others, cutoff=0.98)

selected_rows_ad = rbind(ct_dat_flt$ad_edges, ct_dat_others$ad_edges)
selected_rows_ctl = rbind(ct_dat_flt$ctrl_edges, ct_dat_others$ctrl_edges)
select_rows_ad_p = rbind(ct_dat_flt$ad_edges_p, ct_dat_others$ad_edges_p)
select_rows_ctl_p = rbind(ct_dat_flt$ctrl_edges_p, ct_dat_others$ctrl_edges_p)

edges_to_remove = c('ERG --> ELOVL7', 'Micro: IRF8 --> HK2', 'Micro: IRF8 --> CSF3R', 'SOX5 --> RORA',
                    'Micro: IRF8 --> IL17RA', 'Micro: NFATC2 --> RREB1', 'Micro: NFATC2 --> ARHGAP25',
                    'IN_PVALB: PRDM1 --> ST8SIA4', 'NFIB --> EYA4', 'PRDM1 --> TRPC4', 'REL --> SRGN',
                    'SRRM3 --> LRFN5', 'Oligo: IKZF1 --> DENND3', 'Oligo: ELF1 --> RAD51B',
                    'Oligo: IKZF1 --> NCK2', 'PRDM1 --> PLCH1', 'SOX8 --> OLIG1',
                    'ZEB1 --> COL11A1')

selected_rows_ad = selected_rows_ad[!(rownames(selected_rows_ad)%in%edges_to_remove), ]
selected_rows_ctl = selected_rows_ctl[!(rownames(selected_rows_ctl)%in%edges_to_remove), ]
select_rows_ad_p = select_rows_ad_p[!(rownames(select_rows_ad_p)%in%edges_to_remove), ]
select_rows_ctl_p = select_rows_ctl_p[!(rownames(select_rows_ctl_p)%in%edges_to_remove), ]

# PLot
ad_mat = as.matrix(selected_rows_ad)
ctrl_mat = as.matrix(selected_rows_ctl)
Breaks <- seq(0, max(c(ad_mat, ctrl_mat)+0.05), 0.05)


min_col = min(min(ad_mat), min(ctrl_mat))
max_col = max(max(ad_mat), max(ctrl_mat))
avg_col = (max_col - min_col)/2

ad_mat = ad_mat[, 1:3]
colnames(ad_mat) = c('AD prior', 'SCZ prior', 'Data-driven')

min(ad_mat)
max(ad_mat)

ad_mat = rbind(ad_mat, ad_mat1)
select_rows_ad_p = rbind(select_rows_ad_p, ad_p_mat)

edges_to_remove = c('Micro: IRF8 --> SRGAP2-AS1', 'ERG --> ADGRL4', 'ERG --> PODXL', 'ERG --> ABCB1', 
                    'OPC: ZEB1 --> ZEB1-AS1', 'OPC: ZEB1 --> LHFPL3')

ad_mat = ad_mat[!(rownames(ad_mat)%in%edges_to_remove), ]
select_rows_ad_p = select_rows_ad_p[!(rownames(select_rows_ad_p)%in%edges_to_remove), ]


col = colorRamp2(c(min_col, avg_col, max_col), c("#b9d2b1", "white", "#ffd2c5"))
#pdf(file="Fig4B_edges_heatmap_v4.pdf", width=20, height=25)
png(file="fig4c_edge_improtance_heatmap.png", width=25, height=30, unit='in', res=300)
ht_opt("ROW_ANNO_PADDING" = unit(20, "in"))

ht = Heatmap(ad_mat,
             cell_fun = function(j, i, x, y, w, h, fill)
             {
               if(select_rows_ad_p[i, j] < 0.001) {
                 grid.text("***", x, y, just="centre", gp = gpar(fontsize = 35))
               } else if(select_rows_ad_p[i, j] < 0.01) {
                 grid.text("**", x, y, just="centre", gp = gpar(fontsize = 35))
               } else if(select_rows_ad_p[i, j] < 0.05) {
                 grid.text("*", x, y, just="centre", gp = gpar(fontsize = 35))
               }
             },
             cluster_columns=F, cluster_rows=F , name = "AD", column_title = 'AD samples',
             column_title_gp = gpar(fontsize = 40), 
             width = ncol(ad_mat)*unit(50, "mm"), 
             height = nrow(ad_mat)*unit(20, "mm"), col=col,
             heatmap_legend_param = list(labels_gp = gpar(fontsize = 30), direction = "horizontal"),
             column_names_gp = grid::gpar(fontsize = 40),
             row_names_gp = grid::gpar(fontsize = 40))
# Heatmap(ctrl_mat,
#         cell_fun = function(j, i, x, y, w, h, fill)
#         {
#           if(select_rows_ctl_p[i, j] <= 0.001) {
#             grid.text("***", x, y, just="centre", gp = gpar(fontsize = 28))
#           } else if(select_rows_ctl_p[i, j] <= 0.01) {
#             grid.text("**", x, y, just="centre", gp = gpar(fontsize = 28))
#           } else if(select_rows_ctl_p[i, j] < 0.05) {
#             grid.text("*", x, y, just="centre", gp = gpar(fontsize = 28))
#           }
#         },
#         cluster_columns=F, cluster_rows=T , name = "CTL", column_title = 'Control samples',
#         column_title_gp = gpar(fontsize = 40), width = ncol(ctrl_mat)*unit(30, "mm"), 
#         height = nrow(ctrl_mat)*unit(20, "mm"), col=col,
#         heatmap_legend_param = list(labels_gp = gpar(fontsize = 20), direction = "horizontal", column_gap = unit(3, "cm")),
#         column_names_gp = grid::gpar(fontsize = 32),
#         row_names_gp = grid::gpar(fontsize = 32))

draw(ht, ht_gap = unit(7, "mm"), heatmap_legend_side="top", legend_title_gp = gpar(fontsize = 40))#, annotation_name_gp= gpar(fontsize = 14))
dev.off()



