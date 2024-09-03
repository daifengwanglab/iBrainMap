library(ggplot2)
library(viridis)
library(dplyr)

data = read.csv(file='./Figures/Figures 4-6/data/tf_degs.csv')
size_range = c(2, 8)

# colour blind palette from http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/
cbPalette <- c("#E69F00", "#56B4E9",
               "#009E73", "#F0E442", "#0072B2",
               "#D55E00", "#CC79A7", "#DF69A7")
ggplot(data, aes(x = cellgroup, y = gene, colour = log_fold_change, 
                 size = inv_fdr, group = stage)) +
  geom_point() +
  scale_color_gradientn(colours = viridis::viridis(10)) +
  # scale_color_gradientn(colours = terrain.colors(20)) +
  # scale_color_gradient2(low = "blue", high = "red") +
  
  scale_size_continuous(range = size_range) +
  facet_grid(. ~ stage, space = "free", scales ="free", switch = "y")  +
  # scale_x_discrete(position = "right") +
  labs(y = 'Transcription Factors',
       colour = 'LogFC',
       size = '-log10(p.value)',
       x = "cellgroup"
  ) +
  theme_bw(base_size = 15) +
  theme(
    legend.text = element_text(size = 16),
    axis.text.x = element_text(size = 16, angle=90),
    axis.title.x = element_text(colour = "gray6"),
    axis.text.y = element_text(size = 16, vjust = 0.5),
    legend.title = element_text(size = 16),
    panel.spacing = unit(0.1, "lines"),
    strip.background = element_rect(fill = NA),
    strip.text = element_text(size = 18, colour = "gray6") #,
    # strip.text.y.left = element_text(angle = 0)
  )
ggsave('top_DEGs_dot_plot.png', width=60, height=40, units='cm', dpi=300)
