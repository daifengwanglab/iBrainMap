### 2024-09-01
- Add uppercase argument for `alphabetize_shape`
- Make panel labels lowercase by default
- Revised filtering for main histogram panel
- Various labeling changes

### 2024-08-30
- Add alt name argument for ancestry visualization
- Add optional exclusions for ancestry visualization
- Add verbosity argument and CLI outputs to edge discovery
- Change column display names
- Fix range for enrichment log axis
- Fix warning message for ancestry and supplement visualization
- Revise edge discovery visualization
- Revise ancestry analysis visualization

### 2024-08-16
- Tweak model structure for embedding analysis

### 2024-08-13 (1-2)
- Figure revisions
- HBCC graph embedding figure
- Reruns

### 2024-08-08
- Enhanced script usability
- HBCC graph embedding analyses
- Updated file directories and data

### 2024-08-05
- Annotation corrections
- File name consistency fixes
- Reruns
- Revert language changes

### 2024-07-02 (1-5)
- Figure revisions
- Importance score language change
- Plot revisions
- PRS language change
- Supplementary analyses

### 2024-06-07
- Group anti-centering measure for prioritization analysis

### 2024-06-06
- Add scaling methods to prioritization
- Revise prioritization sorting

### 2024-06-05
- Readd grouped prioritization comparison
- Touch up for prioritization figure

### 2024-06-03
- Add cross-disease comparison to prioritization
- Plot coloration revisions for prioritization

### 2024-06-02 (1-2)
- Add more CT groupings to prioritization analysis
- Real data for prioritization
- Revisions to cell type prioritization plot

### 2024-05-28
- Add butterfly plot for cell type prioritization comparison
- File organization, renaming, and accreditations

### 2024-05-14
- Adjust formatting

### 2024-05-07
- Distribution plot changes
- Edge discovery plot changes

### 2024-05-06
- Plot formatting

### 2024-05-05
- Density plot generation
- General plot formatting revisions

### 2024-04-12
- Frequency analyses

### 2024-04-01
- Formatting changes
- Reruns

### 2024-03-26
- Add Ting's modified scripts to repository
- Formatting changes

### 2024-03-15
- PRS formatting changes

### 2024-03-12 (1-2)
- Adjust range and layout for PRS comparison plots
- Rerun and refresh enrichments
- Various hyperparameter tweaks
- Violin plot update
- Workaround for Seaborn bug with `data` and `y` arguments in multiple plotting functions

### 2024-02-29
- Add `gene_max_num` to `plot_edge_discovery_enrichment` for enrichment significance normalization
- General QOL improvements
- Refresh edge and cross-ancestry enrichment panels

### 2024-02-22
- General bugfixes
- QOL additions
- Refresh edge discovery
- Revise cross-enrichments

### 2024-02-16
- Upload alternate figure version

### 2024-02-12 (1-3)
- Add new heatmap visualization
- Figure 3 reorganization
- Figure updates
- Many QOL additions
- New enrichment analysis
- PRS analysis

### 2024-02-11
- Add pkl generation function from source CSV
- Switch to 25p25p

### 2024-02-06 (1-2)
- Figure updates
- Filter GO terms (in progress)
- QOL and additional utility functions
- Reruns
- Revise cross-ancestry enrichment

### 2024-01-30 (1-2)
- Add new circle heatmap plotting functions
- Append corresponding GO terms with descriptions in all applications
- Calculate new enrichments
- Cross-ancestry enrichments
- Figure updates and rearrangement
- Result Recalculations

### 2024-01-19 (1-2)
- Ancestry enrichment analysis for individual and conserved edges
- Figure rearrangements
- Notebook reorganization
- QOL updates for plotting functions

### 2024-01-09
- Figure and extended figure layouts

### 2023-12-28
- Scaling changes to PRS analysis

### 2023-12-22
- Change PRS trendline to monotonic fit

### 2023-12-20
- Add covariate considerations to PRS analysis
- Add genotype metadata file
- Recalculate panels
- Usage of the `pingouin` library for statistics

### 2023-12-17
- Figure updates

### 2023-12-14
- Extend PRS and enrichment analyses to data/AD/SCZ
- Extend bottom figure 3
- Rearrange figure 3 bottom
- Scaling updates to PRS analysis
- Small labeling updates to PRS analysis

### 2023-12-13 (1-4)
- Add advanced thresholding to edge counting computation
- Add inset to individual gene enrichment
- Ease of distinction between shaded sections in individual gene enrichment
- Expand background in enrichment calculation
- Figure updates
- More shading changes to enrichment plot
- Refresh metadata
- Several small bugfixes
- Small layout changes
- Sort SCZ PRS by correlation rather than p

### 2023-12-12 (1-3)
- Figure updates
- Small visual bugfix for module analysis

### 2023-12-11 (1-2)
- Add mean capability for individual edge comparison
- Add text wrapping
- Adjust color bar min and max for module analysis
- Formatting changes
- Individual edge enrichment analysis formatting
- Trendline, statistics, and formatting for SCZ PRS analysis

### 2023-12-07 (1-6)
- Add SCZ PRS analysis saving
- Bugfix for extended ct-ct comparison
- Change PRS analysis to scatterplot and add significance
- Change bottom layout
- Change unique edge enrichment format
- Figure updates
- Formatting changes and nested subplots
- Inset edge discovery analysis
- Recompute enrichment panel

### 2023-12-06 (1-2)
- Add additional extended comparisons
- Additional graph filters
- Adjust figures
- Figure layout reformatting
- Fix file formatting error, will need to be updated on next model run
- QOL updates
- SCZ analysis lengthened

### 2023-12-05 (1-3)
- Add QOL for several utility functions
- Add `plot_labels` function for plotting figure labels
- Add averaging capabilities into `load_graph_by_id`
- Add scaling to PRS correlation visualization
- Change main subjects for figure 3
- Enhanced PRS analysis
- Figure layout revisions
- Miscellaneous bugfixes
- New figure assets and revisions
- Reorganizing of figure files
- Separate figure 3 into several unique modules
- Various formatting changes

### 2023-11-30 (1-2)
- Figure layout revisions
- Figure updates
- Implement PRS analysis
- QOL updates

### 2023-11-29 (1-2)
- Add flow icon to figure 3
- Add transparency to main figure(s)
- Figure revisions
- Fix label logic and streamline placement

### 2023-11-28 (1-2)
- Additional graph visualizations
- Additional SCZ analysis
- Figure updates
- Many figure revisions
- Many other large miscellaneous changes
- Refresh enrichments
- Several QOL improvements, including utility filters and `alphabetize_shape`
- Small language changes

### 2023-11-21
- Figure updates

### 2023-11-19 (1-3)
- Automated figure assembly
- Complete refactor of figure 3
- Figure and effect updates
- Many fixes and optimizations to existing plots
- Many utility plotting functions, including `create_subfigure_mosaic`, which creates a mosaic constrained (optional) layout using subfigures
- Spacing updates
- Three new panels
- Updated enrichment, extractions, background, and groupings

### 2023-11-15 (1-3)
- Color consistency for module analysis
- Figure updates
- Fixes and QOL for plotting functions
- Many visual alterations to figure 3
- New main subjects for figure 3
- Panel resizings

### 2023-11-09
- Figure mock-up

### 2023-11-08 (1-2)
- Adjust `plot_edge_comparison` to always have range 0-1 on both axes
- Figure updates
- Remove training edges globally
- Rerun all plots
- Update to new model formatting

### 2023-11-07 (1-2)
- Figure shading alteration

### 2023-11-06 (1-2)
- Extend individual enrichment
- Figure updates
- Individual coloration consistency
- Make panels more concise
- Module analysis criterion changes
- Shorten contrast distribution analysis and extend edge range

### 2023-11-03
- Figure amendment

### 2023-11-02 (1-2)
- Barplot of module discovery
- Deprecation of population attention
- Lineplot of per-head edge representation and enrichment to match
- Lineplot ratio adjustment
- Module analysis collation and reduction of whitespace
- `attention_stack` self loop removal

### 2023-11-01
- Figure layout updates and placeholders

### 2023-10-30 (1-2)
- Add module analysis for individual networks
- Add real enrichments with macro `compare_graphs_enrichment` among others
- Add significance to cluster enrichment
- Bugfixes
- Figure updates
- Formatting changes
- Improve depth of association calculation for edge variance heatmap
- Reorganise notebook
- Revise distribution comparison visualization

### 2023-10-27
- Figure plot and layout updates

### 2023-10-26 (1-2)
- Figure changes
- Heatmap visualization tweaks
- Heatmap visualization updates
- Linkage analysis for contrasts

### 2023-10-19
- Figure update

### 2023-10-19
- Bugfix for `string_is_synthetic` case of genes like 'C10orf90'
- Revisions for individual variance heatmap, including filtering and clustering

### 2023-10-18
- Figure updates

### 2023-10-18
- Add individual x edge heatmap
- Improve recursive synthetic graph filtering to have a parameter `limit`
- Split and update heatmap figure

### 2023-10-17
- Figure changes and refresh

### 2023-10-17
- Small figure updates

### 2023-10-17
- Adjust node sizes
- Adjust heatmap size
- Temporarily adjust heatmap scale

### 2023-10-16
- Figure updates

### 2023-10-16
- Clarified comparisons
- Miscellaneous bugfixes
- Modified aspect ratios
- New data
- New individual dosage analysis
- New visualizations for figures 3 and 4
- Other small updates

### 2023-10-13
- Figure revisions

### 2023-10-11
- Figure revisions

### 2023-09-19 (1-2)
- Additional plot input sanitization
- Figure revisions
- Fix enrichment plot scaling to start from zero
- New FREEZE and graphs with corresponding reading adjustments
- Recalculation of manual notebook parameters

### 2023-09-18 (1-2)
- Add FDR correction to SNP analysis
- Add eQTL SNP visualization to individual plots
- Calculate and read real enrichments
- Figure revisions
- New pathway enrichment for individual comparisons
- Refine SNP analysis and add utility arguments
- Swap enrichment axes

### 2023-09-15
- Add multiple Manhattan plots

### 2023-09-13
- Add various utility functions
- Implement dosage analysis

### 2023-09-12 (1-2)
- Add gene prioritization for group analyses
- Add framework for dosage correlation analysis
- Add stacked barchart visualization for `plot_prediction_confusion`
- Figure revisions
- Sanitize input and optimize processing for `compute_prediction_confusion`
- Separate combined figures into individual elements
- Small logic and formatting changes

### 2023-09-06 (1-2)
- Add legend titles
- Additional formatting for `plot_BRAAK_comparison`
- Additional utility plotting functions for legend, including `plot_remove_legend` and `plot_outside_legend`
- Consistent coloration in figure 3
- Figure updates
- Fix bug in new node attribute calculation function
- Fix zoom in `plot_individual_edge_comparison`
- Fixed text coloration in graph plotting
- Highlight outliers in `plot_individual_edge_comparison`
- More organization for notebook
- Several small formatting fixes

### 2023-09-05 (1-4)
- Add `get_node_attributes` function to sync color/shape/size changes between plots
- Figure layout and plot format updates
- Legend recalculation
- Update graph node colors to match color dictionary

### 2023-08-31 (1-2)
- Adapt `plot_contrast_curve` to additional DataFrame-type arguments
- Add batch graph loading with `get_graphs_from_sids`
- Additional arguments for many functions
- Additional population analysis plots
- Individual figure updates
- SNP analysis plot framework and dummy figure

### 2023-08-30 (1-2)
- Add general-purpose circle-style heatmap plotting with `plot_circle_heatmap`
- Add head comparison plot, `plot_head_comparison`
- Add many utility functions
- Add multi-column capability to `load_graph_by_id`
- Add supplemental figures
- Add utility function `get_attention_columns`
- Change default TF-TG recalculation timing in `concatenate_graphs` to `after`, also add configurable option
- Change filename structure
- Change plot ordering and grouping
- Change several plot styles
- Figure updates
- Fix several small bugs and unoptimal behaviors
- Flip enrichment plot
- Implement `mean` sorting method for `plot_contrast_curve`
- Manual plot adjustments
- Optimize `remove_duplicate_edges` by using hash table for duplicate edges (I am dumb)
- Revise automatic threshold for `concatenate_graphs`
- Revised individual loading structure for `graph_analysis` notebook
- Small plot revisions for readability

### 2023-08-29 (1-3)
- Add more objectives for figure 4 plots
- Added filtering for aggregate graphs
- Fixed `concatenate_graphs` bug with `threshold=True` filtering out all edges
- Increase flexibility for many (previously) BRAAK calculations in figure 4
- New figures and layouts
- New plot computations
- Place new plots into functions
- Various bugfixes and small formatting optimizations

### 2023-08-28
- Add many utility functions
- Implement new figure plots
- Label plots in notebook
- Many more arguments for a variety of functions
- Many alternative new plots and layouts
- `concatenate_graphs` bugfixes and optimizations

### 2023-08-23
- Figure updates

### 2023-08-21 (1-3)
- Add features and automatic culling to `compute_aggregate_edge_summary` and `compute_edge_summary`
- Add function to filter graph based on synthetic vertices
- Add individual plot visualizations for specific subjects
- Add new plots and computations for characterization of contrast subgroups
- Additional plotting functionality for `plot_individual_edge_comparison`, allowing for broken axes
- Fix individual edge comparison plot bug causing common edges to be excluded
- Update edge text translation

### 2023-08-17
- Additional automated filtering capabilities
- New scaling strategy, should be more consistent

### 2023-08-16
- Add common edge detection, and set as default for concatenation
- Change isolated node visualization for `plot_graph_comparison`
- Everything generally looks better
- Fix scaling issues with position calculations
- Fix vertex attribute updating upon concatenation
- General bugfixes
- Make graph computation more modular
- Make node sizes vary by type
- New custom plot scaling methods to counter auto-scaling by `graph-tool`
- New synthetic node detection

### 2023-08-15 (3)
- Add new protocol for `cull_isolated_leaves`
- Fix for gene prioritization cell gene isolation
- Fix `num_x_labels` bug
- Fix `remove_text_by_centrality` `efilt` bug
- Full runs for aggregate comparisons
- Greatly optimize `concatenate_graphs` through `_remove_duplicate_edges`
- Many small optimizations and visualizations
- Optimize `compute_edge_summary`
- Optimize `plot_aggregate_edge_summary`
- Visibility updates

### 2023-08-15 (1-2)
- Added function to compare individual and aggregate graphs
- Move figure code to notebooks for the future
- New graph comparison plots
- Visibility updates

### 2023-08-08 (1-2)
- Changed the majority of plots
- Many new plots, including enrichment and individual edge comparison
- New strategy for plot exporting
- Refactored computation section
- Update figures

### 2023-08-01 (1-3)
- Add coloration option to graph combination
- Add example plots
- Figure alternatives
- Increase compatibility of `concatenate_graphs` with visualization and utility functions
- Sankey update

### 2023-07-25
- Add graph embedding analysis

### 2023-07-24 (1-2)
- Add enhanced graph subsetting features
- Add filtering to graph computation
- Add graph embedding loading
- Added several functions facilitating common gene positioning across graph plots
- Adjust coloration and formatting
- Fix PDF line weights by using alpha
- Fix certain plot labels, especially for NPS comparison
- Make file loading more modular
- Revise figures
- Scale text properly for `visualize_graph_diffusion`
- Tuning of variable selection for data figure
- Utilize new data

### 2023-07-18
- Change file structure

### 2023-07-13 (1-5)
- Change to PDF
- First figure versions for `diffusion` and `data`

### 2023-07-12
- Reorganization
- New plot format

### 2023-07-10
- Add new plots
- Code refactor in `General Analysis`
- Many new functions for plotting and computation (e.g. `subset_graph`)
- Reformat old plots
- Switch to mosaic plotting

### < 2023-07-10
- Creation of the changelog
