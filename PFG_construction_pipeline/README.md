This folder contains code to generate personalized functional genmic graphs (PFGs) linnking cell type interactions and gene regulatory networks.
### Step 1a: Extracting cell-cell interactions
The first step in constructing PFGs is extracting cell-cell interactions. It can be run using the following command:
```
Rscript generate_cell_cell_interactions.R "gene_expr_file" "cell_meta_data" "cell_id_column" "cell_group_column" "gene_name_file" "save" "species"
```
The arguments must be provided in the following order:
* **gene_expr_file** = This parameter should contain the path to the gene expression file (.npz file needs to be provided). It should contain cell X gene matrix.
* **cell_meta_data** = This is the metadata containing information about the cell and cell grouping. Must contain at least two columns: cell barcode column and cell group column.
* **cell_id_column** = The column name that contains the cell id information.
* **cell_group_column** = The column name that contains the cell grouping information (e.g., cell type).
* **gene_name_file** = This file must contain gene names for which the expression data is available.
* **save** = The path where the output files are to be saved.
* **species** = Optional. By default, the species is "human". Choose between "human" and "mouse".

### Step 1b: Extracting cell type regulon moules
This step uses GRNBoost2 and SCENIC to extract cell type regulon modules which are then used for constructing PFGs. It can be run using the following command:
```
python -u generate_grn_modules.py --gene_expr_file='path_to_gene_expression_file' --gene_name_file='path_to_gene_name_file' --gene_select_file='path_to_gene_select_file' --tf_list_file='path_to_TF_list_file' --cell_meta_data='path_to_cell_metadata_file' --cell_name_col='cell_id_column' --cell_group_col='cell_group_column'  --motif_path='path_to_motif_file' --feather_path='path_to_feather_file' --n_cores=10 --save='output_file_name'
```
The arguments must be provided in the following order:
* **--gene_expr_file** = This parameter should contain the path to the gene expression file (.npz file needs to be provided). It should contain cell X gene matrix.
* **--cell_meta_data** = This is the metadata containing information about the cell and cell grouping. Must contain at least two columns: cell id column and cell group column.
* **--cell_id_column** = The column name that contains the cell id information.
* **--cell_group_column** = The column name that contains the cell grouping information (e.g., cell type).
* **--gene_name_file** = This file must contain gene names for which the expression data is available.
* **--gene_select_file** = This file must contain a list of genes to be used while generating the regulon modules.
* **--tf_list_file** = This file must contain a list of TFs to be considered while generating the regulon modules.
* **--motif_path** = Path to the motif file.
* **--feather_path** = Path to the feather file.
* **--save** = The path where the output files are to be saved.
* **--n_cores** = The number of cores to be used to run this code. Default: 10.

The above code generates three files: (1) {save}_grnboost_GRN.csv, (2) {save}_celltype_rss.csv, and (3) {save}_regulon_list.csv. Files (2) and (3) are used to generate GRNs in the next step.

### Step 1c: Extracting graph node features
This step involves generating node features for each node in the graph. It can be run using the following command:
```
python -u generate_node_features.py --gene_expr_file='path_to_gene_expression_file' --gene_name_file='path_to_gene_name_file' --gene_select_file='path_to_gene_select_file' --tf_list_file='path_to_TF_list_file' --cell_meta_data='path_to_cell_metadata_file' --cell_name_col='cell_id_column' --cell_group_col='cell_group_column' --save='output_file_name'
```
The arguments must be provided in the following order:
* **--gene_expr_file** = This parameter should contain the path to the gene expression file (.npz file needs to be provided). It should contain cell X gene matrix.
* **--cell_meta_data** = This is the metadata containing information about the cell and cell grouping. Must contain at least two columns: cell id column and cell group column.
* **--cell_id_column** = The column name that contains the cell id information.
* **--cell_group_column** = The column name that contains the cell grouping information (e.g., cell type).
* **--gene_name_file** = This file must contain gene names for which the expression data is available.
* **--gene_select_file** = This file must contain a list of genes to be used while generating the regulon modules.
* **--tf_list_file** = This file must contain a list of TFs to be considered while generating the regulon modules.
* **--save** = The path where the output files are to be saved.

### Step 1d: Generating personalized functional genomic graphs (PFGs)
The following code is used to generate PFGs using the ouputs from Steps 1a-c.
```
python -u generate_PFGs.py --cci_file='path_to_cci_file' --rss_file='path_to_rss_file' --regulon_file='path_to_regulon_file' --node_feature_file='path_to_node_feature_file' --gene_filter_file='path_to_gene_filter_file' --ad_prior_file='path_to_ad_prior_file' --scz_prior_file='path_to_scz_prior_file'  --num_tf_percent=num_tf_percent --num_tg_percent=num_tg_percent --degree=diffusion_degree --save='output_file_path' --sample_id='donor_name'
```
The arguments must be provided in the following order:
* **--cci_file** = Path to the cell-cell interaction file generated from Step 1a.
* **--rss_data** = Path to the rss file generated from Step 1b.
* **--regulon_file** = Path to the regulon file generated from Step 1b.
* **--node_feature_file** = Path to the node feature file generated from Step 1c.
* **--gene_filter_file** = Path to the file containing list of genes to be excluded from the anlaysis.
* **--ad_prior_file** = Path to the file containing list of known AD genes.
* **--scz_prior_file** = Path to the file containing list of known SCZ genes.
* **--num_tf_percent** = Top percent of TFs to be picked for each cell type.
* **--num_tg_percent** = Top percent of TGs per TF.
* **--degree** = Diffusion degree that is used to select the neighbourhood of the node. Chosse between 1, 2, and 3.
* **--sample_id** = Donor name to be used while saving the graphs.
* **--save** = The path where the output files are to be saved.

The above command produces "{save}/{sample_id}_graph.pkl" file containing the PFG.
