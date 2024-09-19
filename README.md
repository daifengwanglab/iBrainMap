# iBrainMap

Precision medicine for brain diseases faces many challenges, including understanding the heterogeneity of disease phenotypes. Such heterogeneity can be attributed to the variations in cellular and molecular mechanisms across individuals. However, personalized mechanisms remain elusive, especially at the single-cell level. To address this, the PsychAD project generated population-level single-nucleus RNA-seq data for 1,494 human brains with over 6.3 million nuclei covering diverse clinical phenotypes and neuropsychiatric symptoms (NPSs) in Alzheimer’s disease (AD). Leveraging this data, we analyzed personalized single-cell functional genomics involving cell type interactions and gene regulatory networks. In particular, we developed a knowledge-guided graph neural network model to learn latent representations of functional genomics (embeddings) and quantify importance scores of cell types, genes, and network edges for each individual. Our embeddings improved phenotype classifications and revealed potentially novel subtypes and population trajectories for AD progression, cognitive impairment, and NPSs. Our importance scores prioritized personalized functional genomic information and showed significant differences in cell type-specific regulatory mechanisms across various phenotypes. Such information also allowed us to further identify subpopulation-level biological pathways, including ancestry for AD. Finally, we associated genetic variants with cell type gene regulatory network changes across individuals, i.e., gene regulatory QTLs (grQTLs), providing novel functional genomic insights compared to existing QTLs. We further validated our results using external cohorts. Our analyses have been summarized in an open-source computational framework named iBrainMap for general personalized studies. All results are also available as a personalized functional genomic atlas for AD.

![Figure1](https://github.com/user-attachments/assets/3785a4c6-c8e9-4bb9-b50a-3a915197537d)

## System Requirements
This code has been tested on Ubuntu 18.04 with the following dependencies:

Python version: 3.9.16

- Python packages:\
numpy=1.23.5\
scipy=1.10.1\
pandas=1.5.3\
scanpy=1.9.3\
torch=2.0.0+cu117\
torch-geometric=2.3.0\
torch-scatter=2.1.1\
torch-sparse=0.6.17+pt20cu117\
numba=0.56.4\
scikit-learn=1.3.0\
matplotlib=3.7.1\
seaborn=0.12.2\
networkx=3.1\
pyscenic=\
arboreto=0.1.6

R version: 4.4.1

- R packages:\
reticulate=1.6-5\
parallel=\
Seurat=\
Matrix=\
reshape2=

## Installation

Download the iBrainMap code from github by downloading the respository zip directly or using git commands and navigate to the repository:

```
git clone https://github.com/daifengwanglab/iBrainMap.git
cd iBrainMap
```
This takes around 2 minutes depending on the network speed.

## Usage
### Constructing Personalized Functional Genomic Graphs (PFGs)
We provide details of the PFG construction with an example. PFG construction involves four sub-steps:
#### a. Inferring cell type interactions
<ins>Step 1a</ins> infers cell type interactions and requires snRNA-seq data in .npz format, cell metadata information, and the gene_ids of the genes with expression data being used.
``` Command Line
# Step 1a: Inferring cell type interactions
Rscript generate_cell_cell_interactions.R "./demo/step1/sample_3_gexpr.npz" "./demo/step1/sample_3_cell_meta.csv" "barcodekey" "subclass" "./demo/step1/sample_3_gene_ids.csv" "./demo/step1/sample_3" "human"
```
The above code produces "./demo/step1/sample_3_comm_nw_weight.csv" file that contains information about the cell type interactions and its weights.

#### b. Extracting cell type regulon modules
<ins>Step 1b</ins> infers cell type gene regulon modules and requires snRNA-seq data in .npz format, cell metadata information, gene IDs, and a list of genes and TFs to be considered. Additionally, we need a motif file and a feather file, which can be downloaded from https://resources.aertslab.org/cistarget/motif_collections/ and https://resources.aertslab.org/cistarget/ respectively. For more details, please refer to the pySCENIC documentation (https://pyscenic.readthedocs.io/en/latest/index.html).
``` Command Line
# Step 1b: Extracting cell type regulon modules
python -u generate_grn_modules.py --gene_expr_file='./demo/step1/sample_3_gexpr.npz' --gene_name_file='./demo/step1/sample_3_gene_ids.csv' --gene_select_file='./demo/step1/genes_to_use.csv' --tf_list_file='./demo/step1/tf_list.txt' --cell_meta_data='./demo/step1/sample_3_cell_meta.csv' --cell_name_col='barcodekey' --cell_group_col='subclass'  --motif_path='path_to_motif_file' --feather_path='path_to_feather_file' --n_cores=10 --save='./demo/step1/sample_3'
```
The above code produces three files: (1) ./demo/step1/sample_3_grnboost_GRN.csv, (2) ./demo/step1/sample_3_celltype_rss.csv, and (3) ./demo/step1/sample_3_regulon_list.csv.

#### c. Extracting graph node features
<ins>Step 1c</ins> extracts node features for each node in our PFG (cell types, TFs, and genes). This requires snRNA-seq data in .npz format, cell metadata information, gene IDs, and a list of genes and TFs to be considered. Additionally, we need a motif file and a feather file, which can be downloaded from https://resources.aertslab.org/cistarget/motif_collections/ and https://resources.aertslab.org/cistarget/ respectively. For more details, please refer to the pySCENIC documentation (https://pyscenic.readthedocs.io/en/latest/index.html).
``` Command Line
# Step 1c: Extracting graph node features
python -u generate_node_features.py --gene_expr_file='./demo/step1/sample_3_gexpr.npz' --gene_name_file='./demo/step1/sample_3_gene_ids.csv' --gene_select_file='./demo/step1/genes_to_use.csv' --tf_list_file='./demo/step1/tf_list.txt' --cell_meta_data='./demo/step1/sample_3_cell_meta.csv' --cell_name_col='barcodekey' --cell_group_col='subclass' --save='./demo/step1/sample_3'
```
This creates ./demo/step1/sample_3_NodeFeatures.pkl file containing node features.

#### d. Generating PFGs
Finally, <ins>Step 1d</ins> combines the outputs of all the above steps to generate PFG. It needs the cci file from Step 1a, rss and regulon files from Step 1b, and node feature file from Step 1c. Additionaly, the code needs gene filter file (list of genes to avoid), known AD and SCZ gene files. 
``` Command Line
# Step 1d: Generating PFGs
python -u generate_PFGs.py --cci_file='./demo/step1/sample_3_comm_nw_weight.csv' --rss_file='./demo/step1/sample_3_celltype_rss.csv' --regulon_file='./demo/step1/sample_3_regulon_list.csv' --node_feature_file='./demo/step1/sample_3_NodeFeatures.pkl' --gene_filter_file='./demo/step1/genes_to_use.txt' --ad_prior_file='./demo/step1/known_AD_genes.csv' --scz_prior_file='./demo/step1/known_SCZ_genes.csv'  --num_tf_percent=10 --num_tg_percent=10 --degree=2 --save='./demo/step1/' --sample_id='sample_3'
```
This creates ./demo/step1/sample_3_graph.pkl file containing the donor's PFG.

<br>
Complete details on the arguments and the files required for all the above-mentioned steps are present <a href= "https://github.com/daifengwanglab/iBrainMap/tree/main/PFG_construction_pipeline">here</a>.

---

### Training Knowledge-Guided Graph Neural Network (KG-GNN) model from scratch
KG-GNN model training involves two steps and each step is explained with a demo example below. The files for running the demo are available in the demo/step2/ folder. The users must first extract the demo graphs from "demo/step2/sample_graphs.zip" file.

#### a. Cross-validation training for parameter tuning
<ins>Step 2a</ins> is mainly used for parameter tuning. The code performs 5-fold cross-validation to check the robustness of each set of parameters.
``` Command Line
# Step 2a: Cross-validation training for parameter tuning
python -u train_cv.py --data_dir="demo/step2/sample_graphs/" --phenotype_file='demo/step2/metadata.csv' --id_column="sampleid" --phenotype_column="labels" --save "./demo/step2/config.txt" > "cv_train_logs.txt"
```
The above code uses default parameter setting for training the model. Details on tuning the parameters and additional arguments that can be provided with the above code are available <a href= "https://github.com/daifengwanglab/iBrainMap/tree/main/KGGNN_pipeline">here</a>.

#### b. Training the final model (trained on the full dataset)
<ins>Step 2b</ins> uses the config file created from step 2a for training a final model using full training dataset.
``` Command Line
# Step 2b: Training the final model
python -u final_train.py --data_dir="demo/step2/sample_graphs/" --phenotype_file="demo/step2/metadata.csv" --phenotype_column='labels' --id_column="sampleid" --heldout_sampleID_file="None" --config_file="./demo/step2/config.txt" --save="./demo/step2/test/" --verbose="test_model" > "final_train_logs.txt"
```
---

### Using the pre-trained model
We also provide a pre-trained model for AD phenotype classification and extract graph embeddings and edge importance scores. We divided this process into two steps.
#### a. Testing on new samples
The users can use the pre-trained model from <ins>Step 2b</ins> or download PsychAD AD pre-trained model (trained on donors from PsychAD dataset) to extract the predictions and performance of the model on new donors. Here, we are using the donors from the training phase as the test samples. For PsychAD pre-trained model, the users must download files from here[[zenodo link]] and provide the path to the "psychad_AD_best_model.pth" file.
``` Command Line
# Step 3a: Testing on new samples
python -u test.py --test_data_dir="./demo/step2/sample_graphs/" --phenotype_file="./demo/step2/metadata.csv" --phenotype_column='labels' --id_column="sampleid" --config_file="./demo/step2/config.txt" --model_file='psychad_AD_best_model.pth' --save='./demo/step2/' --verbose='test_output' > "test_logs.txt"
```

#### b. Extracting graph embeddings and edge importance scores
Along with phenotype predictions, the users can also extract graph embeddings and edge importance scores of donors for further downstream analysis.
``` Command Line
# Step 3a: Extracting graph embeddings and edge importance scores
python -u get_emb_attn.py --test_data_dir="./demo/step2/sample_graphs/" --config_file="./demo/step2/config.txt" --model_file='psychad_AD_best_model.pth' --save='./demo/step2/' --verbose="test_output" > "emb_attn_logs.txt"
```

