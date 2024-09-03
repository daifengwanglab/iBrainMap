# Knowledge-Guided Graph Neural Network (KG-GNN) pipeline
This folder contains code to train our KG-GNN model for phenotype classification. It is divided into two sections.

## Training KG-GNN model from scratch
### Cross-validation training for parameter tuning
The first step in the GNN training pipeline is optimal parameter tuning. Five-fold cross-validation is used to select the optimal parameters. It can be run using the following command:
```
python -u train_cv.py --data_dir="/path_to_indiv_graph_folder" --phenotype_file='path_to_phenotype_file' --phenotype_column="class_label_column_name" --id_column="sampleID_column_name" --save "/path_to_save_param_config_file" > "/path_to_output.txt"
```
The following settings are required to be given in the above command:
* **--data_dir** = This parameter specifies the location of graphs for each individual. The files in this directory should be pickle files (.pkl format).
* **--phenotype_file** = This parameter specifies the file path for metadata containing class labels (e.g., disease phenotypes).
* **--phenotype_column** = The column name in the metadata file that contains the class label. 
* **--id_column** = The column name in the metadata file that contains the sample IDs. 
* **--save** = The path to save the optimal parameter settings into a config file.

The above code uses default parameter settings to train the model. Users can modify the following parameters to tune the model:
* **--num_classes** = The number of output classes the model needs to predict.
* **--num_heads** = The number of graph attention heads used by the model.
* **--edge_attr_comb** = Edge attribute combination. For example, "1,1,2" indicates 1 AD, 1 SCZ, and 2 data-driven heads.
* **--num_subgraph** = The number of subgraphs the model generates while training. 
* **--num_gat_nodes** = The number of hidden nodes in each GAT layer (comma separated).
* **--num_fcn_nodes** = The number of hidden nodes in each FCN layer (comma separated).
* **--need_layer_norm** = Flag to add layer normalization (Default=False).
* **--need_batch_norm** = Flag to add batch normalization (Default=False).
* **--dropout** = Dropout rate (Default=0.6)
* **--learn_rate** = Learning rate for model training (Default=0.001)
* **--step_size** = Step Size for the decay (Default=10)
* **--gamma** = Decay rate (Default=0.9)
* **--opt** = The Optimizer for training (Default='Adam')
* **--batch_size** = Batch size for training (Default=10)
* **--epochs** = Number of epochs the model trains (Default=100)
* **--stagnant** = Early stop criteria (Default=10)
* **--cv_k** = Choose k in K-fold cross-validation (Default=5)

The above code produces two files:
* /dir/perf_record.txt: Provides the performance metrics of the model
* /dir/param_config.txt: This file provides all the parameter settings used in the model. This file is necessary during testing.

The users can use the above code to train several parameters individually. We also provide **iBrainMap_KGGNN_01_tune_parameters.py**, which performs a grid search of different parameters and finds the best parameter combination. To run a grid search, the users can run the following command:
```
python -u tune_parameters.py --data_dir="/path_to_indiv_graph_folder" --phenotype_file='path_to_phenotype_file' --phenotype_column="class_label_column_name" --id_column="sampleID_column_name" --test_dir="/path_to_test_dir" --save "/path_to_save_param_config_file" > "/path_to_output.txt"
```
The following settings need to be modified in the above command:
* **--data_dir** = This parameter specifies the location of graphs for each individual. The files in this directory should be pickle files (.pkl format).
* **--phenotype_file** = This parameter specifies the file path for metadata containing class labels (e.g., disease phenotypes).
* **--phenotype_column** = The column name in the metadata file that contains the class label.
* **--id_column** = The column name in the metadata file that contains the sample IDs.
* **--test_dir** = The path to save the sample IDs for the heldout test set.
* **--save** = The path to save the optimal parameter settings into a config file.

The users can also specify the search space for the parameters by modifying the iBrainMap_KGGNN_01_tune_parameters.py file.

**Note:**  Running iBrainMap_KGGNN_01_tune_parameters.py will take a substantial amount of time as it trains more than 100 models to find the optimal settings (several days).

### Training the final model (trained on the full dataset)
Once the optimal hyper-parameters are selected, the following code trains the final model using the full dataset and optimal parameters. 
```
python -u final_train.py --data_dir="/path_to_indiv_graph_folder" --phenotype_file="path_to_phenotype_file" --phenotype_column='class_label_column_name' --id_column="sampleids_column_name" --heldout_sampleID_file="path_to_heldout_sampleids_file" --save="/path_to_model_save" --config_file="path_to_config_file" --verbose="model_name" > "/path_to_output.txt"
```
The following settings need to be modified in the above command:
* **--data_dir** = This parameter specifies the location of graphs for each individual. The files in this directory should be pickle files (.pkl format).
* **--phenotype_file** = This parameter specifies the file path for metadata containing class labels (E.g., disease phenotypes).
* **--id_column** = The column name in the metadata file that contains the sample IDs. 
* **--phenotype_column** = The column name in the metadata file that contains the class label. 
* **--config_file** = This parameter specifies the config file path containing optimal hyper-parameter settings.
* **--save** = The path to save the final model.
* **--verbose** = This parameter specifies the name to be given to the model being saved.

---

## Using pre-trained model
### Testing on the held-out or independent test dataset
After the model is trained on the complete training dataset, the following code is used to test the performance of the model on the held-out or independent dataset.
```
python -u test.py --test_data_dir="/path_to_test_indiv_graph_folder" --phenotype_file="path_to_phenotype_file" --phenotype_column='class_label_column_name' --id_column="sampleids_column_name" --config_file="path_to_config_file" --model_file='path_to_trained_model_file' --save='path_to_output_dir' --verbose="output_file_name"> "/path_to_output.txt"
```
The following settings need to be modified in the above command:
* **--test_data_dir** = This parameter specifies the location of graphs for each individual in the test set. The files in this directory should be pickle files (.pkl format).
* **--phenotype_file** = This parameter specifies the file path for metadata containing class labels (E.g., disease phenotypes).
* **--id_column** = The column name in the metadata file that contains the sample IDs. 
* **--phenotype_column** = The column name in the metadata file that contains the class label. 
* **--config_file** = This parameter specifies the config file path containing optimal hyper-parameter settings.
* **--model_file** = The path for the trained KGGNN model to be saved.
* **--save** = The path to save the final model.
* **--verbose** = This parameter specifies the name to be given to the model being saved.

### Extracting graph embeddings and edge attentions
Once the training and testing are complete, one can use this code to extract the graph embeddings and edge attentions for each individual using our trained model.
```
python -u get_emb_attn.py --test_data_dir="/path_to_test_indiv_graph_folder" --config_file="path_to_config_file" --model_file='path_to_trained_model_file' --save='path_to_output_dir' --verbose="file_name"> "/path_to_output.txt"
```
The following settings need to be modified in the above command:
* **--test_data_dir** = This parameter specifies the location of graphs for each individual in the test set. The files in this directory should be pickle files (.pkl format).
* **--config_file** = This parameter specifies the config file path containing optimal hyper-parameter settings.
* **--model_file** = The path specifies the path the trained KGGNN model.
* **--save** = The path to save the final model.
* **--verbose** = This parameter specifies the name for the files to be saved.
