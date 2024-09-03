import argparse
import json
import torch
import gc
import pickle as pk
import pandas as pd
import sklearn.metrics as skm
from torch_geometric.loader import DataLoader
from iBrainMap_KGGNN_Model import KG_GNN
import iBrainMap_data_utils as idu
import numpy as np

def get_classification_performance(y_true, y_pred, y_pred_score, num_classes):
    """Function to return the acc, bacc, auprc, and auc"""
    auc = None
    if num_classes == 2:
        auc = skm.roc_auc_score(y_true, y_pred_score)
    else:
        auc = skm.roc_auc_score(y_true, y_pred_score, average="weighted", multi_class="ovr")

    auprc = skm.average_precision_score(y_true, y_pred_score, average='weighted')
    acc = skm.accuracy_score(y_true, y_pred)
    bacc = skm.balanced_accuracy_score(y_true, y_pred)
    return acc, bacc, auc, auprc

def predict(model, data_dl, num_classes):
    """ Function to train an epoch """
    with torch.no_grad():
        model.eval()
        y_pred, y_true, y_pred_score = [], [], []

        for data in data_dl:
            #data = data.to(device)
            out, _, _ = model(data.x, data.edge_index, data.batch, data.edge_attr,
                              return_attention_weights=True)
            data.y = data.y.to(torch.int64)
            y_true.extend(data.y.detach().cpu().numpy())
            pred_score = torch.softmax(out, dim=1).detach().cpu().numpy()

            if num_classes==2:
                y_pred_score.extend(pred_score[:, 1])
            elif num_classes>2:
                y_pred_score.extend(pred_score)
            y_pred.extend(out.argmax(1).detach().cpu().numpy())

    return y_true, y_pred, np.asarray(y_pred_score)

def classify_test(args):
    """ Function to train the model for each round """
    ### ---------------------- Step 1: Load data --------------------- ###
    print("### " + args.verbose + "###")
    
    if len(args.test_data_dir.split(',')) == 2:
        test_data_dir = args.test_data_dir.split(',')
        phenotype_file = args.phenotype_file.split(',')
        id_column = args.id_column.split(',')
        phenotype_column = args.phenotype_column.split(',')
        
        graphs1, sample_ids1, _ = idu.get_graphs(test_data_dir[0], phenotype_file[0], id_column[0],
                                                 phenotype_column[0], args.edge_attr_comb)
        
        graphs2, sample_ids2, _ = idu.get_graphs(test_data_dir[1], phenotype_file[1], id_column[1],
                                                 phenotype_column[1], args.edge_attr_comb)
        
        print(type(graphs1), len(graphs1))
        graphs = graphs1 + graphs2
        sample_ids = sample_ids1 + sample_ids2

        # Class labels
        y_label1 = [graphs1[i].y.item() for i in range(len(graphs1))]
        y_label2 = [graphs2[i].y.item() for i in range(len(graphs2))]
        y_label = y_label1 + y_label2

        print(len(graphs), len(sample_ids), len(y_label))
    else:
        # Load all the graphs
        graphs, sample_ids, _ = idu.get_graphs(args.test_data_dir, args.phenotype_file, args.id_column,
                                               args.phenotype_column, args.edge_attr_comb)
    
        # Class labels
        y_label = [graphs[i].y.item() for i in range(len(graphs))]

    ### -----------------Step 3: Load model------------------ ###
    # Get model parameters
    input_feat_len = graphs[0].num_node_features
    num_gat_nodes = [int(x) for x in args.num_gat_nodes.split(',')]
    num_fcn_nodes = [int(x) for x in args.num_fcn_nodes.split(',')]
    args.num_classes = len(set(y_label))

    model = KG_GNN(input_feat_len, num_gat_nodes, args.num_heads, num_fcn_nodes,
                   args.num_classes, args.dropout, need_layer_norm=args.need_layer_norm,
                   need_batch_norm=args.need_layer_norm, need_attn_concat=False)

    model.load_state_dict(torch.load(args.model_file))
    # print(model)
    
    ### ---------------- Step 5: Test on Whole Graphs ----------------- ###
    te_loader = DataLoader(graphs, batch_size=1, shuffle=False)
    teo_true, teo_pred, teo_pred_score = predict(model, te_loader, args.num_classes)
    _, teo_bacc, teo_auc, teo_auprc = get_classification_performance(teo_true, teo_pred,
                                                                     teo_pred_score,
                                                                     args.num_classes)
    print('Independent Data: BACC = %.4f, AUPRC = %.4f, AUC = %.4f'%(teo_bacc, teo_auprc, teo_auc))
    print(skm.confusion_matrix(teo_true, teo_pred))
    
    
    ### ---------------------- # Save outputs of test original graphs #########
    mdl_output = {}
    mdl_output['samples'] = sample_ids
    mdl_output['te_orig_bacc'] = teo_bacc
    mdl_output['te_orig_auprc'] = teo_auprc
    mdl_output['te_orig_auc'] = teo_auc
    mdl_output['te_truth'] = teo_true
    mdl_output['te_pred_scores'] = teo_pred_score
    
    with open(f'{args.verbose}_predictions.pkl', 'wb') as f:
        pk.dump(mdl_output, f)

# Input
def main():
    """ Main method """
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--test_data_dir', type=str, help='Path to the folder containing all individual graphs',
                        default='./test_dir')
    parser.add_argument('--phenotype_file', type=str, help='Path to the phenotype file',
                        default="./test_phen.csv")
    parser.add_argument('--phenotype_column', type=str, help='specify label column',
                        default='labels')
    parser.add_argument('--id_column', type=str, help='specify sample ID column',
                        default='sampleid')
    parser.add_argument('--config_file', type=str, help="path to KGGNN config file",
                        default='./demo/step2/config.txt')
    parser.add_argument('--model_file', type=str, help="path to pre-trained model file",
                        default='./demo/step2/psychad_AD_best_model.pth')
    parser.add_argument('--save', type=str, help="Path to save the outputs",
                        default='./demo/step2/')
    parser.add_argument('--verbose', type=str, help="Save test output file name",
                        default='output')
    inp_args = parser.parse_args()

    with open(inp_args.config_file) as json_file:
        data = json.load(json_file)
    
    for k, v in data.items():
        #print(k, v)
        parser.add_argument('--' + k, default=v)

    inp_args = parser.parse_args()
    # print(inp_args)

    classify_test(inp_args)

if __name__ == '__main__':
    main()
