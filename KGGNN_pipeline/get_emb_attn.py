import argparse
import json
import torch
import gc
import pickle as pk
import pandas as pd
import sklearn.metrics as skm
from torch_geometric.loader import DataLoader
from iBrainMap_KGGNN_model import KG_GNN
import iBrainMap_data_utils as idu
import numpy as np


def get_embed_attn(model, data_dl, num_classes):
    """ Function to train an epoch """
    with torch.no_grad():
        model.eval()

        all_attn, all_graph_embed, all_edge_idx = [], [], []
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for data in data_dl:
            #data = data.to(device)
            out, attn, embed = model(data.x, data.edge_index, data.batch, data.edge_attr,
                                     return_attention_weights=True)

            all_graph_embed.append(embed.detach().cpu().numpy())
            all_attn.append(attn[1].detach().cpu().numpy())
            all_edge_idx.append(attn[0].detach().cpu().numpy())

    return all_graph_embed, all_attn, all_edge_idx

def get_test_features(args):
    """ Function to get the graph embeddings and attentions from the trained model """
    print("### " + args.verbose + " ###")

    ### ---------------------- Step 1: Load data --------------------- ###
    # Load all the graphs
    graphs, sample_ids = idu.get_graphs_with_no_labels(args.test_data_dir, args.edge_attr_comb)

    ### -----------------Step 3: Load model------------------ ###
    # Get model parameters
    input_feat_len = graphs[0].num_node_features
    num_gat_nodes = [int(x) for x in args.num_gat_nodes.split(',')]
    num_fcn_nodes = [int(x) for x in args.num_fcn_nodes.split(',')]
    args.num_classes = 2

    model = KG_GNN(input_feat_len, num_gat_nodes, args.num_heads, num_fcn_nodes, args.num_classes,
                   args.dropout, need_layer_norm=args.need_layer_norm,
                   need_batch_norm=args.need_layer_norm, need_attn_concat=False, is_training=False)

    model.load_state_dict(torch.load(args.model_file))
    # print(model)
    
    ### ---------------- Step 5: Test on Whole Graphs ----------------- ###
    te_loader = DataLoader(graphs, batch_size=1, shuffle=False)
    teo_ge, teo_attn, teo_edge_idx = get_embed_attn(model, te_loader, args.num_classes)

    ### ---------------------- # Save outputs of test original graphs #########
    mdl_output = {}
    mdl_output['attn_scores'] = teo_attn
    mdl_output['graph_embeddings'] = teo_ge
    mdl_output['samples'] = sample_ids
    mdl_output['edge_idx'] = teo_edge_idx

    with open(f'{args.save}+{args.verbose}_embed_attn.pkl', 'wb') as f:
        pk.dump(mdl_output, f)

# Input
def main():
    """ Main method """
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_dir', type=str, help='Path to the folder containing all individual graphs',
                        default='./demo/step2/sample_graphs/')
    parser.add_argument('--save', type=str, help="Path to save the outputs",
                        default='./demo/step2/')  
    parser.add_argument('--verbose', type=str, help="Save output file name",
                        default='output')
    parser.add_argument('--config_file', type=str, help="path to KGGNN config file",
                        default='./demo/step2/config.txt')
    parser.add_argument('--model_file', type=str, help="path to the pre-trained model file",
                        default='./demo/step2/psychad_AD_best_model.pth')
    inp_args = parser.parse_args()

    with open(inp_args.config_file) as json_file:
        data = json.load(json_file)
    
    for k, v in data.items():
        #print(k, v)
        parser.add_argument('--' + k, default=v)

    inp_args = parser.parse_args()
    # print(inp_args)

    get_test_features(inp_args)

if __name__ == '__main__':
    main()
