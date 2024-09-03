import glob
import math
import os
import pickle as pk
import pandas as pd
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
import random


def split_train_heldout_graphs(graphs, sample_ids, heldout_samples):
    """ Function to split data into train and heldout """

    # graphs = indiv_nws.copy()
    # heldout_samples = random.sample(sample_ids, 5)

    # Create a list of indexes for heldout samples
    if heldout_samples is None:
        return graphs, None, None, None
    else:
        he_idx = [i for i, sample_id in enumerate(sample_ids) if sample_id in heldout_samples]
        # Split into train and heldout graph lists
        train_graphs = [item for i, item in enumerate(graphs) if i not in he_idx]
        train_sample_ids = [sample_ids[i] for i, _ in enumerate(graphs) if i not in he_idx]

        heldout_graphs = [item for i, item in enumerate(graphs) if i in he_idx]
        heldout_sample_ids = [sample_ids[i] for i, _ in enumerate(graphs) if i in he_idx]
    
    return train_graphs, heldout_graphs, train_sample_ids, heldout_sample_ids
    
def get_attention_heads(edge_attr, attn_head_comb, beta=0.3):
    """ Function to define attention heads based on user combination.
    For example 1, 1, 1 -> 1 AD head, 1 SCZ head, 1 Data driven head """
    
    # This is a list of combination of attention heads
    attr_comb = [int(num) for num in attn_head_comb.split(',')]
    #print(attr_comb)
    
    # Get the indexes based on beta value
    attr_idx = [0, 5]
    if beta == 0.1:
        attr_idx = [0, 5]
    elif beta == 0.3:
        attr_idx = [1, 6]
    elif beta == 0.5:
        attr_idx = [2, 7]
    elif beta == 0.7:
        attr_idx = [3, 8]
    elif beta == 0.9:
        attr_idx = [4, 9]

    # Get attention heads
    attr_mat = None

    for ix, attn_heads in enumerate(attr_comb):
        #print (ix, attn_heads)
        if attn_heads > 0:
            # 3 heads of the order AD, SCZ, data
            if ix == 0:
                repeated_column = np.tile(edge_attr[:, attr_idx[ix]], (attn_heads, 1))
            elif ix == 1:
                repeated_column = np.tile(edge_attr[:, attr_idx[ix]], (attn_heads, 1))
            else:
                repeated_column = np.tile(edge_attr[:, (edge_attr.shape[1] - 1)], (attn_heads, 1))

            #print(repeated_column)
            # Create a matrix of attention weights
            if attr_mat is None:
                attr_mat = repeated_column
            else:
                attr_mat = np.row_stack([attr_mat, repeated_column])

    attr_mat = attr_mat.T
    return torch.tensor(attr_mat)


def get_labels(phen, phen_column):
    """ Function to get labels """
    # Reduce classes for multi-class
    if phen_column == 'r01x': # BRAAK Stages
        phen['label'] = 0
        # # Grouping Technique 1
        # phen.loc[phen[phen_column] == 3, 'label'] = 1
        # phen.loc[phen[phen_column] == 4, 'label'] = 1
        # phen.loc[phen[phen_column] == 5, 'label'] = 2
        # phen.loc[phen[phen_column] == 6, 'label'] = 2
        
        # # Grouping Technique 2
        # phen = phen.loc[phen[phen_column]!=0, :]
        # print('Phen after removing 0s in R01x', phen.shape)
        # phen.loc[phen[phen_column] > 3, 'label'] = 1

        # Grouping Technique 3
        phen = phen.loc[phen[phen_column]!=0, :]
        phen.loc[phen[phen_column] == 4, 'label'] = 1
        phen.loc[phen[phen_column] == 5, 'label'] = 1
        phen.loc[phen[phen_column] == 6, 'label'] = 2
    elif phen_column in ['r03x', 'r04x']: # CERAD Score
        # # Grouping Technique 1   
        # phen['label'] = phen[phen_column]
        # phen['label'] -= 1
        
        # Grouping Technique 2
        phen = phen.loc[phen['label']!=4, :]
        phen['label'] = phen[phen_column]
        phen['label'] -= 1
    elif phen_column == 'r05x': # CDR
        phen['label'] = 0
        phen.loc[phen[phen_column] == 0.5, 'label'] = 1 
        phen.loc[phen[phen_column] == 1, 'label'] = 1
        phen.loc[phen[phen_column] >= 2, 'label'] = 2
    elif phen_column == 'r07x': # ApoE Phenotype
        phen['label'] = np.nan
        phen.loc[phen[phen_column] == 'ApoE4_0', 'label'] = 0
        phen.loc[phen[phen_column] == 'ApoE4_1', 'label'] = 1
        phen.loc[phen[phen_column] == 'ApoE4_2', 'label'] = 2
    elif phen_column in ['c15x', 'c16x', 'c02x', 'c03x']:
        phen['label'] = 0
        phen.loc[phen[phen_column] == 'AD', 'label'] = 1
    else:
        phen['label'] = phen[phen_column]
    return phen

def get_subgraph(graphs, sample_ids, num_subgraph, labels, balanced=False, save=None):
    """ Function to get subgraphs """
    all_graphs, all_sample_id = [], []

    # Define a function to remove isolated nodes from the graphs
    transform = T.Compose([T.remove_isolated_nodes.RemoveIsolatedNodes()])
    
    if balanced:
        label_val_counts =pd.Series(labels).value_counts()
        subgraphs_per_gr = {}
        for i in range(len(label_val_counts)):
            if (label_val_counts.max() % label_val_counts[i] == 0):
               subgraphs_per_gr[i] = num_subgraph
            #elif (label_val_counts.min() % label_val_counts[i] == 0):
            #    subgraphs_per_gr[i] = (int(label_val_counts.max()/label_val_counts[i])+1)*3
            #    subgraphs_per_gr[i] = subgraphs_per_gr[i]*num_subgraph
            else:
                subgraphs_per_gr[i] = int(label_val_counts.max()/label_val_counts[i])
                rem = label_val_counts.max()%label_val_counts[i]
                if float(rem/label_val_counts[i]) > 0.7:
                    subgraphs_per_gr[i] = subgraphs_per_gr[i] + 1
                subgraphs_per_gr[i] = subgraphs_per_gr[i]*num_subgraph

        for s_id in sample_ids:
            br = labels[sample_ids.index(s_id)]
            NUM_SUBGRAPHS = subgraphs_per_gr[br]
            data = graphs[sample_ids.index(s_id)]
            data.x = data.x.contiguous()
            data.edge_attr = data.edge_attr.contiguous()
            data.edge_index =  data.edge_index.contiguous()
            data.y = data.y.contiguous()
            
            loader = NeighborLoader(data, num_neighbors=[10] * 100,
                                    batch_size=math.ceil(data.x.shape[0]/NUM_SUBGRAPHS))
            
            if len(loader) > NUM_SUBGRAPHS:
                print (f"Length greater than {NUM_SUBGRAPHS} detected ...")
                print ("Skipping this graph ...")
                continue
            for d in loader:
                all_graphs.append(d)
                all_sample_id.append(s_id)
    else:        
        for i, data in enumerate(graphs):
            data.x = data.x.contiguous()
            data.edge_attr = data.edge_attr.contiguous()
            data.edge_index =  data.edge_index.contiguous()
            data.y = data.y.contiguous()

            # Create subgraph
            loader = NeighborLoader(data, num_neighbors=[10] * 100,
                                    batch_size=math.ceil(data.x.shape[0]/num_subgraph))

            # Check if there are more subgraphs than the required number
            if len(loader) > num_subgraph:
                print (f"Length greater than {num_subgraph} detected ...")
                print ("Skipping this graph ...")
                continue

            # Remove any isolated nodes
            for d in loader:
                try:
                    all_graphs.append(transform(d))
                    if sample_ids is not None:
                        all_sample_id.append(sample_ids[i])
                except:
                    continue

    dataset = []
    for item in all_graphs:
        data = Data(x=item.x, edge_index=item.edge_index,
                    edge_attr=item.edge_attr, y=torch.tensor(item.y), pos=None)
        dataset.append(data)

    if save is not None:
        with open(os.path.join(save, 'sampled_graphs.pickle'), 'wb') as f:
            pk.dump(dataset, f)

        with open(os.path.join(save, 'sampled_graphs_IDs.pickle'), 'wb') as f:
            pk.dump(all_sample_id, f)

    return dataset, all_sample_id

# def get_subgraph(graphs, sample_ids, num_subgraph, save=None):
#     """ Function to get subgraphs """
#     all_graphs, all_sample_id = [], []

#     # Define a function to remove isolated nodes from the graphs
#     transform = T.Compose([T.remove_isolated_nodes.RemoveIsolatedNodes()])

#     # Iterate through the graphs to get subgraphs'
#     for i, data in enumerate(graphs):
#         data.x = data.x.contiguous()
#         data.edge_attr = data.edge_attr.contiguous()
#         data.edge_index =  data.edge_index.contiguous()
#         data.y = data.y.contiguous()

#         # Create subgraph
#         loader = NeighborLoader(data, num_neighbors=[10] * 100,
#                                 batch_size=math.ceil(data.x.shape[0]/num_subgraph))

#         # Check if there are more subgraphs than the required number
#         if len(loader) > num_subgraph:
#             print (f"Length greater than {num_subgraph} detected ...")
#             print ("Skipping this graph ...")
#             continue

#         # Remove any isolated nodes
#         for d in loader:
#             all_graphs.append(transform(d))
#             if sample_ids is not None:
#                 all_sample_id.append(sample_ids[i])'''''

#     dataset = []
#     for item in all_graphs:
#         data = Data(x=item.x, edge_index=item.edge_index,
#                     edge_attr=item.edge_attr, y=torch.tensor(item.y), pos=None)
#         dataset.append(data)

#     if save is not None:
#         with open(os.path.join(save, 'sampled_graphs.pickle'), 'wb') as f:
#             pk.dump(dataset, f)

#         with open(os.path.join(save, 'sampled_graphs_IDs.pickle'), 'wb') as f:
#             pk.dump(all_sample_id, f)

#     return dataset, all_sample_id


def get_graphs(data_dir, phen_file, id_column, phen_column, attn_head_comb):
    """ Function to get the graphs for training """
    # Read the data files and phenotype file
    data_fls = glob.glob(data_dir + '*.pkl')
    phen = pd.read_csv(phen_file)

    # Remove NAs from phenotype if exists
    phen = phen[phen[phen_column].notna()]
    phen = get_labels(phen, phen_column)
    #print(phen.shape, len(data_fls))
    
    # Remove samples of age group < 60
    #phen = phen.loc[phen.Age >= 50, :]

    all_graphs, all_samples, all_labels = [], [], []
    random.seed(444)
    cntr = 0
    for fl in data_fls:
        indiv = fl.split('/')[-1]
        indiv = indiv.replace('_graph.pkl', '')

        #Check if the individual ID is in phenotype ID
        if indiv in list(phen[id_column]):
            #print('entered')
            try:
                indiv_data = pd.read_pickle(fl)
                temp = indiv_data['graph_28ct']

                # Get edge attribute combination
                edge_attr = get_attention_heads(temp['edge_attr'].numpy(), attn_head_comb)
                temp['edge_attr'] = edge_attr

                # Get the class label
                y = phen['label'][phen[id_column] == indiv].values[0]
                temp.y = torch.tensor(y)

                # Append to the list of graphs
                all_graphs.append(temp)
                all_samples.append(indiv)
                all_labels.append(y)

            except Exception as e:
                print("***ERROR***" + indiv)
                print(e, indiv)
                continue
    return all_graphs, all_samples, all_labels

def get_graphs_with_no_labels(data_dir, attn_head_comb):
    """ Function to get the graphs for training """
    # Read the data files and phenotype file
    data_fls = glob.glob(data_dir + '*.pkl')

    all_graphs, all_samples = [], []
    random.seed(444)

    for fl in data_fls:
        indiv = fl.split('/')[-1]
        indiv = indiv.replace('_graph.pkl', '')

        try:
            indiv_data = pd.read_pickle(fl)
            temp = indiv_data['graph_28ct']

            # Get edge attribute combination
            edge_attr = get_attention_heads(temp['edge_attr'].numpy(), attn_head_comb)
            temp['edge_attr'] = edge_attr

            # Append to the list of graphs
            all_graphs.append(temp)
            all_samples.append(indiv)

        except Exception as e:
            print("***ERROR***" + indiv)
            #print(e, indiv)
            continue
    return all_graphs, all_samples
