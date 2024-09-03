import pandas as pd
import numpy as np
import glob
import os
import argparse
from itertools import product, combinations_with_replacement, combinations
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import train_cv as ikt
import json

def get_set_product(num_layers=[2], node_sizes=[4096, 2048, 1024, 512, 256, 128], thresh=None):    
    node_comb = []
    for nl in num_layers:
        combs=product(node_sizes, repeat=nl)
        for combination in combs:
            flag=True
            if thresh is not None:
                if (sum(combination) > thresh) or (sum(combination)==0):
                    flag = False
            if flag:
                comb = ','.join(map(str, combination))
                comb_list = list(comb.split(','))                        
                if sorted(comb_list, reverse=True):
                    node_comb.append(comb)
    return node_comb

def get_set_combination(num_layers=[2], node_size=[4096, 2048, 1024, 512, 256, 128], repeat=False):
    node_comb = []
    cntr = 0
    for nl in num_layers:
        combs=None
        if repeat:
            combs = combinations_with_replacement(node_size, nl)
        else:
            combs = combinations(node_size, nl)
        for combination in combs:
            comb = ','.join(map(str, combination))
            comb_list = list(comb.split(','))
            if sorted(comb_list, reverse=True):
                node_comb.append(comb)
            else:
                cntr += 1
    return node_comb

def split_rounds(data_dir, phen_file, phen_column, save_fldr, splits_k=5):
    """ Function to split the data into 5 different test sets """
    # Read the data files and phenotype file
    data_fls = glob.glob(data_dir + '*.pkl')
    phen = pd.read_csv(phen_file)

    # Remove NAs from phenotype if exists
    phen = phen[phen[phen_column].notna()]

    # Get donors
    donors = []
    for fl in data_fls:
        indiv = fl.split('/')[-1]
        indiv = indiv.replace('_graph.pkl', '')
        if indiv in list(phen.SubID):
            donors.append(indiv)
    
    donors = np.array(donors)
    phen = phen.loc[phen.SubID.isin(donors), :]
    
    if splits_k == 1:
        tr_id, te_idx, _, _ = train_test_split(donors, phen[phen_column],
                                              stratify=phen[phen_column], test_size=0.15)
        z = te_idx
        print(z[0:5])
        donor_list = pd.DataFrame(z)
        donor_list.columns = ['SubID']
        donor_list.to_csv(save_fldr+'test_round0.csv', index=False)
    else:
        # Split into 5 test sets
        kf = StratifiedKFold(n_splits=splits_k, random_state=42, shuffle=True)
        for i, (_, te_idx) in enumerate(kf.split(donors, phen[phen_column])):
            z = donors[te_idx]
            donor_list = pd.DataFrame(z)
            donor_list.columns = ['SubID']
            donor_list.to_csv(save_fldr+'test_round'+str(i)+'.csv', index=False)

def run_train_splits(args):
    """ Function to trian model using 5 different splits"""
    # Step 1: Create 5 different test sample files if test_dir is None
    if args.test_dir == 'None':
        args.test_dir = './'+args.phenotype_column+'/test_dir/'
    else:
        print('test dir already exist')

    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    split_rounds(args.data_dir, args.phenotype_file, args.phenotype_column, args.test_dir, splits_k=1)

    args.gamma = 0.9
    args.step_size = 10
    args.opt = 'Adam'
    args.stagnant = 7
    args.epochs = 50
    args.cv_k = 5
    args.return_results = True
    args.need_layer_norm = True
    args.need_batch_norm = True
    args.batch_size = 5 # You can also use it as a parameter [10, 15, 20, 50, 100]
    args.dropout = 0.6

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    test_files = glob.glob(args.test_dir+'*.csv')
    test_files.sort()
    print(test_files)

    # Step 2: Run 5 fold CV for various parameters
    total_num_feat = 2766
    
    attn_comb = ['0,0,8', '0,8,0', '8,0,0', '0,4,4', '4,0,4', '0,2,6', '2,0,6', '2,2,4', '3,3,3',
                 '0,2,2', '1,1,2', '2,0,2']
    gat_comb = ['1024,512,256,128', '2048,1024,512,256', '2048,2048,512,256']
    fcn_comb = ['256,128,64,32', '128,64,32', '64,32', '256,64']

    ## Remove this later ##
    for idx, fl in enumerate(test_files):
        if idx > 0:
            break
        cntr = 1
        mdl_perf, mdl_args = [], []
        for learn_rate in [0.0001]: #[0.0001, 0.001, 0.01]:
            for num_subgraph in [3, 5]:
                for include_orig in [True]:
                    for gat_idx, num_nodes in enumerate(gat_comb):
                        for num_fcn in fcn_comb:
                            for head_comb in attn_comb:
                                    print('------------------------------------------')
                                    print(f"Running Round {idx} parameter combination {cntr}")
                                    #args.data_dir = data_fldr + '/'
                                    args.num_heads = sum([int(x) for x in head_comb.split(',')])
                                    args.edge_attr_comb = head_comb
                                    args.num_gat_nodes = num_nodes
                                    args.num_fcn_nodes = num_fcn

                                    args.learn_rate = learn_rate
                                    args.heldout_sampleID_file = fl
                                    args.num_subgraph = num_subgraph
                                    args.train_with_orig = include_orig
                                    args.node_feat_index_file = None
                                    args.beta=0.3 # Add another for loop later for diff beta vals.

                                    print(args)
                                    perf_dict = ikt.train_cv(args)
                                    mdl_str = f'dr{args.dropout}|lr{learn_rate}|ngat{num_nodes}|'
                                    mdl_str += f'nfcn{num_fcn}|attn_comb{head_comb}|nheads{args.num_heads}'
                                    mdl_str += f'nsubgraphs{num_subgraph}|addOrig{include_orig}|beta03|'
                                    perf_dict['model'] = mdl_str
                                    mdl_perf.append(perf_dict)
                                    mdl_args.append(args)
                                    cntr += 1

        perf_df = pd.DataFrame.from_dict(mdl_perf)
        perf_df.to_csv(args.save + "round" + str(idx+1)+ '.csv', sep = '\t', index=False)
        
        # --- Sort DF and the list
        comb_dat = list(zip(perf_df['val_orig_bacc'], mdl_args))
        
        # Sort the combined data by the 'perf' column in descending order
        comb_dat.sort(key=lambda x: x[0], reverse=True)
        
        # Separate the sorted data back into the DataFrame and the list of strings
        mdl_args_sorted = [x[1] for x in comb_dat]
        
        # Save the best epoch configuration
        with open(args.save + 'config.txt', 'w') as f:
            json.dump(mdl_args_sorted[0].__dict__, f, indent=2)


def main():
    """ Main method """
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--data_dir', type=str, help='Path to the folder containing all individual graphs',
                        default='./demo/step2/sample_graphs/')
    parser.add_argument('--test_dir', type=str, help='Path to the heldout sampleID file',
                        default='./demo/step2/test_dir/')
    parser.add_argument('--phenotype_file', type=str, help='Path to the phenotype file',
                        default='./demo/step2/metadata.csv')
    parser.add_argument('--id_column', type=str, help='specify which column to use for ID',
                        default='sampleid')
    parser.add_argument('--phenotype_column', type=str, help='specify which column to use for labels',
                        default='labels')
    parser.add_argument('--save', type=str, help="path to save the config file with optimal combination of parameters",
                        default="./demo/step2/")
    args = parser.parse_args()
    print(args)

    run_train_splits(args)

if __name__ == '__main__':
    main()
