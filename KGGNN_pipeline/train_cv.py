import math
import json
import os
import gc
import sys
import argparse
import time
import warnings
import random
from collections import Counter
import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import iBrainMap_data_utils as idu
from iBrainMap_KGGNN_model import KG_GNN
warnings.filterwarnings('ignore')


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def get_optimizer(opt_name, model, learn_rate, l2_norm=0.005):
    """ Function to define the optimizer """
    if opt_name =='Adagra':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learn_rate)
    if opt_name =='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
    if opt_name =='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    if opt_name =='AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)
    if opt_name =='Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=learn_rate)
    if opt_name =='ASGD':
        optimizer = torch.optim.ASGD(model.parameters(), lr=learn_rate)
    if opt_name =='LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=learn_rate)
    if opt_name =='NAdam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=learn_rate)
    if opt_name =='RAdam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=learn_rate)
    if opt_name =='RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learn_rate)
    if opt_name =='Rprop':
        optimizer = torch.optim.Rprop(model.parameters(), lr=learn_rate)
    return optimizer


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


def train_step(model, data_dl, num_classes, optimizer, scheduler, weight=None):
    """ Function to train an epoch """
    total_loss = 0
    y_pred, y_true, y_pred_score = [], [], []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = None
    if weight is None:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        weight = weight.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)


    for data in data_dl:
        data = data.to(device)
        # print('data.x', data.x.shape)
        # print('data.edge_index', data.edge_index.shape)
        # print('data.edge_attr', data.edge_attr.shape)
        # print('data.batch', data.batch)

        out, _, _ = model(data.x, data.edge_index, data.batch, data.edge_attr,
                          return_attention_weights=True)

        data.y = data.y.to(torch.int64)
        # print(data.y)
        pred_loss = criterion(out, data.y)

        total_loss += pred_loss.detach().cpu().numpy()
        y_true.extend(data.y.detach().cpu().numpy())
        pred_score = torch.softmax(out, dim=1).detach().cpu().numpy()

        if num_classes==2:
            y_pred_score.extend(pred_score[:, 1])
        elif num_classes>2:
            y_pred_score.extend(pred_score)

        y_pred.extend(out.argmax(1).detach().cpu().numpy())

        if optimizer is not None:
            optimizer.zero_grad()
            pred_loss.backward()
            optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return total_loss/len(data_dl), y_true, y_pred, np.asarray(y_pred_score)

def predict(model, data_dl, num_classes):
    """ Function to train an epoch """

    y_pred, y_true, y_pred_score = [], [], []
    # all_attn, all_graph_embed = [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for data in data_dl:
        data = data.to(device)
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


def fit(epochs, stagnant_val, model, optimizer, scheduler, tr_loader, te_loader, num_classes, weight=None):
    """ Function for trinaing the model and selecting the best performance """
    max_tr_bacc, max_val_bacc = 0, 0
    max_tr_acc, max_val_acc = 0, 0
    max_tr_auprc, max_val_auprc  = 0, 0
    max_tr_auc, max_val_auc = 0, 0

    max_val_loss = 0.0
    stagnant = 0

    best_model = None
    best_val_true, best_val_pred_score = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        if num_classes == 2:
            tr_loss, tr_true, tr_pred, tr_score = train_step(model, tr_loader, num_classes,
                                                             optimizer, scheduler)
        else:
            tr_loss, tr_true, tr_pred, tr_score = train_step(model, tr_loader, num_classes,
                                                             optimizer, scheduler, weight)

        # Validation phase
        model.eval()
        val_loss, val_true, val_pred, val_score = train_step(model, te_loader,
                                                             num_classes, None, None)

        # Performance evaluation
        tr_acc, tr_bacc, tr_auc, tr_prc = get_classification_performance(tr_true, tr_pred,
                                                                         tr_score, num_classes)
        val_acc, val_bacc, val_auc, val_prc = get_classification_performance(val_true, val_pred,
                                                                             val_score, num_classes)

        print("\n*** Epoch = %d ***"%(epoch))
        print("Train: Loss- %.4f, ACC- %.4f, BACC- %.4f, AUC- %.4f, AUPRC- %.4f"%(tr_loss, tr_acc,
                                                                                  tr_bacc, tr_auc,
                                                                                  tr_prc))
        print(skm.confusion_matrix(tr_true, tr_pred))

        print("Val: Loss- %.4f, ACC- %.4f, BACC- %.4f, AUC- %.4f, AUPRC- %.4f"%(val_loss, val_acc,
                                                                                val_bacc, val_auc,
                                                                                val_prc))
        print(skm.confusion_matrix(val_true, val_pred))

        if epoch == 0:
            max_val_loss = val_loss
            max_tr_acc, max_val_acc = tr_acc, val_acc
            max_tr_bacc, max_val_bacc = tr_bacc, val_bacc
            max_tr_auprc, max_val_auprc = tr_prc, val_prc
            max_tr_auc, max_val_auc = tr_auc, val_auc
            best_model = model
            best_val_true, best_val_pred_score = val_true, val_score
            stagnant = 0
            print(f'Stagnant after this epoch: {stagnant}')
        else:
            stagnant += 1
            if val_bacc >= max_val_bacc and tr_bacc >= val_bacc: # and (val_loss < max_val_loss):
                if (tr_bacc > 0.95) and ((tr_bacc - val_bacc) > 0.1):
                    stagnant += 1
                else:
                    max_val_loss = val_loss
                    max_tr_acc, max_val_acc = tr_acc, val_acc
                    max_tr_bacc, max_val_bacc = tr_bacc, val_bacc
                    max_tr_auprc, max_val_auprc = tr_prc, val_prc
                    max_tr_auc, max_val_auc = tr_auc, val_auc
                    best_model = model
                    best_val_true, best_val_pred_score = val_true, val_score

                    stagnant = 0
                    print(f'Stagnant after this epoch: {stagnant}')
                    continue
            print(f'Stagnant after this epoch: {stagnant}')
        if stagnant >= stagnant_val and epoch > 13:
            break

        gc.collect()

    tr_dict = {'acc': max_tr_acc, 'bacc': max_tr_bacc, 'auc': max_tr_auc, 'auprc': max_tr_auprc}

    val_dict = {'acc': max_val_acc, 'bacc': max_val_bacc, 'auc': max_val_auc,
                'auprc': max_val_auprc, 'truth': best_val_true, 'pred_scores':best_val_pred_score}

    return best_model, tr_dict, val_dict


def train_cv(args):
    """ Function to run cross-validation modeling"""
    ### ---------------------- Step 1: Load data --------------------- ###
    attr_comb = [int(num) for num in attn_head_comb.split(',')]
    if args.num_heads != sum(attr_comb):
        print('Number of heads != Sum of attention head combination'
        sys.exit(0)
    
    # Load all the graphs
    graphs, sample_ids, _ = idu.get_graphs(args.data_dir, args.phenotype_file, args.id_column,
                                           args.phenotype_column, args.edge_attr_comb)
    print('sample_ids', len(sample_ids))

    # Split the data into training and heldout based on heldout sample IDS
    hout_samples = None
    if args.heldout_sampleID_file != 'None':
        hout_samples = pd.read_csv(args.heldout_sampleID_file)
        hout_samples = hout_samples.iloc[:, 0].tolist()

    train_graphs, _, _, _ = idu.split_train_heldout_graphs(graphs, sample_ids, hout_samples)
    print(len(train_graphs))

    if hout_samples is None:
        train_samples = list(sample_ids)
    else:
        train_samples = list(set(sample_ids) - set(hout_samples))

    # Class labels
    y_train = [train_graphs[i].y.item() for i in range(len(train_graphs))]

    # Define 5 fold CV split
    kfold = StratifiedKFold(n_splits=args.cv_k, random_state=1, shuffle=True)

    tr_bacc_sc, tr_auc_sc, tr_auprc_sc = [], [], []
    val_bacc_sc, val_auc_sc, val_auprc_sc = [], [], []

    tr_sub_bacc_sc, tr_sub_auc_sc, tr_sub_auprc_sc = [], [], []
    val_sub_bacc_sc, val_sub_auc_sc, val_sub_auprc_sc = [], [], []

    for i, (tr_idx, te_idx) in enumerate(kfold.split(train_graphs, y_train)):
        st_time = time.perf_counter()
        print("********** Fold %d **********"%(i+1))

        ### -----------------Step 2: Generate sub-graphs ------------------ ###
        # Get train and test graphs
        X_train = [train_graphs[i] for i in tr_idx]
        X_test = [train_graphs[i] for i in te_idx]

        # Get train and test labels
        y_tr = [X_train[i].y.item() for i in range(len(X_train))]
        y_te = [X_test[i].y.item() for i in range(len(X_test))]

        args.num_classes = len(set(y_tr))

        # Get train and test sample ids
        id_train = [train_samples[i] for i in tr_idx]
        id_test = [train_samples[i] for i in te_idx]

        # Generate subgraphs for Training samples
        # if args.num_classes == 2:
        #     X_tr_sub, _ = idu.get_subgraph(X_train, id_train, args.num_subgraph)
        #     X_te_sub, _ = idu.get_subgraph(X_test, id_test, args.num_subgraph)
        # else:

        print('train:', Counter(y_tr))
        print('test:', Counter(y_te))
        X_tr_sub, _ = idu.get_subgraph(X_train, id_train, args.num_subgraph,
                                                  y_tr, balanced=True)
        X_te_sub, _ = idu.get_subgraph(X_test, id_test, args.num_subgraph, y_te)

        #X_tr_sub.extend(X_train)
        y_tr_sub = [X_tr_sub[i].y.item() for i in range(len(X_tr_sub))]

        # Generate subgraphs for Training samples
        #X_te_sub.extend(X_test)
        y_te_sub = [X_te_sub[i].y.item() for i in range(len(X_te_sub))]

        print('Train samples with original grpahs: ', len(X_train), Counter(y_tr))
        print('Train samples with subgrpahs: ', len(X_tr_sub), Counter(y_tr_sub))
        print('Validation samples with original grpahs: ', len(X_test), Counter(y_te))
        print('Validation samples with subgrpahs: ', len(X_te_sub), Counter(y_te_sub))

        print('Sample 1 info')
        print('data.x', X_tr_sub[0].x.shape)
        print('data.edge_index', X_tr_sub[0].edge_index.shape)
        print('data.edge_attr', X_tr_sub[0].edge_attr.shape)

        # Create data loaders
        tr_sub_loader = DataLoader(X_tr_sub, batch_size=args.batch_size, shuffle=False)
        te_sub_loader = DataLoader(X_te_sub, batch_size=args.batch_size, shuffle=False)

        tr_total_samp = len(y_tr)
        cnt_class = Counter(y_tr)
        weight = []
        for i in range(len(cnt_class)):
            weight.append(float(tr_total_samp/cnt_class[i]))
        class_wt = torch.Tensor(weight)

        ### -----------------Step 3: # Model creation------------------ ###
        # Get model parameters
        input_feat_len = X_train[0].num_node_features
        num_gat_nodes = [int(x) for x in args.num_gat_nodes.split(',')]
        num_fcn_nodes = [int(x) for x in args.num_fcn_nodes.split(',')]

        model = KG_GNN(input_feat_len, num_gat_nodes, args.num_heads, num_fcn_nodes,
                       args.num_classes, args.dropout, need_layer_norm=args.need_layer_norm,
                       need_batch_norm=args.need_layer_norm, need_attn_concat=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(model)

        # Define the loss function snd initialize optimizer
        opt = get_optimizer(args.opt, model, args.learn_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size,
                                                    gamma=args.gamma)

        ### -----------------Step 4: Train model ------------------ ###
        if args.num_classes == 2:
            mdl, perf_tr, perf_v = fit(args.epochs, args.stagnant, model, opt, scheduler,
                                       tr_sub_loader, te_sub_loader, args.num_classes)
        else:
            print('class weights: ', class_wt)
            mdl, perf_tr, perf_v = fit(args.epochs, args.stagnant, model, opt, scheduler,
                                       tr_sub_loader, te_sub_loader, args.num_classes, class_wt)


        tr_sub_bacc_sc.append(perf_tr['bacc'])
        tr_sub_auc_sc.append(perf_tr['auc'])
        tr_sub_auprc_sc.append(perf_tr['auprc'])

        val_sub_bacc_sc.append(perf_v['bacc'])
        val_sub_auc_sc.append(perf_v['auc'])
        val_sub_auprc_sc.append(perf_v['auprc'])

        ### ---------------- Step 5: Test on Whole Graphs ----------------- ###
        tr_loader = DataLoader(X_train, batch_size=1, shuffle=False)
        te_loader = DataLoader(X_test, batch_size=1, shuffle=False)

        tro_true, tro_pred, tro_pred_score = predict(mdl, tr_loader, args.num_classes)
        _, tro_bacc, tro_auc, tro_auprc = get_classification_performance(tro_true, tro_pred,
                                                                         tro_pred_score,
                                                                         args.num_classes)

        vo_true, vo_pred, vo_pred_score = predict(mdl, te_loader, args.num_classes)
        _, vo_bacc, vo_auc, vo_auprc = get_classification_performance(vo_true, vo_pred,
                                                                      vo_pred_score,
                                                                      args.num_classes)

        tr_bacc_sc.append(tro_bacc)
        tr_auc_sc.append(tro_auc)
        tr_auprc_sc.append(tro_auprc)

        val_bacc_sc.append(vo_bacc)
        val_auc_sc.append(vo_auc)
        val_auprc_sc.append(vo_auprc)

        end_time = time.perf_counter()
        gc.collect()
        print(f"Fold {i+1} complete in in {(end_time - st_time)/60.00} minutes")

    ### ----------------- Print and save k-fold CV results ---------------- ###
    # Write the performance to a file and print them
    print(str(args.cv_k)+"-fold CV complete")
    print('Train SUB BACC = ', tr_sub_bacc_sc)
    print('Train SUB AUPRC = ', tr_sub_auprc_sc)
    print('Train SUB AUC = ', tr_sub_auc_sc)
    print("")

    print('Val SUB BACC = ', val_sub_bacc_sc)
    print('Val SUB AUPRC = ', val_sub_auprc_sc)
    print('Val SUB AUC = ', val_sub_auc_sc)
    print("")

    print('Train ORIG BACC = ', tr_bacc_sc)
    print('Train ORIG AUPRC = ', tr_auprc_sc)
    print('Train ORIG AUC = ', tr_auc_sc)
    print("")

    print('VAL ORIG BACC = ', val_bacc_sc)
    print('VAL ORIG AUPRC = ', val_auprc_sc)
    print('VAL ORIG AUC = ', val_auc_sc)
    print("")

    print('Train SubGraph: BACC=%.5f +/- %.3f, AUPRC=%.5f +/- %.3f, AUC=%.5f +/- %.3f'%(np.mean(tr_sub_bacc_sc), np.std(tr_sub_bacc_sc),
                                                                                        np.mean(tr_sub_auprc_sc), np.std(tr_sub_auprc_sc),
                                                                                        np.mean(tr_sub_auc_sc), np.std(tr_sub_auc_sc)))
    print('Val SubGraph: BACC=%.5f +/- %.3f, AUPRC=%.5f +/- %.3f, AUC=%.5f +/- %.3f'%(np.mean(val_sub_bacc_sc), np.std(val_sub_bacc_sc),
                                                                                      np.mean(val_sub_auprc_sc), np.std(val_sub_auprc_sc),
                                                                                      np.mean(val_sub_auc_sc), np.std(val_sub_auc_sc)))

    print('Train ORIG Graph: BACC=%.5f +/- %.3f, AUPRC=%.5f +/- %.3f, AUC=%.5f +/- %.3f'%(np.mean(tr_bacc_sc), np.std(tr_bacc_sc),
                                                                                          np.mean(tr_auprc_sc), np.std(tr_auprc_sc),
                                                                                          np.mean(tr_auc_sc), np.std(tr_auc_sc)))
    print('Val ORIG Graph: BACC=%.5f +/- %.3f, AUPRC=%.5f +/- %.3f, AUC=%.5f +/- %.3f'%(np.mean(val_bacc_sc), np.std(val_bacc_sc),
                                                                                        np.mean(val_auprc_sc), np.std(val_auprc_sc),
                                                                                        np.mean(val_auc_sc), np.std(val_auc_sc)))

    out_file = 'results/' + args.phenotype_column + '/'+ args.phenotype_column +' _perf_record.txt'
    header_str = "data_dir,edge_attr_comb,num_gat_nodes,num_fcn_nodes,num_subgraphs,dropout,learn_rate"
    header_str += ",num_classes,Train_Sub_BACC,Train_Sub_AUPRC,Train_Sub_AUC"
    header_str += ",Val_Sub_BACC,Val_Sub_AUPRC,Val_Sub_AUC"
    header_str += ",Train_Orig_BACC,Train_Orig_AUPRC,Train_Orig_AUC"
    header_str += ",Val_Orig_BACC,Val_Orig_AUPRC,Val_Orig_AUC\n"

    if not os.path.exists(out_file):
        with open(out_file, 'w') as write_fl:
            write_fl.write(header_str)
            write_fl.close()

    write_str = args.data_dir + ',' + args.edge_attr_comb.replace(',', '|') + ','
    write_str += args.num_gat_nodes.replace(',', '|')
    write_str += ',' + args.num_fcn_nodes.replace(',', '|') + ',' + str(args.num_subgraph) + ','
    write_str += str(args.dropout) + ',' + str(args.learn_rate)
    write_str += ',' + str(args.num_classes)

    write_str += ",%.5f +/- %.3f,%.5f +/- %.3f,%.5f +/- %.3f"%(np.mean(tr_sub_bacc_sc), np.std(tr_sub_bacc_sc),
                                                               np.mean(tr_sub_auprc_sc), np.std(tr_sub_auprc_sc),
                                                               np.mean(tr_sub_auc_sc), np.std(tr_sub_auc_sc))

    write_str += ",%.5f +/- %.3f,%.5f +/- %.3f,%.5f +/- %.3f"%(np.mean(val_sub_bacc_sc), np.std(val_sub_bacc_sc),
                                                               np.mean(val_sub_auprc_sc), np.std(val_sub_auprc_sc),
                                                               np.mean(val_sub_auc_sc), np.std(val_sub_auc_sc))

    write_str += ",%.5f +/- %.3f,%.5f +/- %.3f,%.5f +/- %.3f"%(np.mean(tr_bacc_sc), np.std(tr_bacc_sc),
                                                               np.mean(tr_auprc_sc), np.std(tr_auprc_sc),
                                                               np.mean(tr_auc_sc), np.std(tr_auc_sc))

    write_str += ",%.5f +/- %.3f,%.5f +/- %.3f,%.5f +/- %.3f\n"%(np.mean(val_bacc_sc), np.std(val_bacc_sc),
                                                                 np.mean(val_auprc_sc), np.std(val_auprc_sc),
                                                                 np.mean(val_auc_sc), np.std(val_auc_sc))

    with open(out_file, 'a') as write_fl:
        write_fl.write(write_str)
        write_fl.close()

    perf_dict = {'tr_sub_bacc': np.mean(tr_sub_bacc_sc), 'tr_sub_prc': np.mean(tr_sub_auprc_sc),
                 'tr_sub_auc': np.mean(tr_sub_auc_sc), 'val_sub_bacc': np.mean(val_sub_bacc_sc),
                 'val_sub_prc': np.mean(val_sub_auprc_sc), 'val_sub_auc': np.mean(val_sub_auc_sc),
                 'tr_orig_bacc': np.mean(tr_bacc_sc), 'tr_orig_prc': np.mean(tr_auprc_sc),
                 'tr_orig_auc': np.mean(tr_auc_sc), 'val_orig_bacc': np.mean(val_bacc_sc),
                 'val_orig_prc': np.mean(val_auprc_sc), 'val_orig_auc': np.mean(val_auc_sc),
                 'tr_sub_bacc_sd': np.std(tr_sub_bacc_sc), 'tr_sub_prc_sd': np.std(tr_sub_auprc_sc),
                 'tr_sub_auc_sd': np.std(tr_sub_auc_sc), 'val_sub_bacc_sd': np.std(val_sub_bacc_sc),
                 'val_sub_prc_sd': np.std(val_sub_auprc_sc), 'val_sub_auc_sd': np.std(val_sub_auc_sc),
                 'tr_orig_bacc_sd': np.std(tr_bacc_sc), 'tr_orig_prc_sd': np.std(tr_auprc_sc),
                 'tr_orig_auc_sd': np.std(tr_auc_sc), 'val_orig_bacc_sd': np.std(val_bacc_sc),
                 'val_orig_prc_sd': np.std(val_auprc_sc), 'val_orig_auc_sd': np.std(val_auc_sc)}

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # Save the configuration file
    with open(args.save + '/param_config.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.return_results:
        return perf_dict

def main():
    """ Main method """
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--data_dir', type=str, help='Path to the folder containing all individual graphs',
                        default='./demo/step2/')
    parser.add_argument('--heldout_sampleID_file', type=str, help='Path to the heldout sampleID file',
                        default='None')
    parser.add_argument('--phenotype_file', type=str, help='Path to the phenotype file',
                        default='./demo/step2/metadata.csv')
    parser.add_argument('--id_column', type=str, help='specify which column to use for labels',
                        default='sampleid')
    parser.add_argument('--phenotype_column', type=str, help='specify which column to use for labels',
                        default='labels')

    # Graph Parameters
    parser.add_argument('--num_heads', type=int, default=8, help='Number of graph attention heads')
    parser.add_argument('--edge_attr_comb', type=str, default='2,2,4',
                        help='Edge attribute combination. Eg: 1,1,2 - 1 AD, 1 SCZ, 2 data-driven')
    parser.add_argument('--num_subgraph', type=int, help='Number of subgraphs for sampling',
                        default=3)

    # KGGNN model Hyper-parameters
    parser.add_argument('--num_classes', type=int, help='Number of classes in dataset', default=2)
    parser.add_argument('--num_gat_nodes', type=str, default='256,128',
                        help='Number of hidden nodes for each layer (comma separated)')
    parser.add_argument('--num_fcn_nodes', type=str, default='64,32',
                        help='Number of hidden nodes for each FFN layer (comma separated)')
    parser.add_argument('--need_layer_norm', type=bool, default=False,
                        help='Flag to include layer norm')
    parser.add_argument('--need_batch_norm', type=bool, default=False,
                        help='Flag to include batch norm')

    # Other parameters
    parser.add_argument('--learn_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--step_size', type=int, default=10, help='Step Size for the decay')
    parser.add_argument('--gamma', type=float, default=0.9, help='Decay rate')
    parser.add_argument('--opt', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate')

    # Model training
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--stagnant', type=int, default=10, help='Early stop criteria')
    parser.add_argument('--cv_k', type=int, default=5, help='Choose k fold cross validation')
    parser.add_argument('--train_with_orig', type=bool, default=False,
                        help='Flag to train both subgrahs and orig graphs.')


    parser.add_argument('--return_results', type=bool, default=False,
                        help='Flag to return the performance results')

    parser.add_argument('--save', type=str, default="./demo/step2/test_model/",
                        help='Path to save the model')

    args = parser.parse_args()
    print(args)
    train_cv(args)

if __name__ == '__main__':
    main()
